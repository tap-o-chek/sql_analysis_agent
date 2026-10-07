"""Детерминированные проверки без LLM.

inspect_filters — смотрит в реальные данные по текстовым фильтрам запроса: есть ли
  такое значение и нет ли похожих написаний, которые фильтр теряет.
check_result — проверки качества результата: пустота, NULL, дубли, отрицательные значения, выбросы.
"""

import difflib
import re
import sqlite3

import pandas as pd
import sqlglot
from sqlglot import exp

MAX_DISTINCT = 100  # колонки с большим числом уникальных значений (адреса, названия) не проверяем


def _norm(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"^г\.?\s+", "", s.strip())  # «г. Москва» → «москва»
    return re.sub(r"[^\w]+", " ", s).strip()


def _similar(value: str, pattern: str) -> bool:
    a, b = _norm(value), _norm(pattern)
    if not b or re.findall(r"\d+", a) != re.findall(r"\d+", b):  # 'КЗН-1' и 'КЗН-2' — разные коды
        return False
    return a == b or (len(b) >= 3 and (b in a or a in b)) or difflib.SequenceMatcher(None, a, b).ratio() >= 0.8


def _resolver(tree: exp.Expression, columns: dict[str, dict[str, str]]):
    """Возвращает функцию: колонка из запроса → имя таблицы (с учётом алиасов), или None."""
    aliases = {}
    for t in tree.find_all(exp.Table):
        aliases[t.alias_or_name] = t.name
        aliases[t.name] = t.name

    def resolve(col: exp.Column) -> str | None:
        if col.table:
            table = aliases.get(col.table)
            return table if table and col.name in columns.get(table, {}) else None
        owners = {t for t in aliases.values() if col.name in columns.get(t, {})}
        return owners.pop() if len(owners) == 1 else None

    return resolve


def _text_filters(tree: exp.Expression, columns: dict[str, dict[str, str]]):
    """Находит фильтры вида col = 'x', col LIKE 'x', col IN ('x', 'y') по текстовым колонкам.
    Возвращает [(таблица, колонка, оператор, [литералы])]."""
    resolve = _resolver(tree, columns)
    found = []
    for node in tree.find_all(exp.EQ, exp.Like, exp.In):
        if isinstance(node, exp.In):
            col, literals, op = node.this, node.expressions, "IN"
        else:
            col, lit = node.this, node.expression
            if isinstance(col, exp.Literal):  # 'x' = col
                col, lit = lit, col
            literals, op = [lit], ("LIKE" if isinstance(node, exp.Like) else "=")
        if not isinstance(col, exp.Column) or not literals:
            continue
        if not all(isinstance(l, exp.Literal) and l.is_string for l in literals):
            continue
        table = resolve(col)
        if table is None or columns[table][col.name] != "TEXT":
            continue
        found.append((table, col.name, op, [l.this for l in literals]))
    return found


def inspect_filters(conn: sqlite3.Connection, sql: str, columns: dict[str, dict[str, str]]) -> list[str]:
    """Сверяет текстовые фильтры запроса с реальными значениями в базе. Возвращает список замечаний."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError:
        return []

    issues, seen = [], set()
    for table, col, op, literals in _text_filters(tree, columns):
        key = (table, col, op, tuple(literals))
        if key in seen:
            continue
        seen.add(key)

        counts = conn.execute(
            f'SELECT "{col}", COUNT(*) FROM "{table}" GROUP BY 1 ORDER BY 2 DESC LIMIT {MAX_DISTINCT + 1}'
        ).fetchall()
        if len(counts) > MAX_DISTINCT:
            continue
        counts = {v: n for v, n in counts if v is not None}

        if op == "LIKE":
            matched = {r[0] for r in conn.execute(
                f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" LIKE ?', (literals[0],))}
            patterns = [literals[0].strip("%_")]
        else:
            matched = {v for v in counts if v in literals}
            patterns = literals

        similar = [v for v in counts if v not in matched and any(_similar(v, p) for p in patterns)]
        where = f'{table}.{col} {op} {", ".join(repr(l) for l in literals)}'
        if similar:
            listed = ", ".join(f"'{v}' ({counts[v]} стр.)" for v in similar)
            caught = ", ".join(f"'{v}'" for v in matched) or "ни одного значения"
            issues.append(
                f"Фильтр {where} захватывает {caught}, но в данных есть похожие написания: {listed}. "
                f"Скорее всего, это то же самое — перечисли все варианты через IN (...)."
            )
        elif not matched:
            top = ", ".join(f"'{v}'" for v in list(counts)[:20])
            issues.append(f"Фильтр {where} не совпадает ни с одним значением. Значения в колонке: {top}.")
    return issues + _date_bound_issues(conn, tree, columns)


DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}")


def _date_bound_issues(conn: sqlite3.Connection, tree: exp.Expression, columns) -> list[str]:
    """Колонка с датой и временем, сравниваемая как строка с датой без времени:
    '2025-12-31 10:00' > '2025-12-31', поэтому BETWEEN ... AND '2025-12-31' теряет последний день."""
    resolve = _resolver(tree, columns)
    comparisons = []  # (колонка, литерал, оператор)
    for node in tree.find_all(exp.Between, exp.LTE, exp.GT, exp.EQ):
        if isinstance(node, exp.Between):
            comparisons.append((node.this, node.args.get("high"), "BETWEEN"))
        else:
            comparisons.append((node.this, node.expression, node.key.upper()))

    issues, seen = [], set()
    for col, lit, op in comparisons:
        if not (isinstance(col, exp.Column) and isinstance(lit, exp.Literal) and DATE_ONLY.match(lit.this)):
            continue
        table = resolve(col)
        if table is None or (table, col.name) in seen:
            continue
        sample = conn.execute(f'SELECT "{col.name}" FROM "{table}" WHERE "{col.name}" IS NOT NULL LIMIT 1').fetchone()
        if not (sample and isinstance(sample[0], str) and DATETIME.match(sample[0])):
            continue
        seen.add((table, col.name))
        issues.append(
            f"{table}.{col.name} хранит дату со временем (например, '{sample[0]}'), а сравнивается через {op} "
            f"с датой без времени '{lit.this}'. Строки сравниваются посимвольно, поэтому записи за "
            f"{lit.this} после 00:00 обрабатываются неверно. Используй date({col.name}) "
            f"или полуоткрытый интервал: {col.name} >= 'начало' AND {col.name} < 'день после конца'."
        )
    return issues


NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?:[.,](\d+))?\s*(тыс|млн|млрд)?", re.I)
MULTIPLIER = {"тыс": 1e3, "млн": 1e6, "млрд": 1e9}


def unsupported_numbers(answer: str, df: pd.DataFrame, context: str = "") -> list[str]:
    """Числа из ответа, которых нет ни в результате, ни в вопросе/SQL (context).
    Простая безреференсная проверка «не выдумала ли LLM цифры». Числа до 10 не проверяем
    (порядковые номера, «топ-3», коды маршрутов), округления до 5% допускаем."""
    allowed = {float(len(df))}
    for col in df.select_dtypes("number").columns:
        allowed.update(abs(float(v)) for v in df[col].dropna())
    for m in NUMBER.finditer(context):
        allowed.add(float(re.sub(r"\s", "", m.group(1))))

    def supported(x: float) -> bool:
        return any(x == a or (a and abs(x - a) / abs(a) <= 0.05) for a in allowed)

    bad = []
    for m in NUMBER.finditer(answer):
        value = float(re.sub(r"\s", "", m.group(1)) + (f".{m.group(2)}" if m.group(2) else ""))
        value *= MULTIPLIER.get((m.group(3) or "").lower(), 1)
        if value > 10 and not supported(value):
            bad.append(m.group(0).strip())
    return bad


def check_result(df: pd.DataFrame, truncated: bool) -> list[str]:
    """Проверки качества результата запроса. Возвращает список предупреждений."""
    if df.empty:
        return ["Запрос вернул пустой результат."]

    warnings = []
    if truncated:
        warnings.append(f"Результат обрезан до первых {len(df)} строк.")

    for col in df.columns:
        n_null = int(df[col].isna().sum())
        if n_null:
            warnings.append(f"В колонке «{col}» {n_null} из {len(df)} значений пустые (NULL).")

    n_dup = int(df.duplicated().sum())
    if n_dup:
        warnings.append(f"В результате {n_dup} полностью повторяющихся строк.")

    label_col = next((c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])), None)
    numeric = [c for c in df.select_dtypes("number").columns if not str(c).lower().endswith("id")]
    for col in numeric:
        s = df[col].dropna()
        n_neg = int((s < 0).sum())
        if n_neg:
            warnings.append(f"В колонке «{col}» {n_neg} отрицательных значений.")
        if len(s) >= 8:
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                lo, hi = q1 - 3 * iqr, q3 + 3 * iqr
                out = df.loc[s[(s < lo) | (s > hi)].index]
                if len(out):
                    examples = ", ".join(
                        (f"{r[label_col]}: " if label_col else "") + f"{r[col]:,.0f}".replace(",", " ")
                        for _, r in out.head(3).iterrows()
                    )
                    warnings.append(f"В колонке «{col}» {len(out)} выбросов (сильно вне типичного диапазона): {examples}.")
    return warnings
