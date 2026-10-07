"""Доступ к БД только на чтение: проверка SQL, выполнение с лимитами, описание схемы."""

import sqlite3
import time
from pathlib import Path

import pandas as pd
import sqlglot
from sqlglot import exp

from sqlagent.config import DB_PATH, MAX_ROWS, QUERY_TIMEOUT_S

# Узлы, которых не должно быть в запросе на чтение (на случай, если sqlglot распарсит их внутри SELECT)
FORBIDDEN = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command)


class SQLValidationError(ValueError):
    pass


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Подключение в режиме read-only: даже если проверка SQL что-то пропустит, база не изменится."""
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def validate_sql(sql: str) -> None:
    """Пропускаем только один запрос SELECT (в том числе с WITH и UNION)."""
    try:
        statements = [s for s in sqlglot.parse(sql, read="sqlite") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise SQLValidationError(f"Синтаксическая ошибка: {e}") from e
    if len(statements) != 1:
        raise SQLValidationError(f"Нужен ровно один запрос, получено {len(statements)}.")
    tree = statements[0]
    if not isinstance(tree, exp.Query):
        raise SQLValidationError(f"Разрешены только SELECT-запросы, получено: {tree.key.upper()}.")
    if any(tree.find_all(*FORBIDDEN)):
        raise SQLValidationError("Запрос содержит операции изменения данных.")


def run_query(conn: sqlite3.Connection, sql: str,
              max_rows: int = MAX_ROWS, timeout_s: float = QUERY_TIMEOUT_S) -> tuple[pd.DataFrame, bool]:
    """Выполняет запрос, возвращает (результат, обрезан_ли_он). Ошибки SQLite пробрасывает наверх."""
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(max_rows + 1)
    except sqlite3.OperationalError as e:
        if "interrupted" in str(e):
            raise sqlite3.OperationalError(f"Запрос выполнялся дольше {timeout_s:g} с и был прерван.") from e
        raise
    finally:
        conn.set_progress_handler(None, 0)
    columns = [d[0] for d in cur.description]
    truncated = len(rows) > max_rows
    return pd.DataFrame(rows[:max_rows], columns=columns), truncated


def table_columns(conn: sqlite3.Connection) -> dict[str, dict[str, str]]:
    """{таблица: {колонка: тип}}"""
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY rowid")]
    return {t: {row[1]: row[2].upper() for row in conn.execute(f"PRAGMA table_info({t})")} for t in tables}


def describe_schema(conn: sqlite3.Connection, sample_rows: int = 3) -> str:
    """CREATE TABLE с комментариями (SQLite хранит исходный текст) + несколько строк-примеров."""
    parts = []
    for name, ddl in conn.execute("SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY rowid"):
        sample = pd.read_sql_query(f"SELECT * FROM {name} LIMIT {sample_rows}", conn)
        parts.append(f"{ddl};\n/* Примеры строк {name}:\n{sample.to_string(index=False)}\n*/")
    return "\n\n".join(parts)
