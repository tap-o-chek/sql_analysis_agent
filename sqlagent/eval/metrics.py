"""Сравнение результата агента с эталоном.

results_match — мягкая execution accuracy: результаты запросов совпадают как наборы строк.
  - колонки сопоставляются по значениям, а не по именам; лишние колонки у агента допустимы
    (агент добавил госномер к названию маршрута — ответ от этого не хуже);
  - числа сравниваются по модулю и с точностью до 6 значащих цифр
    (недостача -1 350 800 и 1 350 800 — один ответ; ROUND(AVG(...)) не ломает сравнение);
  - '01' из strftime и 1 — одно и то же;
  - порядок строк учитывается, только если order_matters (вопросы «топ-N»).
facts_present — все ключевые факты звучат в текстовом ответе.
"""

import math
import numbers
import re
from collections import Counter
from itertools import islice, product

import pandas as pd

NUMERIC_STR = re.compile(r"-?\d+(?:\.\d+)?")


def _norm_cell(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, str):
        s = v.strip()
        if not NUMERIC_STR.fullmatch(s):
            return s
        v = float(s)
    if isinstance(v, numbers.Number) and not isinstance(v, bool):
        return float(f"{abs(float(v)):.6g}")
    return v


def _columns(df: pd.DataFrame) -> list[list]:
    return [[_norm_cell(v) for v in df.iloc[:, j].tolist()] for j in range(df.shape[1])]


def results_match(pred: pd.DataFrame, gold: pd.DataFrame, order_matters: bool = False) -> bool:
    if len(pred) != len(gold) or pred.shape[1] < gold.shape[1]:
        return False
    P, G = _columns(pred), _columns(gold)
    # для каждой колонки эталона — колонки агента с тем же набором значений
    candidates = [[j for j, pc in enumerate(P) if Counter(pc) == Counter(gc)] for gc in G]
    if any(not c for c in candidates):
        return False
    gold_rows = list(zip(*G))
    for combo in islice(product(*candidates), 5000):
        if len(set(combo)) != len(combo):
            continue
        pred_rows = list(zip(*(P[j] for j in combo)))
        if pred_rows == gold_rows if order_matters else Counter(pred_rows) == Counter(gold_rows):
            return True
    return False


def _join_digit_groups(s: str) -> str:
    """'1 350 800' → '1350800', чтобы факты с числами находились при любом форматировании."""
    return re.sub(r"(?<=\d)[   ](?=\d{3}(?!\d))", "", s)


def facts_present(answer: str, facts: list[str]) -> bool:
    text = _join_digit_groups(answer).lower().replace("ё", "е")
    for fact in facts:
        f = _join_digit_groups(fact).lower().replace("ё", "е")
        if re.fullmatch(r"[\d.,]+", f):
            if not re.search(rf"(?<!\d){re.escape(f)}(?!\d)", text):
                return False
        elif f not in text:
            return False
    return True
