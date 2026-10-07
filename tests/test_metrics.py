import pandas as pd

from sqlagent.eval.metrics import facts_present, results_match


def test_extra_columns_and_names_ignored():
    gold = pd.DataFrame({"name": ["КЗН-2"]})
    pred = pd.DataFrame({"route_id": [10], "route_name": ["КЗН-2"], "plate": ["Т322УВ16"]})
    assert results_match(pred, gold)


def test_sign_and_rounding_ignored():
    assert results_match(pd.DataFrame({"x": [-1_881_000]}), pd.DataFrame({"y": [1_881_000.0]}))
    assert results_match(pd.DataFrame({"x": [1120467.0]}), pd.DataFrame({"y": [1120467.3349]}))


def test_month_string_vs_int():
    gold = pd.DataFrame({"m": ["01", "02"], "n": [857, 762]})
    pred = pd.DataFrame({"month": [2, 1], "cnt": [762, 857]})
    assert results_match(pred, gold)
    assert not results_match(pred, gold, order_matters=True)


def test_rows_must_align():
    gold = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    assert not results_match(pd.DataFrame({"a": ["x", "y"], "b": [2, 1]}), gold)
    assert not results_match(pd.DataFrame({"a": ["x"], "b": [1]}), gold)


def test_facts():
    assert facts_present("Недостача на КЗН-2 — 1 350 800 ₽", ["КЗН-2", "1350800"])
    assert facts_present("В Москве 17 банкоматов.", ["17"])
    assert not facts_present("В Москве 170 банкоматов.", ["17"])
