import pandas as pd
import pytest

from sqlagent.checks import check_result, inspect_filters, unsupported_numbers
from sqlagent.db.access import SQLValidationError, connect, table_columns, validate_sql


@pytest.fixture(scope="module")
def db():
    conn = connect()
    yield conn, table_columns(conn)
    conn.close()


def test_finds_city_variants(db):
    issues = inspect_filters(*db[:1], "SELECT COUNT(*) FROM atms WHERE city = 'Москва'", db[1])
    assert len(issues) == 1 and "г. Москва" in issues[0]


def test_unknown_value_lists_existing(db):
    issues = inspect_filters(db[0], "SELECT * FROM collections WHERE status = 'done'", db[1])
    assert "completed" in issues[0]


def test_cyrillic_like_is_case_sensitive(db):
    issues = inspect_filters(db[0], "SELECT * FROM contracts WHERE subject LIKE '%ремонт%'", db[1])
    assert "Ремонт помещений" in issues[0]


def test_codes_with_other_digits_are_not_variants(db):
    assert inspect_filters(db[0], "SELECT * FROM routes WHERE name = 'КЗН-2'", db[1]) == []


def test_date_boundary_on_datetime_column(db):
    sql = "SELECT COUNT(*) FROM collections WHERE collected_at BETWEEN '2025-07-01' AND '2025-12-31'"
    assert "дату со временем" in inspect_filters(db[0], sql, db[1])[0]


def test_correct_filters_pass(db):
    sql = "SELECT COUNT(*) FROM atms WHERE city IN ('Москва', 'г. Москва', 'москва')"
    assert inspect_filters(db[0], sql, db[1]) == []


@pytest.mark.parametrize("sql", ["DELETE FROM atms", "SELECT 1; DROP TABLE atms", "PRAGMA table_info(atms)",
                                 "UPDATE invoices SET paid_at = '2025-12-31'"])
def test_only_select_allowed(sql):
    with pytest.raises(SQLValidationError):
        validate_sql(sql)


def test_check_result_outlier_and_negative():
    df = pd.DataFrame({"route": list("ABCDEFGHIJ"), "d": [-1_350_800] + [-100 * i for i in range(1, 10)]})
    warnings = " ".join(check_result(df, truncated=False))
    assert "отрицательных" in warnings and "A: -1 350 800" in warnings


def test_unsupported_numbers():
    df = pd.DataFrame({"route": ["КЗН-2"], "s": [-1_502_800]})
    assert unsupported_numbers("КЗН-2: недостача 1 502 800 ₽, около 1,5 млн", df) == []
    assert unsupported_numbers("Всего 20 договоров на 10 000 000 ₽", df) == ["20", "10 000 000"]
