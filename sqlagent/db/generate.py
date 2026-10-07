"""Генератор синтетической БД банка: инкассация банкоматов и закупки.

Данные полностью синтетические (Faker + random с фиксированным seed), но в них
специально заложены аномалии, которые агент должен уметь находить:
  1. Маршрут инкассации с систематическими недостачами начиная с июня.
  2. Дробление закупки: серия договоров с одним поставщиком чуть ниже порога
     закупки без конкурса (500 000 ₽), поставщик зарегистрирован за 2 недели
     до первого договора.
  3. Дубли счетов: тот же номер и сумма по тому же договору, оплачены дважды.
А также «грязные» данные, как в реальной жизни: разное написание городов,
NULL в суммах отменённых выездов и в датах оплаты.

Список заложенных аномалий пишется в data/anomalies.json — это ground truth
для оценки агента.

Запуск: python -m sqlagent.db.generate
"""

import argparse
import json
import random
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path

from faker import Faker

SEED = 42
START = date(2025, 1, 1)
END = date(2025, 12, 31)
SINGLE_SOURCE_LIMIT = 500_000

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# город, округ, код маршрута, код региона на номерах, «грязные» варианты написания
CITIES = [
    ("Москва", "ЦФО", "МСК", "77", ["г. Москва", "москва"]),
    ("Санкт-Петербург", "СЗФО", "СПБ", "78", ["СПб", "Санкт-Петербург "]),
    ("Казань", "ПФО", "КЗН", "16", ["г. Казань"]),
    ("Екатеринбург", "УФО", "ЕКБ", "66", ["г. Екатеринбург", "Екатеринбург "]),
    ("Новосибирск", "СФО", "НСК", "54", ["г. Новосибирск"]),
    ("Нижний Новгород", "ПФО", "ННВ", "52", ["Н. Новгород"]),
]
DIRTY_CITY_P = 0.15

ATM_MODELS = ["NCR SelfServ 84", "Wincor Cineo C4060", "GRG H68N", "Hyosung MX8800"]
PLATE_LETTERS = "АВЕКМНОРСТУХ"

SUBJECTS = [
    "Канцелярские товары",
    "Ремонт помещений",
    "Обслуживание банкоматов",
    "Клининговые услуги",
    "IT-оборудование",
    "Охранные услуги",
    "Офисная мебель",
    "Расходные материалы для инкассации",
]

SHORTAGE_SINCE = date(2025, 6, 1)
SHORTAGE_P = 0.35
N_SPLIT_CONTRACTS = 7
N_DUPLICATE_INVOICES = 6


def rand_date(rng: random.Random, start: date, end: date) -> date:
    return start + timedelta(days=rng.randint(0, (end - start).days))


def maybe_dirty(rng: random.Random, city: tuple) -> str:
    name, *_, variants = city
    return rng.choice(variants) if rng.random() < DIRTY_CITY_P else name


def gen_branches() -> list[dict]:
    rows = []
    for city, region, *_ in CITIES:
        for k in (1, 2):
            rows.append({"branch_id": len(rows) + 1, "name": f"{city}, отделение №{k}",
                         "city": city, "region": region, "_city": city})
    return rows


def gen_atms(rng, fake, branches) -> list[dict]:
    city_by_name = {c[0]: c for c in CITIES}
    rows = []
    for b in branches:
        for _ in range(rng.randint(8, 12)):
            rows.append({
                "atm_id": len(rows) + 1,
                "branch_id": b["branch_id"],
                "address": fake.street_address(),
                "city": maybe_dirty(rng, city_by_name[b["city"]]),
                "model": rng.choice(ATM_MODELS),
                "installed_at": rand_date(rng, date(2015, 1, 1), date(2024, 12, 31)).isoformat(),
            })
    return rows


def gen_routes(rng, branches) -> list[dict]:
    code_by_city = {c[0]: (c[2], c[3]) for c in CITIES}
    rows = []
    n_by_code: dict[str, int] = {}
    for b in branches:
        code, plate_region = code_by_city[b["city"]]
        for _ in range(2):
            n_by_code[code] = n_by_code.get(code, 0) + 1
            plate = (rng.choice(PLATE_LETTERS) + f"{rng.randint(1, 999):03d}"
                     + rng.choice(PLATE_LETTERS) + rng.choice(PLATE_LETTERS) + plate_region)
            rows.append({"route_id": len(rows) + 1, "branch_id": b["branch_id"],
                         "name": f"{code}-{n_by_code[code]}", "vehicle_plate": plate})
    return rows


def gen_collections(rng, atms, routes, shortage_route_id) -> list[dict]:
    routes_by_branch: dict[int, list[int]] = {}
    for r in routes:
        routes_by_branch.setdefault(r["branch_id"], []).append(r["route_id"])

    rows = []
    for atm in atms:
        route_id = rng.choice(routes_by_branch[atm["branch_id"]])  # банкомат закреплён за маршрутом
        day = START + timedelta(days=rng.randint(0, 3))
        while day <= END:
            ts = datetime(day.year, day.month, day.day, rng.randint(8, 19), rng.choice((0, 15, 30, 45)))
            p = rng.random()
            if p < 0.03:
                status, declared, counted = "cancelled", None, None
            elif p < 0.05:
                status, declared, counted = "rescheduled", None, None
            else:
                status = "completed"
                declared = round(min(rng.lognormvariate(13.8, 0.5), 8_000_000), -2)
                declared = int(declared)
                counted = declared
                if route_id == shortage_route_id and day >= SHORTAGE_SINCE and rng.random() < SHORTAGE_P:
                    counted -= rng.randint(50, 500) * 100  # недостача 5–50 тыс.
                elif rng.random() < 0.03:
                    counted += rng.choice((-1, 1)) * rng.randint(1, 50) * 100  # обычная погрешность пересчёта
            rows.append({"route_id": route_id, "atm_id": atm["atm_id"],
                         "collected_at": ts.strftime("%Y-%m-%d %H:%M"), "status": status,
                         "amount_declared": declared, "amount_counted": counted})
            day += timedelta(days=rng.randint(3, 6))

    rows.sort(key=lambda r: r["collected_at"])
    for i, r in enumerate(rows, 1):
        r["collection_id"] = i
    return rows


def gen_suppliers(rng, fake, n=60) -> list[dict]:
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "supplier_id": i,
            "name": fake.company(),
            "inn": "".join(str(rng.randint(0, 9)) for _ in range(10)),
            "city": maybe_dirty(rng, rng.choice(CITIES)),
            "registered_at": rand_date(rng, date(2005, 1, 1), date(2023, 12, 31)).isoformat(),
        })
    return rows


def procurement_method(rng, amount: int) -> str:
    if amount < SINGLE_SOURCE_LIMIT:
        return "single_source" if rng.random() < 0.9 else "quotation"
    if amount < 3_000_000:
        return "quotation" if rng.random() < 0.6 else "tender"
    return "tender"


def gen_contracts(rng, suppliers, branches, shell, split_branch_id, n=600) -> list[dict]:
    normal_suppliers = [s for s in suppliers if s is not shell]
    rows = []
    for _ in range(n):
        amount = int(round(min(max(rng.lognormvariate(12.8, 1.0), 20_000), 20_000_000), -2))
        rows.append({
            "supplier_id": rng.choice(normal_suppliers)["supplier_id"],
            "branch_id": rng.choice(branches)["branch_id"],
            "signed_at": rand_date(rng, START, END).isoformat(),
            "subject": rng.choice(SUBJECTS),
            "amount": amount,
            "procurement_method": procurement_method(rng, amount),
        })

    # Дробление: серия договоров чуть ниже порога с поставщиком-«однодневкой»
    day = date.fromisoformat(shell["registered_at"]) + timedelta(days=14)
    for _ in range(N_SPLIT_CONTRACTS):
        rows.append({
            "supplier_id": shell["supplier_id"],
            "branch_id": split_branch_id,
            "signed_at": day.isoformat(),
            "subject": "Ремонт помещений",
            "amount": rng.randint(4700, 4990) * 100,
            "procurement_method": "single_source",
            "_split": True,
        })
        day += timedelta(days=rng.randint(1, 4))

    rows.sort(key=lambda r: r["signed_at"])
    for i, r in enumerate(rows, 1):
        r["contract_id"] = i
    return rows


def gen_invoices(rng, contracts) -> list[dict]:
    seq_by_supplier: dict[int, int] = {}
    rows = []
    for c in contracts:
        n = 1 if c["amount"] < 300_000 else rng.randint(1, 4)
        weights = [rng.random() + 0.2 for _ in range(n)]
        parts = [int(round(c["amount"] * w / sum(weights), -2)) for w in weights]
        parts[-1] = c["amount"] - sum(parts[:-1])
        issued = date.fromisoformat(c["signed_at"])
        for part in parts:
            issued = min(issued + timedelta(days=rng.randint(5, 60)), END)
            paid = issued + timedelta(days=rng.randint(3, 30))
            if paid > END or rng.random() < 0.05:
                paid = None
            seq_by_supplier[c["supplier_id"]] = seq_by_supplier.get(c["supplier_id"], 0) + 1
            rows.append({
                "contract_id": c["contract_id"],
                "invoice_number": f"СЧ-{c['supplier_id']:03d}-{seq_by_supplier[c['supplier_id']]:04d}",
                "issued_at": issued.isoformat(),
                "amount": part,
                "paid_at": paid.isoformat() if paid else None,
            })

    # Дубли счетов: копия с тем же номером и суммой, тоже оплачена
    candidates = [r for r in rows if r["paid_at"] and r["issued_at"] < "2025-11-01"]
    for orig in rng.sample(candidates, N_DUPLICATE_INVOICES):
        issued = date.fromisoformat(orig["issued_at"]) + timedelta(days=rng.randint(0, 5))
        rows.append({**orig, "issued_at": issued.isoformat(),
                     "paid_at": (issued + timedelta(days=rng.randint(3, 20))).isoformat(),
                     "_dup_of": orig})

    rows.sort(key=lambda r: (r["issued_at"], r["invoice_number"]))
    for i, r in enumerate(rows, 1):
        r["invoice_id"] = i
    return rows


def insert(conn: sqlite3.Connection, table: str, rows: list[dict]) -> None:
    cols = [k for k in rows[0] if not k.startswith("_")]
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [tuple(r[c] for c in cols) for r in rows],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DATA_DIR / "bank.db")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fake = Faker("ru_RU")
    fake.seed_instance(args.seed)

    branches = gen_branches()
    atms = gen_atms(rng, fake, branches)
    routes = gen_routes(rng, branches)
    shortage_route = rng.choice(routes)
    collections = gen_collections(rng, atms, routes, shortage_route["route_id"])

    suppliers = gen_suppliers(rng, fake)
    shell = rng.choice(suppliers)
    shell["registered_at"] = "2025-03-03"
    split_branch = rng.choice(branches)
    contracts = gen_contracts(rng, suppliers, branches, shell, split_branch["branch_id"])
    invoices = gen_invoices(rng, contracts)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.unlink(missing_ok=True)
    conn = sqlite3.connect(args.out)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    for table, rows in [("branches", branches), ("atms", atms), ("routes", routes),
                        ("collections", collections), ("suppliers", suppliers),
                        ("contracts", contracts), ("invoices", invoices)]:
        insert(conn, table, rows)
    conn.commit()
    conn.close()

    split = [c for c in contracts if c.get("_split")]
    anomalies = {
        "note": "Аномалии, специально заложенные генератором (seed=%d). Ground truth для оценки." % args.seed,
        "collection_shortages": {
            "route_id": shortage_route["route_id"],
            "route_name": shortage_route["name"],
            "since": SHORTAGE_SINCE.isoformat(),
            "description": f"С {SHORTAGE_SINCE:%d.%m.%Y} ~{SHORTAGE_P:.0%} выездов маршрута с недостачей 5–50 тыс. ₽",
        },
        "split_purchases": {
            "supplier_id": shell["supplier_id"],
            "supplier_name": shell["name"],
            "supplier_registered_at": shell["registered_at"],
            "branch_id": split_branch["branch_id"],
            "contract_ids": [c["contract_id"] for c in split],
            "total_amount": sum(c["amount"] for c in split),
            "description": f"{len(split)} договоров «Ремонт помещений» по 470–499 тыс. ₽ без конкурса "
                           f"(порог {SINGLE_SOURCE_LIMIT:,} ₽) за несколько недель".replace(",", " "),
        },
        "duplicate_invoices": [
            {"original_invoice_id": r["_dup_of"]["invoice_id"], "duplicate_invoice_id": r["invoice_id"],
             "invoice_number": r["invoice_number"], "contract_id": r["contract_id"], "amount": r["amount"]}
            for r in invoices if "_dup_of" in r
        ],
        "dirty_values": {
            "city_variants": {c[0]: c[4] for c in CITIES},
            "description": "atms.city и suppliers.city в ~15% строк записаны иначе, чем branches.city",
        },
    }
    (args.out.parent / "anomalies.json").write_text(
        json.dumps(anomalies, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"БД записана в {args.out}")
    for name, rows in [("branches", branches), ("atms", atms), ("routes", routes),
                       ("collections", collections), ("suppliers", suppliers),
                       ("contracts", contracts), ("invoices", invoices)]:
        print(f"  {name:<12} {len(rows):>6}")


if __name__ == "__main__":
    main()
