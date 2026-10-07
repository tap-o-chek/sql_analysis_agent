-- Синтетическая БД: инкассация банкоматов и закупки банка.
-- Все данные сгенерированы скриптом sqlagent/db/generate.py, реальных данных нет.
-- Суммы в рублях, даты в формате ISO 'YYYY-MM-DD' (или 'YYYY-MM-DD HH:MM' для времени).

-- ============ Инкассация ============

CREATE TABLE branches (
    branch_id   INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,           -- название отделения
    city        TEXT NOT NULL,
    region      TEXT NOT NULL            -- федеральный округ
);

CREATE TABLE atms (
    atm_id        INTEGER PRIMARY KEY,
    branch_id     INTEGER NOT NULL REFERENCES branches(branch_id),  -- обслуживающее отделение
    address       TEXT NOT NULL,
    city          TEXT NOT NULL,         -- написание города может отличаться от branches.city
    model         TEXT NOT NULL,
    installed_at  TEXT NOT NULL
);

CREATE TABLE routes (
    route_id       INTEGER PRIMARY KEY,
    branch_id      INTEGER NOT NULL REFERENCES branches(branch_id),
    name           TEXT NOT NULL,        -- код маршрута, например 'МСК-1'
    vehicle_plate  TEXT NOT NULL         -- госномер инкассаторской машины
);

CREATE TABLE collections (
    collection_id    INTEGER PRIMARY KEY,
    route_id         INTEGER NOT NULL REFERENCES routes(route_id),
    atm_id           INTEGER NOT NULL REFERENCES atms(atm_id),
    collected_at     TEXT NOT NULL,      -- дата и время выезда
    status           TEXT NOT NULL,      -- 'completed' | 'cancelled' | 'rescheduled'
    amount_declared  INTEGER,            -- сумма по счётчикам банкомата; NULL, если выезд не состоялся
    amount_counted   INTEGER             -- сумма по пересчёту в кассовом центре; NULL, если выезд не состоялся
    -- расхождение = amount_counted - amount_declared (отрицательное значение = недостача)
);

-- ============ Закупки ============

CREATE TABLE suppliers (
    supplier_id    INTEGER PRIMARY KEY,
    name           TEXT NOT NULL,
    inn            TEXT NOT NULL,        -- ИНН, 10 цифр
    city           TEXT NOT NULL,
    registered_at  TEXT NOT NULL         -- дата регистрации юрлица
);

CREATE TABLE contracts (
    contract_id         INTEGER PRIMARY KEY,
    supplier_id         INTEGER NOT NULL REFERENCES suppliers(supplier_id),
    branch_id           INTEGER NOT NULL REFERENCES branches(branch_id),  -- отделение-заказчик
    signed_at           TEXT NOT NULL,
    subject             TEXT NOT NULL,   -- предмет закупки
    amount              INTEGER NOT NULL,
    procurement_method  TEXT NOT NULL    -- 'single_source' (до 500 000 ₽ можно без конкурса) | 'quotation' | 'tender'
);

CREATE TABLE invoices (
    invoice_id      INTEGER PRIMARY KEY,
    contract_id     INTEGER NOT NULL REFERENCES contracts(contract_id),
    invoice_number  TEXT NOT NULL,       -- номер счёта поставщика
    issued_at       TEXT NOT NULL,
    amount          INTEGER NOT NULL,
    paid_at         TEXT                 -- NULL, если счёт ещё не оплачен
);
