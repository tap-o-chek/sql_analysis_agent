"""Отчёт по прогонам: читает eval/results/*.jsonl, печатает markdown и пишет eval/results/report.md.

    uv run python -m sqlagent.eval.report
    uv run python -m sqlagent.eval.report 7b 27b      # только выбранные конфигурации
"""

import json
import statistics
import sys
from collections import defaultdict

from sqlagent.eval.run import RESULTS_DIR

TRIGGERS = {"execute": "ошибка выполнения", "inspect_filters": "проверка фильтров", "review": "проверка результата (LLM)"}


def table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def pct(k: int, n: int) -> str:
    return f"{k / n:.0%} ({k}/{n})" if n else "—"


def percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p * (len(xs) - 1)))]


def summary(recs: list[dict]) -> dict[str, str]:
    n = len(recs)
    answerable = [r for r in recs if r["expect"] == "answer"]
    answered = [r for r in answerable if not r["refused"] and r["answer"]]
    with_facts = [r for r in recs if r["facts_ok"] is not None]
    max_attempts = recs[0]["config"]["max_attempts"]
    latencies = [r["latency_s"] for r in recs]
    n_correct = sum(r["final_correct"] for r in recs)
    return {
        "Точность, первая попытка (= одна LLM)": pct(sum(r["first_correct"] for r in recs), n),
        "Точность, итог агента": pct(n_correct, n),
        "Исправил (первая ✗ → итог ✓)": pct(sum(not r["first_correct"] and r["final_correct"] for r in recs), n),
        "Испортил (первая ✓ → итог ✗)": pct(sum(r["first_correct"] and not r["final_correct"] for r in recs), n),
        "Ключевые факты в ответе": pct(sum(r["facts_ok"] for r in with_facts), len(with_facts)),
        "Ответы с числами не из таблицы": pct(sum(bool(r["unsupported_numbers"]) for r in answered), len(answered)),
        "Ложные отказы (данные есть)": pct(sum(r["refused"] for r in answerable), len(answerable)),
        "Попыток SQL в среднем": f"{statistics.mean(r['n_attempts'] for r in recs):.2f}",
        "Исчерпал все попытки": pct(sum(r["n_attempts"] >= max_attempts for r in recs), n),
        "Вызовов LLM в среднем": f"{statistics.mean(r['llm_calls'] for r in recs):.2f}",
        "Токенов в среднем (вход + выход)": f"{statistics.mean(r['tokens_in'] + r['tokens_out'] for r in recs):,.0f}".replace(",", " "),
        "Время p50 / p95, с": f"{percentile(latencies, 0.5):.1f} / {percentile(latencies, 0.95):.1f}",
        "Время на один правильный ответ, с": f"{sum(latencies) / n_correct:.1f}" if n_correct else "—",
        "Ответ проверки не прошёл схему": str(sum(r.get("review_parse_errors", 0) for r in recs)),
        "Упавшие прогоны": str(sum(bool(r["error"]) for r in recs)),
    }


def by_category(configs: dict[str, list[dict]]) -> str:
    categories = list(dict.fromkeys(r["category"] for recs in configs.values() for r in recs))
    rows = []
    for cat in categories:
        row = [cat]
        for recs in configs.values():
            rs = [r for r in recs if r["category"] == cat]
            first, final = sum(r["first_correct"] for r in rs), sum(r["final_correct"] for r in rs)
            row.append(f"{first}/{len(rs)} → **{final}/{len(rs)}**" if rs else "—")
        rows.append(row)
    return table(["Категория", *configs], rows)


def triggers(recs: list[dict]) -> str:
    """Что запускало исправление и было ли оно по делу."""
    stats = defaultdict(lambda: [0, 0, 0])  # [сработала, ложная тревога, следующая попытка верная]
    for r in recs:
        if r["expect"] != "answer":
            continue
        atts = r["attempts"]
        for i, a in enumerate(atts):
            if not a["rejected_by"]:
                continue
            s = stats[a["rejected_by"]]
            s[0] += 1
            s[1] += a["ex"]
            s[2] += i + 1 < len(atts) and atts[i + 1]["ex"]
    rows = [[TRIGGERS.get(k, k), n, pct(fa, n), pct(fixed, n)] for k, (n, fa, fixed) in stats.items()]
    return table(["Что отправило SQL на исправление", "Раз", "Ложная тревога (SQL уже был верным)",
                  "Следующая попытка верна"], rows) if rows else "_Исправлений не было._"


def pass_k(recs: list[dict]) -> str | None:
    runs = defaultdict(list)
    for r in recs:
        runs[r["id"]].append(r["final_correct"])
    k = min(len(v) for v in runs.values())
    if k < 2:
        return None
    return (f"pass@{k} (верно хотя бы в одном из {k} прогонов): {pct(sum(any(v) for v in runs.values()), len(runs))}; "
            f"pass^{k} (верно во всех {k}): {pct(sum(all(v) for v in runs.values()), len(runs))}")


def main() -> None:
    names = sys.argv[1:] or sorted(p.stem for p in RESULTS_DIR.glob("*.jsonl"))
    configs = {}
    for name in names:
        lines = (RESULTS_DIR / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
        configs[name] = [json.loads(line) for line in lines if line.strip()]
    if not configs:
        sys.exit("Нет результатов в eval/results/ — сначала запустите sqlagent.eval.run")

    first = next(iter(configs.values()))[0]["config"]
    parts = ["# Оценка агента", ""]
    parts.append("Конфигурации: " + "; ".join(
        f"**{n}** — SQL: `{rs[0]['config']['model']}`, проверка: `{rs[0]['config']['review_model']}`"
        for n, rs in configs.items()) + f". Максимум попыток SQL: {first['max_attempts']}.")
    parts += ["", "## Итог", ""]
    sums = {n: summary(rs) for n, rs in configs.items()}
    metrics = list(next(iter(sums.values())))
    parts.append(table(["Метрика", *configs], [[m, *(sums[n][m] for n in configs)] for m in metrics]))
    parts += ["", "## По категориям (первая попытка → итог)", "", by_category(configs)]
    for name, recs in configs.items():
        parts += ["", f"## Исправления: {name}", "", triggers(recs)]
        if pk := pass_k(recs):
            parts += ["", pk]

    report = "\n".join(parts) + "\n"
    (RESULTS_DIR / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
