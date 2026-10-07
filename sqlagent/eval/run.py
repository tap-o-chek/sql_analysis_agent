"""Прогон агента по набору вопросов. Каждый вопрос → строка в eval/results/<name>.jsonl
с полной трассой, поэтому метрики (report.py) считаются без повторного запуска LLM.

    uv run python -m sqlagent.eval.run --name 7b
    uv run python -m sqlagent.eval.run --name 7b+27b --review-model qwen3.8:27b-mlx
    uv run python -m sqlagent.eval.run --name 27b --model qwen3.8:27b-mlx
    uv run python -m sqlagent.eval.run --name 7b --ids d01,m02     # только выбранные вопросы

Бейзлайн без самоисправления отдельно гонять не нужно: первый SQL агента — это и есть
ответ «одной LLM». Прерванный прогон продолжается с места (готовые вопросы пропускаются).
"""

import argparse
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import yaml

from sqlagent import checks, config
from sqlagent.agent.graph import build_graph, run
from sqlagent.db.access import SQLValidationError, connect, run_query, validate_sql
from sqlagent.eval.metrics import facts_present, results_match

QUESTIONS = config.ROOT / "eval" / "questions.yaml"
RESULTS_DIR = config.ROOT / "eval" / "results"
LLM_STEPS = {"generate_sql", "review"}


def load_questions(path: Path = QUESTIONS) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def execution_match(conn: sqlite3.Connection, sql: str, gold, order_matters: bool) -> bool:
    try:
        validate_sql(sql)
        pred, _ = run_query(conn, sql, max_rows=10_000)
    except (SQLValidationError, sqlite3.Error):
        return False
    return results_match(pred, gold, order_matters)


def evaluate(q: dict, state: dict, conn: sqlite3.Connection) -> dict:
    """Сравнивает итог одного прогона с эталоном. Возвращает запись для jsonl."""
    refuse_expected = q.get("expect", "answer") == "refuse"
    refused = bool(state.get("refused"))
    gold = None if refuse_expected else run_query(conn, q["gold_sql"], max_rows=10_000)[0]

    # все SQL, которые написал агент: отвергнутые (из history) + итоговый
    attempts = [{"sql": h["sql"], "rejected_by": h.get("source")} for h in state.get("history", [])]
    if state.get("sql") and not refused:
        attempts.append({"sql": state["sql"], "rejected_by": None})
    for a in attempts:
        a["ex"] = False if gold is None else execution_match(conn, a["sql"], gold, q.get("order_matters", False))

    if refuse_expected:
        first_correct = refused and state.get("attempts") == 1
        final_correct = refused
    else:
        first_correct = bool(attempts) and attempts[0]["ex"]
        final_correct = not refused and bool(attempts) and attempts[-1]["ex"]

    answer = state.get("answer", "")
    result = state.get("result")
    unsupported = []
    if result is not None and answer and not refused:
        unsupported = checks.unsupported_numbers(answer, result, context=q["question"] + " " + state.get("sql", ""))

    trace = state.get("trace", [])
    llm_steps = [s for s in trace if s["step"] in LLM_STEPS]
    return {
        "first_correct": first_correct,
        "final_correct": final_correct,
        "refused": refused,
        "attempts": attempts,
        "n_attempts": state.get("attempts", 0),
        "facts_ok": facts_present(answer, q["key_facts"]) if q.get("key_facts") else None,
        "unsupported_numbers": unsupported,
        "llm_calls": len(llm_steps),
        "tokens_in": sum(s.get("tokens", {}).get("in", 0) for s in llm_steps),
        "tokens_out": sum(s.get("tokens", {}).get("out", 0) for s in llm_steps),
        "answer": answer,
        "trace": [{k: v for k, v in s.items() if k != "tokens"} for s in trace],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогон агента по eval/questions.yaml")
    parser.add_argument("--name", required=True, help="имя конфигурации → eval/results/<name>.jsonl")
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--review-model", default=None, help="по умолчанию та же, что --model")
    parser.add_argument("--max-attempts", type=int, default=config.MAX_ATTEMPTS)
    parser.add_argument("--repeat", type=int, default=1, help="сколько раз прогнать каждый вопрос (для pass^k)")
    parser.add_argument("--ids", default=None, help="через запятую: только эти вопросы")
    parser.add_argument("--keep-alive", default="30m", help="держать модели в памяти между вопросами")
    args = parser.parse_args()

    questions = load_questions()
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]

    review_model = args.review_model or args.model
    out = RESULTS_DIR / f"{args.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["id"], r["run"]) for r in map(json.loads, out.read_text(encoding="utf-8").splitlines())}

    graph = build_graph(model=args.model, review_model=review_model,
                        max_attempts=args.max_attempts, keep_alive=args.keep_alive)
    todo = [(q, k) for k in range(args.repeat) for q in questions if (q["id"], k) not in done]
    print(f"{args.name}: {len(todo)} прогонов (уже готово {len(done)}), SQL: {args.model}, проверка: {review_model}")
    if not todo:
        return

    print("Прогрев моделей (загрузка в память не должна попасть в замеры времени)...")
    run("Сколько всего банкоматов?", graph)

    with closing(connect()) as conn, out.open("a", encoding="utf-8") as f:
        for i, (q, k) in enumerate(todo, 1):
            t0 = time.perf_counter()
            try:
                state, error = run(q["question"], graph), None
            except Exception as e:  # один упавший вопрос не должен ронять весь прогон
                state, error = {}, f"{type(e).__name__}: {e}"
            latency = time.perf_counter() - t0
            record = {"id": q["id"], "category": q["category"], "expect": q.get("expect", "answer"), "run": k,
                      "config": {"name": args.name, "model": args.model, "review_model": review_model,
                                 "max_attempts": args.max_attempts},
                      "question": q["question"], "latency_s": round(latency, 2), "error": error,
                      **evaluate(q, state, conn)}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            mark = lambda ok: "✓" if ok else "✗"
            print(f"[{i}/{len(todo)}] {q['id']:4} первый {mark(record['first_correct'])} → итог "
                  f"{mark(record['final_correct'])}  попыток {record['n_attempts']}  {latency:5.1f} с"
                  + (f"  ОШИБКА: {error}" if error else ""))

    print(f"\nГотово: {out}\nОтчёт: uv run python -m sqlagent.eval.report")


if __name__ == "__main__":
    main()
