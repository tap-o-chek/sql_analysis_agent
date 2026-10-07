"""CLI: uv run python -m sqlagent "Сколько банкоматов в Москве?" """

import argparse

from sqlagent import config
from sqlagent.agent.graph import build_graph, run


def print_step(step: dict) -> None:
    mark = {True: "✓", False: "✗"}.get(step.get("ok"), "·")
    print(f"{mark} [{step['step']}] {step['text']}")
    if step.get("sql"):
        print("    " + step["sql"].replace("\n", "\n    "))


def main() -> None:
    parser = argparse.ArgumentParser(description="Агент для аналитики по таблицам банка")
    parser.add_argument("question")
    parser.add_argument("--model", default=config.MODEL)
    parser.add_argument("--review-model", default=config.REVIEW_MODEL)
    parser.add_argument("--max-attempts", type=int, default=config.MAX_ATTEMPTS,
                        help="1 = без самоисправления (бейзлайн)")
    args = parser.parse_args()

    graph = build_graph(model=args.model, review_model=args.review_model, max_attempts=args.max_attempts)
    state = run(args.question, graph, on_step=print_step)

    df = state.get("result")
    if df is not None and not df.empty:
        print("\n" + df.head(30).to_string(index=False))
    print("\n" + state.get("answer", ""))


if __name__ == "__main__":
    main()
