"""Сквозной прогон графа и оценки на заглушке LLM, без Ollama.

Заглушка отвечает эталонным SQL (или отказом), поэтому каждый вопрос обязан решиться
с первой попытки. Если упало — либо ошибка в эталоне, либо детерминированная проверка
даёт ложную тревогу на правильном запросе.
"""

import json
from contextlib import closing
from types import SimpleNamespace

import pytest

from sqlagent.agent.graph import Review, build_graph, run
from sqlagent.db.access import connect
from sqlagent.eval.run import evaluate, load_questions

QUESTIONS = load_questions()
BY_QUESTION = {q["question"]: q for q in QUESTIONS}


class FakeLLM:
    def __init__(self, respond):
        self.respond = respond

    def invoke(self, messages):
        return SimpleNamespace(content=self.respond(messages), usage_metadata={"input_tokens": 1, "output_tokens": 1})

    def with_structured_output(self, schema, **kwargs):
        """Как в LangChain с include_raw=True: {"raw", "parsed", "parsing_error"}."""
        fake = self

        class Structured:
            def invoke(self, messages):
                raw = fake.invoke(messages)
                try:
                    return {"raw": raw, "parsed": schema.model_validate_json(raw.content), "parsing_error": None}
                except ValueError as e:
                    return {"raw": raw, "parsed": None, "parsing_error": e}

        return Structured()


def gold_sql(messages):
    q = BY_QUESTION[messages[1].content.split("\n\n")[0]]
    return f"```sql\n{q['gold_sql']}\n```" if q.get("gold_sql") else "CANNOT_ANSWER: в базе нет таких данных"


def review_ok(messages):
    return json.dumps({"conditions": [], "problems": [], "verdict": "ok", "reason": "", "answer": "Ответ в таблице."})


@pytest.fixture(scope="module")
def graph():
    return build_graph(llm=FakeLLM(gold_sql), review_llm=FakeLLM(review_ok))


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_gold_passes_first_try(graph, q):
    state = run(q["question"], graph)
    with closing(connect()) as conn:
        rec = evaluate(q, state, conn)
    assert rec["first_correct"] and rec["final_correct"], rec["trace"]
    assert rec["n_attempts"] == 1, [s["text"] for s in rec["trace"]]


def test_broken_review_json_is_reported():
    g = build_graph(llm=FakeLLM(gold_sql), review_llm=FakeLLM(lambda m: '{"verdict": "maybe"}'))
    state = run("Сколько всего банкоматов?", g)
    review_steps = [s for s in state["trace"] if s["step"] == "review"]
    assert review_steps[-1].get("parse_error") and "не удалось" in state["answer"]
