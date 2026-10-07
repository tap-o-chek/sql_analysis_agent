"""Сквозной прогон графа и оценки на заглушке LLM, без Ollama.

Заглушка отвечает эталонным SQL (или отказом), поэтому каждый вопрос обязан решиться
с первой попытки. Если упало — либо ошибка в эталоне, либо детерминированная проверка
даёт ложную тревогу на правильном запросе.
"""

import json
from contextlib import closing
from types import SimpleNamespace

import pytest

from sqlagent.agent.graph import build_graph, run
from sqlagent.db.access import connect
from sqlagent.eval.run import evaluate, load_questions

QUESTIONS = load_questions()
BY_QUESTION = {q["question"]: q for q in QUESTIONS}


class FakeLLM:
    def __init__(self, respond):
        self.respond = respond

    def invoke(self, messages):
        return SimpleNamespace(content=self.respond(messages), usage_metadata={"input_tokens": 1, "output_tokens": 1})


def gold_sql(messages):
    q = BY_QUESTION[messages[1].content.split("\n\n")[0]]
    return f"```sql\n{q['gold_sql']}\n```" if q.get("gold_sql") else "CANNOT_ANSWER: в базе нет таких данных"


def review_ok(messages):
    return json.dumps({"conditions": [], "problems": [], "verdict": "ok", "reason": "", "answer": "Ответ в таблице."})


@pytest.fixture(scope="module")
def graph():
    return build_graph(llm=FakeLLM(gold_sql), llm_json=FakeLLM(review_ok))


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_gold_passes_first_try(graph, q):
    state = run(q["question"], graph)
    with closing(connect()) as conn:
        rec = evaluate(q, state, conn)
    assert rec["first_correct"] and rec["final_correct"], rec["trace"]
    assert rec["n_attempts"] == 1, [s["text"] for s in rec["trace"]]
