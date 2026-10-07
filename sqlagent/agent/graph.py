"""Граф агента на LangGraph.

    generate_sql → execute ──ошибка──────────────────────────→ generate_sql (или give_up)
                      └─ok→ inspect_filters ──фильтр теряет строки→ generate_sql
                                  └─ok→ check_result → review ──revise→ generate_sql
                                                          └─ok→ verify_answer ──число не из таблицы→ review
                                                                     └─ok→ END

LLM делает две вещи: пишет SQL и оценивает результат (review). Всё, что можно проверить
без LLM (синтаксис и безопасность SQL, реальные значения и даты в фильтрах, качество
результата, числа в ответе), делают детерминированные функции — это быстрее, дешевле
и предсказуемее.
Число попыток ограничено max_attempts; при max_attempts=1 граф вырождается в one-shot
(одна LLM пишет SQL и отвечает) — это бейзлайн для сравнения.
"""

import json
import operator
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated, Callable, TypedDict

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from sqlagent import checks, config
from sqlagent.agent import prompts
from sqlagent.db.access import SQLValidationError, connect, describe_schema, run_query, table_columns, validate_sql


class AgentState(TypedDict, total=False):
    question: str
    sql: str
    attempts: int                                   # сколько раз уже писали SQL
    history: Annotated[list[dict], operator.add]    # отвергнутые запросы и причины — контекст для исправления
    error: str | None
    result: pd.DataFrame | None
    truncated: bool
    filter_issues: list[str]
    warnings: list[str]
    verdict: str                                    # "revise" — переписать SQL, "accept" — ответ готов
    answer: str
    answer_feedback: str | None                     # какие числа в ответе не нашлись в результате
    answer_retries: int
    trace: Annotated[list[dict], operator.add]      # шаги агента для CLI/UI


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip().rstrip(";").strip()


def format_history(history: list[dict]) -> str:
    return "\n\n".join(f"Попытка {i}:\n```sql\n{h['sql']}\n```\nПроблема: {h['problem']}"
                       for i, h in enumerate(history, 1))


def build_graph(model: str = config.MODEL, review_model: str | None = None,
                db_path: Path = config.DB_PATH, max_attempts: int = config.MAX_ATTEMPTS):
    llm = ChatOllama(model=model, base_url=config.OLLAMA_URL, temperature=0, num_ctx=config.NUM_CTX)
    llm_json = ChatOllama(model=review_model or config.REVIEW_MODEL, base_url=config.OLLAMA_URL, temperature=0,
                          num_ctx=config.NUM_CTX, format="json")
    with closing(connect(db_path)) as conn:
        schema = describe_schema(conn)
        columns = table_columns(conn)

    def can_retry(state: AgentState) -> bool:
        return state["attempts"] < max_attempts

    # ---------- узлы ----------

    def generate_sql(state: AgentState) -> dict:
        user = state["question"]
        if state.get("history"):
            user += "\n\n" + prompts.SQL_RETRY.format(history=format_history(state["history"]))
        reply = llm.invoke([SystemMessage(prompts.SQL_SYSTEM.format(schema=schema)), HumanMessage(user)])
        sql = extract_sql(reply.content)
        attempt = state.get("attempts", 0) + 1
        return {"sql": sql, "attempts": attempt, "error": None, "result": None,
                "trace": [{"step": "generate_sql", "text": f"Попытка {attempt}: пишу SQL", "sql": sql}]}

    def execute(state: AgentState) -> dict:
        sql = state["sql"]
        try:
            validate_sql(sql)
            with closing(connect(db_path)) as conn:
                df, truncated = run_query(conn, sql)
        except (SQLValidationError, sqlite3.Error) as e:
            update = {"error": str(e), "trace": [{"step": "execute", "text": f"Ошибка: {e}", "ok": False}]}
            if can_retry(state):
                update["history"] = [{"sql": sql, "problem": f"Ошибка выполнения: {e}"}]
            return update
        return {"result": df, "truncated": truncated,
                "trace": [{"step": "execute", "text": f"Выполнено: {len(df)} строк", "ok": True}]}

    def inspect_filters(state: AgentState) -> dict:
        with closing(connect(db_path)) as conn:
            issues = checks.inspect_filters(conn, state["sql"], columns)
        update = {"filter_issues": issues,
                  "trace": [{"step": "inspect_filters",
                             "text": "\n".join(issues) or "Текстовые фильтры совпадают с данными", "ok": not issues}]}
        if issues and can_retry(state):
            update["history"] = [{"sql": state["sql"], "problem": " ".join(issues)}]
        return update

    def check_result(state: AgentState) -> dict:
        warnings = state.get("filter_issues", []) + checks.check_result(state["result"], state["truncated"])
        return {"warnings": warnings,
                "trace": [{"step": "check_result", "text": "\n".join(warnings) or "Замечаний нет", "ok": not warnings}]}

    def review(state: AgentState) -> dict:
        df = state["result"]
        preview = df.head(20).to_string(index=False) if not df.empty else "(пусто)"
        user = prompts.REVIEW_USER.format(
            question=state["question"], sql=state["sql"], n_rows=len(df), n_shown=min(len(df), 20),
            preview=preview, warnings="\n".join(f"- {w}" for w in state["warnings"]) or "нет")
        if state.get("answer_feedback"):
            user += prompts.ANSWER_FEEDBACK.format(numbers=state["answer_feedback"])
        system = prompts.REVIEW_SYSTEM.replace("{schema}", schema)  # не .format: в промпте есть JSON со скобками
        raw = llm_json.invoke([SystemMessage(system), HumanMessage(user)]).content
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"verdict": "ok", "answer": raw}
        verdict = "revise" if parsed.get("verdict") == "revise" else "ok"
        reason = str(parsed.get("reason") or "").strip()
        answer = str(parsed.get("answer") or "").strip()

        if verdict == "revise" and can_retry(state):
            return {"verdict": "revise", "answer_feedback": None,
                    "history": [{"sql": state["sql"], "problem": f"Проверка результата: {reason}"}],
                    "trace": [{"step": "review", "text": f"Нужно исправить: {reason}", "ok": False}]}
        if verdict == "revise":  # попытки кончились — отвечаем как есть, но честно предупреждаем
            answer = (answer + "\n\n" if answer else "") + f"⚠️ Результат может быть неточным: {reason}"
        return {"verdict": "accept", "answer": answer or "Результат в таблице.",
                "trace": [{"step": "review", "text": "Результат принят, формирую ответ", "ok": True}]}

    def verify_answer(state: AgentState) -> dict:
        bad = checks.unsupported_numbers(state["answer"], state["result"],
                                         context=state["question"] + " " + state["sql"])
        retries = state.get("answer_retries", 0)
        if not bad:
            return {"answer_feedback": None,
                    "trace": [{"step": "verify_answer", "text": "Все числа в ответе есть в результате", "ok": True}]}
        text = f"В ответе есть числа, которых нет в результате: {', '.join(bad)}"
        if retries < 1:
            return {"answer_feedback": ", ".join(bad), "answer_retries": retries + 1,
                    "trace": [{"step": "verify_answer", "text": text + " — переписываю ответ", "ok": False}]}
        return {"answer": state["answer"] + f"\n\n⚠️ Числа {', '.join(bad)} не удалось сверить с таблицей.",
                "answer_feedback": None, "trace": [{"step": "verify_answer", "text": text, "ok": False}]}

    def give_up(state: AgentState) -> dict:
        answer = f"Не удалось построить корректный запрос за {state['attempts']} попыток. Последняя ошибка: {state['error']}"
        return {"answer": answer, "trace": [{"step": "give_up", "text": answer, "ok": False}]}

    # ---------- переходы ----------

    def after_execute(state: AgentState) -> str:
        if state.get("error"):
            return "generate_sql" if can_retry(state) else "give_up"
        return "inspect_filters"

    def after_inspect(state: AgentState) -> str:
        return "generate_sql" if state["filter_issues"] and can_retry(state) else "check_result"

    def after_review(state: AgentState) -> str:
        return "generate_sql" if state["verdict"] == "revise" else "verify_answer"

    def after_verify(state: AgentState) -> str:
        return "review" if state.get("answer_feedback") else END

    g = StateGraph(AgentState)
    for node in (generate_sql, execute, inspect_filters, check_result, review, verify_answer, give_up):
        g.add_node(node.__name__, node)
    g.add_edge(START, "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_conditional_edges("execute", after_execute, ["generate_sql", "inspect_filters", "give_up"])
    g.add_conditional_edges("inspect_filters", after_inspect, ["generate_sql", "check_result"])
    g.add_edge("check_result", "review")
    g.add_conditional_edges("review", after_review, ["generate_sql", "verify_answer"])
    g.add_conditional_edges("verify_answer", after_verify, ["review", END])
    g.add_edge("give_up", END)
    return g.compile()


def run(question: str, graph=None, on_step: Callable[[dict], None] | None = None) -> AgentState:
    """Запускает агента на вопросе. on_step вызывается на каждом новом шаге трассы."""
    graph = graph or build_graph()
    state: AgentState = {}
    n_seen = 0
    for state in graph.stream({"question": question, "attempts": 0}, {"recursion_limit": 50},
                              stream_mode="values"):
        for step in state.get("trace", [])[n_seen:]:
            if on_step:
                on_step(step)
        n_seen = len(state.get("trace", []))
    return state
