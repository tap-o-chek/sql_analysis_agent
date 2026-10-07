import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DB_PATH = Path(os.getenv("SQLAGENT_DB", ROOT / "data" / "bank.db"))
MODEL = os.getenv("SQLAGENT_MODEL", "qwen2.5-coder:7b")
# модель для проверки результата и ответа; можно поставить крупнее, чем для SQL
REVIEW_MODEL = os.getenv("SQLAGENT_REVIEW_MODEL", MODEL)
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
# Ollama по умолчанию режет контекст до 2–4k токенов, схема с примерами строк длиннее
NUM_CTX = int(os.getenv("SQLAGENT_NUM_CTX", 8192))

MAX_ATTEMPTS = int(os.getenv("SQLAGENT_MAX_ATTEMPTS", 4))  # сколько раз агент может написать SQL
MAX_ROWS = 200           # больше строк из результата не забираем
QUERY_TIMEOUT_S = 5.0    # долгие запросы прерываем
