"""
logger.py — Her istegi SQLite'a yaz: cache sonucu, model, token, maliyet, latency kirilimi.
Dashboard (Streamlit) ve degerlendirme bu tablodan okur.
"""
import sqlite3
import threading
import time

# Groq fiyatlari, $ / 1M token (input, output) — groq.com/pricing, Eylul 2026
PRICES = {
    "openai/gpt-oss-20b": (0.075, 0.30),
    "openai/gpt-oss-120b": (0.15, 0.60),
}


def cost_usd(model: str | None, input_tokens: int, output_tokens: int) -> float:
    if not model or model not in PRICES:
        return 0.0
    p_in, p_out = PRICES[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1_000_000


COLUMNS = {
    "ts": "REAL",
    "question": "TEXT",
    "use_cache": "INTEGER",
    "use_router": "INTEGER",
    "cache_hit": "INTEGER",
    "best_similarity": "REAL",
    "verify_calls": "INTEGER",
    "verify_input_tokens": "INTEGER",
    "verify_output_tokens": "INTEGER",
    "verify_error": "TEXT",
    "model": "TEXT",
    "route_reason": "TEXT",
    "input_tokens": "INTEGER",
    "output_tokens": "INTEGER",
    "cost_usd": "REAL",
    "embed_ms": "INTEGER",
    "cache_ms": "INTEGER",
    "retrieval_ms": "INTEGER",
    "llm_ms": "INTEGER",
    "total_ms": "INTEGER",
    "run_tag": "TEXT",   # degerlendirmede "baseline" / "cached" gibi etiket
}


class RequestLogger:
    def __init__(self, path: str = "logs.db"):
        # FastAPI sync endpoint'leri thread pool'da calisir -> tek baglanti + kilit
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        cols = ", ".join(f"{k} {v}" for k, v in COLUMNS.items())
        with self.lock:
            self.conn.execute(f"CREATE TABLE IF NOT EXISTS requests "
                              f"(id INTEGER PRIMARY KEY AUTOINCREMENT, {cols})")
            # Eski logs.db'de olmayan yeni kolonlari ekle (veri kaybetmeden)
            existing = {r[1] for r in self.conn.execute("PRAGMA table_info(requests)")}
            for k, v in COLUMNS.items():
                if k not in existing:
                    self.conn.execute(f"ALTER TABLE requests ADD COLUMN {k} {v}")
            self.conn.commit()

    def log(self, **row):
        row.setdefault("ts", time.time())
        data = {k: row.get(k) for k in COLUMNS}
        keys = ", ".join(data)
        marks = ", ".join("?" for _ in data)
        with self.lock:
            self.conn.execute(f"INSERT INTO requests ({keys}) VALUES ({marks})",
                              list(data.values()))
            self.conn.commit()

    def rows(self, run_tag: str | None = None) -> list[dict]:
        q = "SELECT * FROM requests"
        args = ()
        if run_tag:
            q += " WHERE run_tag = ?"
            args = (run_tag,)
        with self.lock:
            cur = self.conn.execute(q, args)
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]