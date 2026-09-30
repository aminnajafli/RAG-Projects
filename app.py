"""
app.py — RAG + 2 asamali semantic cache + istek loglama
  soru -> embed -> cache? -> retrieval (Chroma) -> Groq LLM -> cevap -> log
Calistir: uvicorn app:app --reload
Test:     http://127.0.0.1:8000/docs
"""
import os
import statistics
import time

import chromadb
from fastapi import FastAPI
from groq import Groq
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from cache import SemanticCache, is_refusal
from logger import RequestLogger, cost_usd
from router import route

SMALL_MODEL = os.getenv("SMALL_MODEL", "openai/gpt-oss-20b")   # basit sorular + cache dogrulama
LARGE_MODEL = os.getenv("LARGE_MODEL", "openai/gpt-oss-120b")
EMBED_MODEL = "BAAI/bge-base-en-v1.5"   # ingest.py ile AYNI olmali
TOP_K = 5

embedder = SentenceTransformer(EMBED_MODEL)
col = chromadb.PersistentClient(path="chroma_db").get_collection("handbook")
# GROQ_API_KEY ortam degiskeninden okunur.
# max_retries: 429/5xx'te SDK retry-after suresine uyarak kendisi tekrar dener.
llm = Groq(max_retries=5)
cache = SemanticCache(verify_model=SMALL_MODEL, path="chroma_db")
logger = RequestLogger("logs.db")

SYSTEM = """You answer questions about the GitLab Handbook.
Use ONLY the provided context. If the answer is not in the context, say you don't know.
Cite the sources you used as [source path]. Be concise."""


def ms_since(t: float) -> int:
    return round((time.perf_counter() - t) * 1000)


def embed(text: str) -> list[float]:
    return embedder.encode([text], normalize_embeddings=True)[0].tolist()


def retrieve(q_emb: list[float], k: int = TOP_K) -> list[dict]:
    res = col.query(query_embeddings=[q_emb], n_results=k)
    return [
        {"text": doc, "source": meta["source"], "score": round(1 - dist, 3)}
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


def generate(question: str, chunks: list[dict], model: str) -> dict:
    context = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in chunks)
    resp = llm.chat.completions.create(
        model=model,
        temperature=0,   # ayni soru + ayni context -> tutarli cevap
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    u = resp.usage
    return {
        "answer": resp.choices[0].message.content,
        "model": model,
        "input_tokens": u.prompt_tokens,
        "output_tokens": u.completion_tokens,
    }


app = FastAPI(title="RAG + semantic cache")


class AskRequest(BaseModel):
    question: str
    use_cache: bool = True        # False + use_router False = baseline (karsilastirma icin)
    use_router: bool = True       # False = her miss buyuk modele gider
    run_tag: str | None = None    # degerlendirmede kosuyu etiketlemek icin


@app.post("/ask")
def ask(req: AskRequest):
    t0 = time.perf_counter()
    timing = {"embed_ms": 0, "cache_ms": 0, "retrieval_ms": 0, "llm_ms": 0}

    t = time.perf_counter()
    q_emb = embed(req.question)   # tek embedding: hem cache hem retrieval icin
    timing["embed_ms"] = ms_since(t)

    cinfo = {"best_similarity": None, "verify_calls": 0,
             "verify_input_tokens": 0, "verify_output_tokens": 0, "verify_error": None}
    # Uzun / cok parcali sorular tekrar etmez ve dogrulayici bunlari guvenilir ayiramaz
    _, reason_for_cache = route(req.question, SMALL_MODEL, LARGE_MODEL)
    cacheable = req.use_cache and not reason_for_cache.startswith(("multi_part", "long"))

    hit = None
    if cacheable:
        t = time.perf_counter()
        hit, cinfo = cache.lookup(req.question, q_emb)
        timing["cache_ms"] = ms_since(t)

    if hit:
        out = {**hit, "cache_hit": True, "model": None, "route_reason": None,
               "input_tokens": 0, "output_tokens": 0}
        answer_cost = 0.0
    else:
        t = time.perf_counter()
        chunks = retrieve(q_emb)
        timing["retrieval_ms"] = ms_since(t)

        if req.use_router:
            model, route_reason = route(req.question, SMALL_MODEL, LARGE_MODEL)
        else:
            model, route_reason = LARGE_MODEL, "router_off"

        t = time.perf_counter()
        out = generate(req.question, chunks, model)
        answer_cost = cost_usd(model, out["input_tokens"], out["output_tokens"])
        # Cascade: kucuk model "bilmiyorum" derse ayni context ile buyuk modele sor
        if model == SMALL_MODEL and is_refusal(out["answer"]):
            first = out
            out = generate(req.question, chunks, LARGE_MODEL)
            answer_cost += cost_usd(LARGE_MODEL, out["input_tokens"], out["output_tokens"])
            out["input_tokens"] += first["input_tokens"]
            out["output_tokens"] += first["output_tokens"]
            route_reason += "+escalated"
        out["route_reason"] = route_reason
        timing["llm_ms"] = ms_since(t)

        out["sources"] = sorted({c["source"] for c in chunks})
        out["top_score"] = chunks[0]["score"] if chunks else None
        out["cache_hit"] = False
        if cacheable:
            cache.store(req.question, q_emb, out["answer"], out["sources"])

    # Maliyet = cevap LLM'i + (hit'te bile) cache dogrulama cagrilari
    out["cost_usd"] = round(
        answer_cost
        + cost_usd(SMALL_MODEL, cinfo["verify_input_tokens"], cinfo["verify_output_tokens"]),
        8,
    )
    out.update(cinfo)
    out.update(timing)
    out["total_ms"] = ms_since(t0)

    logger.log(question=req.question, use_cache=int(req.use_cache),
               use_router=int(req.use_router),
               cache_hit=int(out["cache_hit"]), run_tag=req.run_tag,
               **{k: out.get(k) for k in (
                   "best_similarity", "verify_calls", "verify_input_tokens",
                   "verify_output_tokens", "verify_error", "model", "route_reason",
                   "input_tokens", "output_tokens",
                   "cost_usd", "embed_ms", "cache_ms", "retrieval_ms", "llm_ms", "total_ms")})
    return out


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return round(statistics.quantiles(values, n=100, method="inclusive")[p - 1], 1)


@app.get("/stats")
def stats(run_tag: str | None = None):
    rows = logger.rows(run_tag)
    if not rows:
        return {"requests": 0}
    lat = [r["total_ms"] for r in rows]
    hits = sum(r["cache_hit"] for r in rows)
    return {
        "requests": len(rows),
        "cache_hit_rate": round(hits / len(rows), 3),
        "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 6),
        "avg_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows) / len(rows), 8),
        "p50_ms": pct(lat, 50),
        "p95_ms": pct(lat, 95),
        "model_counts": {m: sum(r["model"] == m for r in rows)
                         for m in (SMALL_MODEL, LARGE_MODEL)},   # cache hit'ler haric
        "avg_cache_ms": round(statistics.mean(r["cache_ms"] or 0 for r in rows)),
        "avg_llm_ms": round(statistics.mean(r["llm_ms"] or 0 for r in rows)),
    }


@app.get("/cache")
def cache_info():
    return {"size": cache.size(), "candidate_threshold": cache.candidate_threshold,
            "exact_threshold": cache.exact_threshold}


@app.delete("/cache")
def cache_clear():
    cache.clear()
    return {"cleared": True}