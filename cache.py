"""
cache.py — 2 asamali semantic cache
  1. Embedding ile aday bul (gevsek esik, hizli)
  2. Kucuk LLM ile dogrula: "hangi eski soru ayni cevabi gerektirir?" (kesin)
Neden 2 asama: embedding benzerligi "ayni konu"yu olcer, "ayni cevap"i degil.
("PTO gir" vs "PTO iptal et" = 0.87, ama cevaplari farkli.) Bkz. experiments/.

Tum adaylar TEK bir LLM cagrisinda dogrulanir (aday basina bir cagri degil),
ve dogrulamanin sert bir zaman butcesi vardir: yetismezse miss sayilir.
"""
import json
import re
import time
import uuid

import chromadb
from groq import Groq

CACHE_COLLECTION = "semantic_cache"

REFUSAL_RE = re.compile(
    r"^\W*(i\s*(do\s*n.?t|don.?t) know|i.?m (sorry|not sure|unable)|i (can(no|.?)t|could ?n.?t) find"
    r"|sorry\b|unfortunately\b)"
    r"|(context|provided (documents?|excerpts?|information)|handbook excerpts?) (does|do) not"
    r"|not (mentioned|specified|covered|included) in the (provided )?(context|documents?)",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    """'Bilmiyorum' turu cevap mi? (cache'e yazilmaz, kucuk modelde buyuge yukseltilir)"""
    if not (answer or "").strip():   # bos cevap (reasoning modeli tum token'i dusunmeye harcadi)
        return True
    return bool(REFUSAL_RE.search(answer[:300]))

VERIFY_PROMPT = """A user asked an HR handbook assistant a NEW question.
Below are previously answered questions. Which ONE of them would have exactly the
same correct answer as the NEW question?
Treat abbreviations and synonyms as equal (e.g. PTO = paid time off = vacation).
Different actions (enter vs cancel) or different topics (parental vs sick leave) are NOT the same.
Reply with only the number of the matching question, or 0 if none match.

NEW question: {new}

Previous questions:
{candidates}"""


class SemanticCache:
    def __init__(self, verify_model: str, path: str = "chroma_db",
                 candidate_threshold: float = 0.70,   # bunun altini hic dogrulama
                 exact_threshold: float = 0.97,       # bunun ustu dogrulamasiz hit
                 max_candidates: int = 3,
                 verify_timeout_s: float = 5.0,       # dogrulama butcesi, asilirsa miss
                 ttl_seconds: int = 24 * 3600):
        # Dogrulama icin ayri istemci: retry yok, kisa timeout.
        # Cache yavaslarsa beklemek yerine LLM'e gitmek daha iyi.
        self.llm = Groq(max_retries=0, timeout=verify_timeout_s)
        self.verify_model = verify_model
        self.client = chromadb.PersistentClient(path=path)
        self.candidate_threshold = candidate_threshold
        self.exact_threshold = exact_threshold
        self.max_candidates = max_candidates
        self.ttl = ttl_seconds
        self.col = self._get_col()

    def _get_col(self):
        return self.client.get_or_create_collection(
            CACHE_COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def _verify(self, question: str, candidates: list[str]) -> tuple[int, int, int, str | None]:
        """(eslesen adayin indexi veya -1, input token, output token, hata).
        Hata/timeout olursa guvenli tarafta kal: -1 (miss)."""
        listing = "\n".join(f"{i}. {q}" for i, q in enumerate(candidates, 1))
        try:
            r = self.llm.chat.completions.create(
                model=self.verify_model, temperature=0, reasoning_effort="low",
                messages=[{"role": "user", "content":
                           VERIFY_PROMPT.format(new=question, candidates=listing)}],
            )
        except Exception as e:
            err = f"{type(e).__name__}: {e}"[:200]
            print(f"[cache] verify hatasi, miss sayiliyor: {err}")
            return -1, 0, 0, err
        text = (r.choices[0].message.content or "").strip()
        m = re.search(r"\d+", text)
        n = int(m.group()) if m else 0
        idx = n - 1 if 1 <= n <= len(candidates) else -1
        return idx, r.usage.prompt_tokens, r.usage.completion_tokens, None

    def lookup(self, question: str, q_emb: list[float]) -> tuple[dict | None, dict]:
        """(hit veya None, istatistik) doner."""
        info = {"best_similarity": None, "verify_calls": 0,
                "verify_input_tokens": 0, "verify_output_tokens": 0,
                "verify_error": None}
        n = min(self.max_candidates, self.col.count())
        if n == 0:
            return None, info

        res = self.col.query(query_embeddings=[q_emb], n_results=n)
        info["best_similarity"] = round(1 - res["distances"][0][0], 3)

        # Esigi gecen ve suresi dolmamis adaylar
        cands = []
        for cid, cached_q, meta, dist in zip(res["ids"][0], res["documents"][0],
                                             res["metadatas"][0], res["distances"][0]):
            sim = round(1 - dist, 3)
            if sim < self.candidate_threshold:
                break                                    # sonrakiler daha da uzak
            if time.time() - meta["created_at"] > self.ttl:
                self.col.delete(ids=[cid])               # suresi dolmus
                continue
            cands.append((cached_q, meta, sim))
        if not cands:
            return None, info

        if cands[0][2] >= self.exact_threshold:          # neredeyse birebir ayni
            match = cands[0]
        else:
            idx, t_in, t_out, err = self._verify(question, [c[0] for c in cands])
            info.update(verify_calls=1, verify_input_tokens=t_in,
                        verify_output_tokens=t_out, verify_error=err)
            if idx < 0:
                return None, info
            match = cands[idx]

        cached_q, meta, sim = match
        return {
            "answer": meta["answer"],
            "sources": json.loads(meta["sources"]),
            "cached_question": cached_q,
            "cache_similarity": sim,
        }, info

    def store(self, question: str, q_emb: list[float], answer: str, sources: list[str]):
        # "Bilmiyorum" cevaplarini cache'leme: sonraki benzer sorular da kotu cevap alir
        if is_refusal(answer):
            return
        self.col.add(
            ids=[str(uuid.uuid4())],
            embeddings=[q_emb],
            documents=[question],
            metadatas=[{"answer": answer, "sources": json.dumps(sources),
                        "created_at": time.time()}],
        )

    def clear(self):
        # Dokumanlar yeniden ingest edilince cagir
        self.client.delete_collection(CACHE_COLLECTION)
        self.col = self._get_col()

    def size(self) -> int:
        return self.col.count()