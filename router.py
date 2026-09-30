"""
router.py — Cache miss'te soruyu hangi model cevaplasin? (sadece soru sinyali)
Ek LLM cagrisi yok, ~0ms. Basit soru -> kucuk model, karmasik soru -> buyuk model.

Karmasik sayilan sorular:
  - karsilastirma / neden / analiz isteyen kelimeler (compare, why, explain...)
  - birden fazla soru iceren (birden cok "?" veya "and" ile bagli iki soru)
  - uzun sorular (cok kosul/detay = cok parca birlestirme)
"""
import re

COMPLEX_PATTERNS = [
    r"\bcompar\w*", r"\bdifferen\w*", r"\bdiffer\b", r"\bvs\.?\b", r"\bversus\b",
    r"\bwhy\b", r"\bexplain\w*", r"\breason\w*",
    r"\bpros\b", r"\bcons\b", r"\btrade-?offs?\b", r"\badvantages?\b", r"\bdisadvantages?\b",
    r"\bimpact\w*", r"\baffect\w*", r"\brelationship\b",
    r"\bwhat happens (if|when)\b", r"\bshould i\b", r"\bbest way\b",
    r"\banaly[sz]\w*", r"\bsummari[sz]\w*", r"\boverview\b",
    r"\ball (the )?(ways|types|options|steps)\b", r"\blist (all|every)\b",
]
COMPLEX_RE = re.compile("|".join(COMPLEX_PATTERNS), re.IGNORECASE)
MULTI_PART_RE = re.compile(r"\?\s*(and|also|plus)?\s*\w.*\?", re.IGNORECASE)
MAX_SIMPLE_WORDS = 20


def route(question: str, small_model: str, large_model: str) -> tuple[str, str]:
    """(secilen model, sebep) doner. Sebep loglanir -> hangi kural ne kadar tetiklendi gorulur."""
    q = question.strip()
    m = COMPLEX_RE.search(q)
    if m:
        return large_model, f"keyword:{m.group(0).lower()}"
    if q.count("?") > 1 or MULTI_PART_RE.search(q):
        return large_model, "multi_part"
    words = len(q.split())
    if words > MAX_SIMPLE_WORDS:
        return large_model, f"long:{words}w"
    return small_model, "simple"