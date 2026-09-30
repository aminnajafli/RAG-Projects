"""
compare_llm_verifier.py — Cache dogrulamasi icin kucuk LLM'i (gpt-oss-20b) test et.
Soru: "Bu iki soru ayni cevabi mi gerektirir?" -> YES / NO
Calistir (PowerShell):
  $env:GROQ_API_KEY="keyin"
  .\\venv\\Scripts\\python.exe compare_llm_verifier.py
"""
import time

from groq import Groq

from compare_embeddings import PAIRS

MODEL = "openai/gpt-oss-20b"
llm = Groq(max_retries=5)

PROMPT = """Two users asked questions to an HR handbook assistant.
Would ONE answer fully and correctly answer BOTH questions?
Treat abbreviations and synonyms as equal (e.g. PTO = paid time off = vacation).
Different actions (enter vs cancel) or different topics (parental vs sick leave) are NOT the same.
Reply with exactly one word: YES or NO.

Question A: {a}
Question B: {b}"""


def same_question(a: str, b: str) -> tuple[bool, int, int]:
    t0 = time.perf_counter()
    r = llm.chat.completions.create(
        model=MODEL, temperature=0,
        messages=[{"role": "user", "content": PROMPT.format(a=a, b=b)}],
    )
    ans = r.choices[0].message.content.strip().upper()
    return ans.startswith("YES"), r.usage.total_tokens, round((time.perf_counter() - t0) * 1000)


correct = 0
for a, b, expected in PAIRS:
    pred, tokens, ms = same_question(a, b)
    ok = pred == expected
    correct += ok
    print(f"{'OK ' if ok else 'ERR'}  beklenen={'SAME' if expected else 'DIFF'}  "
          f"tahmin={'SAME' if pred else 'DIFF'}  {tokens} token  {ms}ms  {a!r} vs {b!r}")

print(f"\nDogruluk: {correct}/{len(PAIRS)}")