"""
compare_verifier.py — 2 asamali cache'in 2. asamasini test et.
Cross-encoder iki soruyu BIRLIKTE okur ve "ayni soru mu?" diye karar verir.
Quora duplicate-question verisiyle egitilmis model -> tam bizim isimiz.
Calistir: .\\venv\\Scripts\\python.exe compare_verifier.py
"""
from sentence_transformers import CrossEncoder

from compare_embeddings import PAIRS

MODELS = ["cross-encoder/quora-distilroberta-base", "cross-encoder/quora-roberta-base"]

for name in MODELS:
    ce = CrossEncoder(name)
    scores = ce.predict([(a, b) for a, b, _ in PAIRS])
    same, diff = [], []
    print(f"\n=== {name} ===")
    for (a, b, is_same), s in zip(PAIRS, scores):
        (same if is_same else diff).append(float(s))
        print(f"  {'SAME' if is_same else 'DIFF'}  {s:.3f}  {a!r} vs {b!r}")
    gap = min(same) - max(diff)
    print(f"  min SAME={min(same):.3f}  max DIFF={max(diff):.3f}  bosluk={gap:+.3f}")
    if gap > 0:
        print(f"  -> onerilen esik ~ {(min(same) + max(diff)) / 2:.2f}")
    else:
        print("  -> bosluk yok")