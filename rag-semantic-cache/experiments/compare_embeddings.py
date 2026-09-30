"""
compare_embeddings.py — Cache icin hangi embedding modeli ve hangi esik?
Ayni anlamdaki (SAME) ciftler yuksek, farkli anlamdakiler (DIFF) dusuk skor almali.
Iyi model = SAME ile DIFF arasinda net bir bosluk birakan model.
Calistir: python compare_embeddings.py
"""
from sentence_transformers import SentenceTransformer

PAIRS = [
    # (soru A, soru B, ayni anlam mi?)
    ("How do I enter PTO in Workday?", "How can I log my time off in Workday?", True),
    ("How do I enter PTO in Workday?", "How do I request vacation days in Workday?", True),
    ("What is the parental leave policy?", "How does parental leave work at GitLab?", True),
    ("How do I get paid for jury duty?", "Is jury duty leave paid?", True),
    ("Who approves my time off?", "Do I need manager approval for PTO?", True),
    ("How do I enter PTO in Workday?", "How do I cancel PTO in Workday?", False),
    ("What is the parental leave policy?", "What is the sick leave policy?", False),
    ("How do I get paid for jury duty?", "How do I get reimbursed for travel?", False),
    ("Who approves my time off?", "Who approves my expense report?", False),
]

MODELS = ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5", "BAAI/bge-base-en-v1.5"]

if __name__ == "__main__":
    for name in MODELS:
        m = SentenceTransformer(name)
        same, diff = [], []
        print(f"\n=== {name} ===")
        for a, b, is_same in PAIRS:
            ea, eb = m.encode([a, b], normalize_embeddings=True)
            sim = float(ea @ eb)
            (same if is_same else diff).append(sim)
            print(f"  {'SAME' if is_same else 'DIFF'}  {sim:.3f}  {a!r} vs {b!r}")
        gap = min(same) - max(diff)
        print(f"  min SAME={min(same):.3f}  max DIFF={max(diff):.3f}  bosluk={gap:+.3f}")
        if gap > 0:
            print(f"  -> onerilen esik ~ {(min(same) + max(diff)) / 2:.2f}")
        else:
            print("  -> bosluk yok: bu modelle guvenli bir esik secilemez")