"""
judge.py — Baseline ve system kosularini karsilastir, kaliteyi LLM ile puanla, ozet cikar.

Kalite olcumu (LLM-as-judge, mutlak puanlama): Her iki kosunun HER cevabi ayri ayri
puanlanir: "Bu cevap sorulan soruyu dogrudan ve somut sekilde cevapliyor mu?" -> PASS / FAIL.
Sonra iki kosunun PASS oranlari karsilastirilir.
Neden cevaplari birbiriyle kiyaslamiyoruz: ilk denemede judge neredeyse ayni iki metne
"WORSE" dedi ve baseline'in kendi hatalarini (ornegin "I don't know") goremedi.

Kullanim (proje klasorunden, iki kosu bittikten sonra):
  set GROQ_API_KEY=...
  python eval\\judge.py              # puanla + ozet
  python eval\\judge.py --no-judge   # sadece maliyet/latency/cache ozeti (LLM cagrisi yok)

Yarida kesilirse tekrar calistir: puanlanmis cevaplari atlar.
Cikti: eval/grades.jsonl, eval/summary.json, eval/summary.md (README'ye yapistirilabilir)
"""
import argparse
import json
import re
import statistics
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
JUDGE_MODEL = "openai/gpt-oss-120b"

JUDGE_PROMPT = """You are grading an assistant that answers questions about the GitLab employee handbook.

PASS if the ANSWER directly and specifically answers the QUESTION as it was asked.
FAIL if the answer:
- says it doesn't know, is unsure, or that the information is not available
- answers a different or narrower question than the one asked
- is so vague that the asker could not act on it

Do not judge style, length, or formatting.

QUESTION: {question}

ANSWER:
{answer}

Reply with exactly one word: PASS or FAIL."""


def load_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l]
    return {r["id"]: r for r in rows}


def pct(values: list[float], p: int) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[p - 1]


def run_judge(runs: dict[str, dict], delay: float):
    from groq import Groq
    llm = Groq(max_retries=5)
    out_path = HERE / "grades.jsonl"
    done = load_grades()
    todo = [(run, qid) for run, rows in runs.items() for qid in rows if (run, qid) not in done]
    print(f"judge: {len(todo)} cevap kaldi, tahmini ~{len(todo) * (delay + 2) / 60:.0f} dk")
    with out_path.open("a", encoding="utf-8") as f:
        for n, (run, qid) in enumerate(todo, 1):
            row = runs[run][qid]
            r = llm.chat.completions.create(
                model=JUDGE_MODEL, temperature=0,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                    question=row["question"], answer=row["answer"])}],
            )
            text = (r.choices[0].message.content or "").upper()
            m = re.search(r"\b(PASS|FAIL)\b", text)
            grade = m.group(1) if m else "UNPARSED"
            f.write(json.dumps({"run": run, "id": qid, "grade": grade}) + "\n")
            f.flush()
            print(f"  [{n}/{len(todo)}] {run:<8} {qid} {grade:<5} {row['question'][:55]}")
            if n < len(todo):
                time.sleep(delay)


def load_grades() -> dict[tuple, str]:
    p = HERE / "grades.jsonl"
    if not p.exists():
        return {}
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l]
    return {(r["run"], r["id"]): r["grade"] for r in rows}


def summarize(base: dict, system: dict, grades: dict) -> dict:
    ids = [i for i in system if i in base]
    B = [base[i] for i in ids]
    S = [system[i] for i in ids]
    group_of = {r["question"]: r["group"] for r in S}

    cost_b = sum(r["cost_usd"] for r in B)
    cost_s = sum(r["cost_usd"] for r in S)
    lat_b = [r["total_ms"] for r in B]
    lat_s = [r["total_ms"] for r in S]

    hits = [r for r in S if r["cache_hit"]]
    false_hits = [r for r in hits if group_of.get(r["cached_question"]) != r["group"]]
    paraphrases = [r for r in S if r["kind"] == "paraphrase"]
    misses = [r for r in S if not r["cache_hit"]]
    small = [r for r in misses if (r["model"] or "").endswith("-20b")]   # "120b" de "20b" icerir!

    escalated = [r for r in misses if "+escalated" in (r.get("route_reason") or "")]

    def pass_rate(run: str, rows: list[dict]) -> dict:
        g = [grades[(run, r["id"])] for r in rows if (run, r["id"]) in grades]
        return {"n": len(g), "pass": g.count("PASS"),
                "pass_rate": round(g.count("PASS") / len(g), 3) if g else None}

    return {
        "questions": len(ids),
        "cost_usd": {"baseline": round(cost_b, 6), "system": round(cost_s, 6),
                     "reduction": round(1 - cost_s / cost_b, 3) if cost_b else None},
        "latency_ms": {"baseline_p50": round(pct(lat_b, 50)), "baseline_p95": round(pct(lat_b, 95)),
                       "system_p50": round(pct(lat_s, 50)), "system_p95": round(pct(lat_s, 95))},
        "cache": {"hit_rate": round(len(hits) / len(S), 3),
                  "paraphrase_hit_rate": round(sum(r["cache_hit"] for r in paraphrases)
                                               / len(paraphrases), 3) if paraphrases else None,
                  "hits": len(hits), "false_hits": len(false_hits),
                  "false_hit_ids": [r["id"] for r in false_hits],
                  "verify_errors": sum(1 for r in S if r.get("verify_error"))},
        "router": {"misses": len(misses), "to_small_model": len(small),
                   "escalated_to_large": len(escalated),
                   "small_share": round(len(small) / len(misses), 3) if misses else None,
                   "reasons": dict(Counter(r["route_reason"] for r in misses))},
        "quality": {
            "baseline": pass_rate("baseline", B),
            "system": pass_rate("system", S),
            "system_cache_hits": pass_rate("system", hits),
            "system_small_model": pass_rate("system", small),
            "system_large_model": pass_rate("system", [r for r in misses if r not in small]),
            "system_escalated": pass_rate("system", escalated),
        } if grades else None,
    }


def to_markdown(s: dict) -> str:
    c, l, k, r = s["cost_usd"], s["latency_ms"], s["cache"], s["router"]
    lines = [
        f"## Sonuclar ({s['questions']} soru)",
        "",
        "| Metrik | Baseline | Sistem | Degisim |",
        "|---|---|---|---|",
        f"| Toplam maliyet | ${c['baseline']:.4f} | ${c['system']:.4f} | "
        f"{-c['reduction'] * 100:+.0f}% |",
        f"| p50 latency | {l['baseline_p50']} ms | {l['system_p50']} ms | "
        f"{(l['system_p50'] / l['baseline_p50'] - 1) * 100:+.0f}% |",
        f"| p95 latency | {l['baseline_p95']} ms | {l['system_p95']} ms | "
        f"{(l['system_p95'] / l['baseline_p95'] - 1) * 100:+.0f}% |",
        "",
        f"- Cache hit orani: {k['hit_rate']:.0%} (farkli ifadeli sorularda {k['paraphrase_hit_rate']:.0%})",
        f"- Yanlis cache hit: {k['false_hits']} / {k['hits']}",
        f"- Dogrulama hatasi/timeout: {k['verify_errors']}",
        f"- Router: cache miss'lerin {r['small_share']:.0%}'i kucuk modele gitti",
    ]
    lines.append(f"- Kucuk model 'bilmiyorum' deyip buyuge yukseltilen: {r['escalated_to_large']}")
    q = s.get("quality")
    if q:
        lines += ["", "| Kalite (cevap soruyu karsiliyor mu?) | n | PASS | PASS orani |",
                  "|---|---|---|---|"]
        for name, v in q.items():
            if v["n"]:
                lines.append(f"| {name} | {v['n']} | {v['pass']} | {v['pass_rate']:.0%} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--delay", type=float, default=6.0)
    args = ap.parse_args()

    base = load_jsonl(HERE / "results_baseline.jsonl")
    system = load_jsonl(HERE / "results_system.jsonl")
    if not base or not system:
        raise SystemExit("Once iki kosuyu da calistir: run_eval.py --run baseline / --run system")

    if not args.no_judge:
        run_judge({"baseline": base, "system": system}, args.delay)
    grades = load_grades()

    summary = summarize(base, system, grades)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = to_markdown(summary)
    (HERE / "summary.md").write_text(md, encoding="utf-8")
    print("\n" + md)


if __name__ == "__main__":
    main()