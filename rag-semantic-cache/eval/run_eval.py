"""
run_eval.py — Test setini calisan API'ye gonder, sonuclari kaydet.

Iki kosu:
  baseline : cache KAPALI, router KAPALI  -> her soru buyuk modele gider
  system   : cache ACIK,   router ACIK    -> bizim sistem

Kullanim (proje klasorunden, sunucu calisirken, ayri terminalde):
  python eval\\run_eval.py --run baseline
  python eval\\run_eval.py --run system

Yarida kesilirse ayni komutu tekrar calistir: bitenleri atlar, kaldigi yerden devam eder.
Groq ucretsiz limitine (120b: dakikada 8K token) takilmamak icin istekler arasi bekler.
429 alip SDK'nin beklemesi latency'yi sisirir ve karsilastirmayi bozar, bu yuzden bekleme onemli.
"""
import argparse
import json
import time
from pathlib import Path

import httpx

API = "http://127.0.0.1:8000"
HERE = Path(__file__).parent
MODES = {
    "baseline": {"use_cache": False, "use_router": False},
    "system": {"use_cache": True, "use_router": True},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=MODES, required=True)
    ap.add_argument("--delay", type=float, default=13.0, help="istekler arasi bekleme (sn)")
    args = ap.parse_args()

    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    out_path = HERE / f"results_{args.run}.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(line)["id"] for line in out_path.read_text(encoding="utf-8").splitlines() if line}

    client = httpx.Client(base_url=API, timeout=300)
    try:
        client.get("/cache").raise_for_status()
    except Exception:
        raise SystemExit(f"API'ye ulasilamadi ({API}). Once uvicorn'u baslat.")

    if args.run == "system" and not done:
        client.delete("/cache").raise_for_status()   # temiz cache ile basla
        print("cache temizlendi")

    todo = [q for q in questions if q["id"] not in done]
    print(f"{args.run}: {len(todo)} soru kaldi ({len(done)} tamamlanmis)"
          f", tahmini ~{len(todo) * (args.delay + 2) / 60:.0f} dk")

    with out_path.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            payload = {"question": q["question"], "run_tag": args.run, **MODES[args.run]}
            try:
                r = client.post("/ask", json=payload)
                r.raise_for_status()
            except Exception as e:
                print(f"  {q['id']} HATA: {e} -> durduruldu, tekrar calistirinca devam eder")
                break
            res = r.json()
            f.write(json.dumps({**q, **res}, ensure_ascii=False) + "\n")
            f.flush()
            tag = ("HIT " if res.get("cache_hit") else "miss") + \
                  f" {(res.get('model') or '-').split('/')[-1]:<12}"
            print(f"  [{i}/{len(todo)}] {q['id']} {tag} {res.get('total_ms'):>6}ms  {q['question'][:60]}")
            if i < len(todo):
                time.sleep(args.delay)

    print(f"bitti -> {out_path}")


if __name__ == "__main__":
    main()