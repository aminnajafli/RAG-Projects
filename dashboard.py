"""
dashboard.py — logs.db ve eval/summary.json'dan tek dosyalik bir HTML dashboard uretir.
Sadece Python standart kutuphanesi kullanir (Streamlit / pyarrow gerekmez).

Calistir:  python dashboard.py      -> dashboard.html olusur, tarayicida acilir
"""
import json
import sqlite3
import webbrowser
from pathlib import Path

DB = Path("logs.db")
SUMMARY = Path("eval/summary.json")
OUT = Path("dashboard.html")

COLS = ["ts", "run_tag", "question", "use_cache", "cache_hit", "best_similarity",
        "verify_error", "model", "route_reason", "cost_usd", "total_ms"]


def load_rows() -> list[dict]:
    if not DB.exists():
        raise SystemExit("logs.db bulunamadi. Once birkac /ask istegi at.")
    with sqlite3.connect(DB) as conn:
        cur = conn.execute(f"SELECT {', '.join(COLS)} FROM requests ORDER BY ts")
        rows = [dict(zip(COLS, r)) for r in cur.fetchall()]
    for r in rows:
        r["run_tag"] = r["run_tag"] or "manual"
        if r["cache_hit"]:
            r["path"] = "cache hit"
        elif "escalated" in (r["route_reason"] or ""):
            r["path"] = "escalated"
        elif (r["model"] or "").endswith("-20b"):
            r["path"] = "small model"
        else:
            r["path"] = "large model"
    return rows


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Cost Dashboard</title>
<style>
:root{--bg:#f9f9f7;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e4e3de;
 --c1:#2a78d6;--c2:#eb6834;--c3:#1baf7a;--c4:#eda100;--good:#008300;--bad:#c4312f}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#0d0d0d;--card:#1a1a19;--ink:#fff;
 --ink2:#c3c2b7;--muted:#8a8984;--line:#2e2e2c;--c1:#3987e5;--c2:#d95926;--c3:#199e70;--c4:#c98500;
 --good:#3fb950;--bad:#f06a67}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0 0 4px} h2{font-size:15px;margin:0 0 12px}
.sub{color:var(--ink2);margin:0 0 20px}
.filters{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:20px}
.filters button{border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:999px;
 padding:6px 14px;cursor:pointer;font:inherit}
.filters button[aria-pressed="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.grid{display:grid;gap:16px}
.kpis{grid-template-columns:repeat(auto-fit,minmax(170px,1fr));margin-bottom:16px}
.three{grid-template-columns:repeat(auto-fit,minmax(300px,1fr));margin-bottom:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0}
.kpi .label{color:var(--ink2);font-size:13px} .kpi .val{font-size:28px;font-weight:600;margin-top:2px}
.kpi .delta{font-size:13px;margin-top:2px} .good{color:var(--good)} .bad{color:var(--bad)}
.note{color:var(--muted);font-size:12px;margin-top:8px}
svg text{fill:var(--ink2);font-size:12px} svg .val{fill:var(--ink)}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--ink2);font-size:12px;margin-top:8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);padding:6px 9px;border-radius:6px;
 font-size:12px;max-width:340px;display:none;z-index:9}
details{margin-top:16px} summary{cursor:pointer;color:var(--ink2)}
.tbl{overflow-x:auto} table{border-collapse:collapse;width:100%;font-size:12px;margin-top:10px}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);white-space:nowrap}
td.q{white-space:normal;min-width:260px}
</style></head><body><main>
<h1>RAG cost &amp; cache dashboard</h1>
<p class="sub">GitLab Handbook RAG · semantic cache + router · generated __GENERATED__</p>
<div class="filters" id="filters" role="group" aria-label="Run filter"></div>
<div class="grid kpis" id="kpis"></div>
<div class="grid three">
 <div class="card"><h2>Which path requests took</h2><div id="ch-count"></div></div>
 <div class="card"><h2>Average cost per request</h2><div id="ch-cost"></div></div>
 <div class="card"><h2>Median latency</h2><div id="ch-lat"></div></div>
</div>
<div class="card" style="margin-bottom:16px"><h2>Cache candidates: embedding similarity vs. verifier decision</h2>
 <div id="ch-sim"></div>
 <div class="legend"><span><i style="background:var(--c1)"></i>hit (verifier: same answer)</span>
 <span><i style="background:var(--muted)"></i>miss</span><span>dashed line = candidate threshold 0.70</span></div>
 <div class="note">Misses to the right of the line are questions the verifier rejected despite high similarity,
 e.g. "enter PTO" vs "cancel PTO" (0.87). Similarity alone is not enough.</div></div>
<div class="card" id="eval-card"><h2 id="eval-title">Evaluation</h2><div class="grid kpis" id="eval"></div>
 <div class="note">Baseline = no cache, every question to gpt-oss-120b. Quality = share of answers an LLM judge
 graded as directly answering the question.</div></div>
<details><summary>Raw requests (table)</summary><div class="tbl"><table id="tbl"></table></div></details>
<div id="tip"></div>
</main>
<script>
const ROWS = __ROWS__;
const SUMMARY = __SUMMARY__;
const PATHS = ["cache hit","small model","large model","escalated"];
const COLOR = {"cache hit":"var(--c1)","small model":"var(--c2)","large model":"var(--c3)","escalated":"var(--c4)"};
const $ = id => document.getElementById(id);
const tip = $("tip");
function showTip(e, html){ tip.innerHTML = html; tip.style.display = "block";
  const x = Math.min(e.clientX + 12, innerWidth - tip.offsetWidth - 8); tip.style.left = x + "px";
  tip.style.top = (e.clientY + 14) + "px"; }
function hideTip(){ tip.style.display = "none"; }
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const median = a => { if(!a.length) return 0; const s=[...a].sort((x,y)=>x-y), m=s.length>>1; return s.length%2? s[m] : (s[m-1]+s[m])/2; };
const q95 = a => { if(!a.length) return 0; const s=[...a].sort((x,y)=>x-y); return s[Math.min(s.length-1, Math.ceil(.95*s.length)-1)]; };
const fmtUsd = v => "$" + (v < 0.01 ? v.toFixed(6) : v.toFixed(4));
const fmtMs = v => v >= 1000 ? (v/1000).toFixed(1) + " s" : Math.round(v) + " ms";

const tags = [...new Set(ROWS.map(r => r.run_tag))].sort();
let active = tags.includes("system") ? "system" : tags[0];
function renderFilters(){
  $("filters").innerHTML = '<span style="color:var(--ink2)">Run:</span>' + tags.map(t =>
    `<button aria-pressed="${t===active}" data-t="${esc(t)}">${esc(t)} (${ROWS.filter(r=>r.run_tag===t).length})</button>`).join("");
  $("filters").querySelectorAll("button").forEach(b => b.onclick = () => { active = b.dataset.t; render(); });
}

function kpi(label, val, delta, cls){ return `<div class="card kpi"><div class="label">${label}</div>
  <div class="val">${val}</div>${delta ? `<div class="delta ${cls||""}">${delta}</div>` : ""}</div>`; }

function bars(el, data, key, fmt){
  const W = 360, rowH = 34, left = 96, right = 86, H = data.length * rowH + 8;
  const max = Math.max(...data.map(d => d[key]), 1e-12);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img">`;
  data.forEach((d, i) => {
    const y = i*rowH + 6, w = Math.max(2, (W-left-right) * d[key] / max);
    s += `<text x="${left-8}" y="${y+15}" text-anchor="end">${d.path}</text>`;
    s += `<rect class="hit" x="${left}" y="${y}" width="${W-left}" height="22" fill="transparent" data-i="${i}"/>`;
    s += `<path d="M${left},${y+2} h${w-4} a4,4 0 0 1 4,4 v10 a4,4 0 0 1 -4,4 h-${w-4} z" fill="${COLOR[d.path]}" pointer-events="none"/>`;
    s += `<text class="val" x="${left+w+6}" y="${y+15}" pointer-events="none">${fmt(d[key])}</text>`;
  });
  el.innerHTML = s + "</svg>";
  el.querySelectorAll("rect.hit").forEach(r => {
    const d = data[+r.dataset.i];
    r.onmousemove = e => showTip(e, `<b>${d.path}</b><br>${d.n} requests · avg ${fmtUsd(d.cost)}<br>median ${fmtMs(d.med)} · p95 ${fmtMs(d.p95)}`);
    r.onmouseleave = hideTip;
  });
}

function strip(el, rows){
  const pts = rows.filter(r => r.use_cache && r.best_similarity != null);
  if(!pts.length){ el.innerHTML = '<div class="note">No cache lookups in this run.</div>'; return; }
  const W = 900, H = 120, L = 48, R = 12, lo = 0.4, hi = 1.0, X = v => L + (W-L-R) * (Math.max(lo, v) - lo) / (hi - lo);
  let s = `<svg viewBox="0 0 ${W} ${H}" width="100%" role="img">`;
  for(let t = 0.4; t <= 1.0001; t += 0.1){ const x = X(t);
    s += `<line x1="${x}" x2="${x}" y1="8" y2="${H-24}" stroke="var(--line)"/><text x="${x}" y="${H-6}" text-anchor="middle">${t.toFixed(1)}</text>`; }
  s += `<line x1="${X(.7)}" x2="${X(.7)}" y1="4" y2="${H-24}" stroke="var(--ink2)" stroke-dasharray="4 4"/>`;
  s += `<text x="${L-8}" y="34" text-anchor="end">hit</text><text x="${L-8}" y="78" text-anchor="end">miss</text>`;
  pts.forEach((r, i) => {
    const y = (r.cache_hit ? 30 : 74) + ((i * 7) % 15) - 7;
    s += `<circle cx="${X(r.best_similarity)}" cy="${y}" r="5" fill="${r.cache_hit ? "var(--c1)" : "var(--muted)"}"
      stroke="var(--card)" stroke-width="2" data-i="${i}" style="cursor:default"/>`;
  });
  el.innerHTML = s + "</svg>";
  el.querySelectorAll("circle").forEach(c => { const r = pts[+c.dataset.i];
    c.onmousemove = e => showTip(e, `${esc(r.question)}<br>similarity ${r.best_similarity.toFixed(3)} · ${r.cache_hit ? "hit" : "miss"}${r.verify_error ? "<br>verifier: " + esc(r.verify_error).slice(0,40) : ""}`);
    c.onmouseleave = hideTip; });
}

function render(){
  renderFilters();
  const rows = ROWS.filter(r => r.run_tag === active);
  const cost = rows.reduce((a, r) => a + (r.cost_usd || 0), 0);
  const hits = rows.filter(r => r.cache_hit).length;
  $("kpis").innerHTML = kpi("Requests", rows.length) + kpi("Cache hit rate", Math.round(100*hits/rows.length) + "%")
    + kpi("Total LLM cost", fmtUsd(cost)) + kpi("Avg cost / 1K requests", "$" + (1000*cost/rows.length).toFixed(3))
    + kpi("Median latency", fmtMs(median(rows.map(r => r.total_ms))), "Groq free tier: latency is noisy", "");
  const data = PATHS.map(p => { const g = rows.filter(r => r.path === p); const ms = g.map(r => r.total_ms);
    return {path: p, n: g.length, cost: g.length ? g.reduce((a, r) => a + r.cost_usd, 0) / g.length : 0,
            med: median(ms), p95: q95(ms)}; }).filter(d => d.n);
  bars($("ch-count"), data, "n", v => v);
  bars($("ch-cost"), data, "cost", fmtUsd);
  bars($("ch-lat"), data, "med", fmtMs);
  strip($("ch-sim"), rows);
  $("tbl").innerHTML = "<tr><th>Question</th><th>Path</th><th>Route</th><th>Similarity</th><th>Cost</th><th>Latency</th></tr>"
    + [...rows].reverse().map(r => `<tr><td class="q">${esc(r.question)}</td><td>${r.path}</td><td>${esc(r.route_reason || "")}</td>
      <td>${r.best_similarity == null ? "" : r.best_similarity.toFixed(3)}</td><td>${fmtUsd(r.cost_usd || 0)}</td><td>${fmtMs(r.total_ms)}</td></tr>`).join("");
}

if(SUMMARY){
  const s = SUMMARY, q = s.quality || {};
  $("eval-title").textContent = `Evaluation: baseline vs system (${s.questions} questions)`;
  const dq = q.system ? Math.round(100*(q.system.pass_rate - q.baseline.pass_rate)) : null;
  $("eval").innerHTML = kpi("LLM cost", fmtUsd(s.cost_usd.system), `${Math.round(-100*s.cost_usd.reduction)}% vs baseline ${fmtUsd(s.cost_usd.baseline)}`, "good")
    + (q.system ? kpi("Quality (PASS)", Math.round(100*q.system.pass_rate) + "%", `${dq >= 0 ? "+" : ""}${dq} pts vs baseline ${Math.round(100*q.baseline.pass_rate)}%`, dq < 0 ? "bad" : "good") : "")
    + kpi("Cache hits on paraphrases", Math.round(100*s.cache.paraphrase_hit_rate) + "%")
    + kpi("Wrong cache hits", `${s.cache.false_hits} / ${s.cache.hits}`)
    + kpi("Misses routed to small model", Math.round(100*s.router.small_share) + "%");
} else { $("eval-card").style.display = "none"; }
render();
</script></body></html>"""


def main():
    rows = load_rows()
    summary = json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else None
    from datetime import datetime
    html = (PAGE.replace("__ROWS__", json.dumps(rows, ensure_ascii=False).replace("</", "<\\/"))
                .replace("__SUMMARY__", json.dumps(summary))
                .replace("__GENERATED__", datetime.now().strftime("%Y-%m-%d %H:%M")))
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT} olusturuldu ({len(rows)} istek). Tarayicida aciliyor...")
    webbrowser.open(OUT.resolve().as_uri())


if __name__ == "__main__":
    main()
