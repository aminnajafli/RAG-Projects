## Sonuclar (54 soru)

| Metrik | Baseline | Sistem | Degisim |
|---|---|---|---|
| Toplam maliyet | $0.0204 | $0.0134 | -34% |
| p50 latency | 1495 ms | 4050 ms | +171% |
| p95 latency | 25027 ms | 34691 ms | +39% |

- Cache hit orani: 24% (farkli ifadeli sorularda 50%)
- Yanlis cache hit: 1 / 13
- Dogrulama hatasi/timeout: 9
- Router: cache miss'lerin 66%'i kucuk modele gitti
- Kucuk model 'bilmiyorum' deyip buyuge yukseltilen: 5

| Kalite (cevap soruyu karsiliyor mu?) | n | PASS | PASS orani |
|---|---|---|---|
| baseline | 54 | 49 | 91% |
| system | 54 | 48 | 89% |
| system_cache_hits | 13 | 12 | 92% |
| system_small_model | 27 | 26 | 96% |
| system_large_model | 14 | 10 | 71% |
| system_escalated | 5 | 1 | 20% |
