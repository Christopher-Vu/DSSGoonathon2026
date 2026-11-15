# Video (≤ 5 min) + deck plan

Record the site with **P** (guided tour) for the map beats; cut to slides where a table or diagram reads better.
Numbers come from `web/public/data/headline.json` (filled by `python -m models.demo.export`); `{…}` = fill from there.

| # | Time | Medium | On screen | Voice-over (draft) |
|---|---|---|---|---|
| 1 | 0:00–0:25 | Site: hero (orbiting 3D risk map) | Big counter → **{vs_city}×** | "In 2018 this block looked ordinary. By 2022, its tenants were gone. We built a model that saw it coming, and showed how the city could have bought those buildings first." |
| 2 | 0:25–0:45 | **Slide** | Pipeline diagram: city records → hexagon tokens → world model → risk → budget optimizer → patient sim | "Every sale, permit, eviction notice and buyout since 2007, as a sequence per hexagon." |
| 3 | 0:45–1:25 | Site: world model view | RGB embedding map; click Mission hexagon → similar blocks rise; 2016↔2018 toggle | "No one told it where the Mission ends. It learned it. Click a block, and it finds every block like it." |
| 4 | 1:25–1:45 | **Slide** | Probe-lift table (locked test) + λ-sweep | "On held-out neighborhoods, the embeddings predict evictions, buyouts and overcrowding better than hand-made features: +{lift}." |
| 5 | 1:45–2:25 | Site: frozen in 2018 → play | Risk surface, then 2019–24 cases landing | "Frozen at the end of 2018. Then 2019 to 2024 happened. {top_decile_share}% of cases landed in the top tenth of hexagons." |
| 6 | 2:25–3:20 | Site: budget slider | Drag 1× → 5×; counters; sweep curve | "Same money the city spent. The model protects {model_1x} households to the city pattern's {city_1x}: {vs_city}× across budgets, every budget." |
| 7 | 3:20–3:35 | Site: futures | City fan, then one hexagon | "Not a point guess: 1,000 futures. Reality landed in the 80% band {coverage}% of the time." |
| 8 | 3:35–4:15 | Site: patient sim | Arcs out of SF; histogram; persona cards | "Displacement isn't just a move. Displaced households lose {median} minutes to the nearest clinic; {pct}% lose 30-minute access." |
| 9 | 4:15–4:45 | **Slide** | Fairness table + 3 caveats + ethics (hexagons only) | "Who it protects, what it assumes, and why we never publish a building list." |
| 10 | 4:45–5:00 | **Slide** | Repo, `make all`, site URL | "Everything reproduces from raw data with one command." |

## Deck (8 slides)
1. Title + headline number
2. Problem: Small Sites buys after the trouble; prevention needs a forecast
3. Pipeline diagram (beat 2)
4. World model: embedding map screenshot + probe lift (locked test) + λ sweep
5. Backtest: sweep table 1/2/3/5× with CIs, method (pre-registered, frozen in 2018, block-face bootstrap)
6. Risk metrics: PR-AUC / precision@k / capture vs. city pattern and base rate; futures coverage
7. Patient sim: headline + breakdown by car / age / language
8. Honesty: fairness, caveats, ethics, reproducibility
