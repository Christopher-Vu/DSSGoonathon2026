# Brief: video script + shot list

Goal: a word-for-word voice-over script and shot list for the ≤5-min submission video, plus the slide content where a beat is better on slides. You have the whole repo; read, don't run heavy compute.

## Read, in order
1. `PRD.md` — deliverables (repo, deck, ≤5-min video), required beat order (§ near line 18), ethics points judges will ask (near line 210).
2. `docs/VIDEO_PLAN.md` — current 10-beat draft + 8-slide deck outline. Refine this; `{…}` are placeholders.
3. `STATUS.md` — every final number with context (risk backtest, metrics, locked WM probe test, futures).
4. `web/public/data/headline.json` — exact values for placeholders. Also `sweep.json` (budget curve, CIs), `fairness.json`, `futures.json`. (gitignored; local only — if missing, use STATUS.md.)
5. `reports/backtest/PREREG.md` — what was pre-registered; keeps method claims honest.
6. `web/src/App.jsx` — what each site section actually shows and its on-screen copy; press **P** on the site for the guided tour.
7. Live site: https://sf-world-model.vercel.app — click through every section before writing shots.
8. Background if needed: `models/world_model/README.md`, `sim/patient/HANDOFF.md`, `reports/risk/ITERATE_RESULTS.md`.

## Rules (hard)
- Headline model is `c_wm_oos_blend` (world model + sale-recency prior). Never mention or show better-scoring non-WM models.
- No rule-of-thumb comparison anywhere. Comparisons are model vs. city pattern vs. random only.
- Hexagons only; never name or show buildings or addresses.
- Don't overstate weaknesses, don't hunt for failures; keep the real ones as stated on the site (futures over-forecast citywide; backtest CI at 1× includes 0 — only 0.5×, 1.5×, 2× exclude it).
- Patient sim numbers (median Δ minutes to clinic, % losing 30-min access, breakdowns) are **pending** — leave as placeholders; that beat gets recorded separately later.
- Every number in the script must trace to a file above. Cite the file next to each number in the draft.

## Deliverable
Write `docs/VIDEO_SCRIPT.md`:
- Per beat: time range, medium (site / slide), exact on-screen action (section, clicks, slider moves), voice-over text.
- ~150 words/min; total ≤ 4:45 of speech.
- Slide content for slide beats (title, 3–5 lines or the table to show).
- A numbers table at the end: each figure, its value, its source file.
