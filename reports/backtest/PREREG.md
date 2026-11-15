# Backtest pre-registration

Written before the first run of the risk-model and backtest versions it describes. Every number we report comes from these fixed choices.

- **Decision:** at the end of 2018, using only data dated ≤2018. No sponsor data.
- **Universe:** rent-controlled-proxy buildings with 5–25 units, excluding Small Sites buildings bought ≤2018.
- **Value (households protected):** buyout tenants at the building in 2019–24, plus the building's unit share of Ellis/owner-move-in notices on its block-face in 2019–24. We assume a purchase prevents all of it, and the same assumption applies to every strategy.
- **Cost:** one hedonic model of post-sale assessed value, fitted on 2014–18 sales and applied to every building.
- **Budget (1×):** predicted cost of the city's 2019–23 Small Sites picks in the universe.
- **Primary comparison:** households protected at 1×, our knapsack (Tweedie value head) vs. **city policy**.
  - City policy is a logistic model of which buildings the city chose, applied to buildings it did not buy.
  - The city's own picks are censored, because buying a building prevents its later displacement.
  - 95% CI from a block-face cluster bootstrap (1000 draws), with the selections held fixed.
- **Secondary results:**
  - the budget sweep at 1, 2, 3, 5, 10 and 20× against random fills (300), the recently-sold heuristic and an oracle upper bound;
  - at 1×: the raw city picks (censored, reported for transparency) and 5-NN lookalikes;
  - fairness: share of protected households by district, size band, and ACS 2014–18 tract income and non-white tercile.
- **Not allowed:** changing the value definition, universe, budget or primary comparison after seeing results. If any of these changes, the change is logged here with its date and reason, and both results are reported.
