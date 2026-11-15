"""Baseline candidate = run-1 model: LightGBM Tweedie on y_val (h=3), trained on every snapshot with full labels."""
import lightgbm as lgb

FEATS = ["units", "age", "owner_occ", "yrs_since_sale", "n_sales_10y", "log_last_jump", "sold_2y",
         "log_assessed_per_unit", "bf_share", "bf_rc_units", "bf_ellis_omi_3y", "bf_ellis_omi_10y",
         "bf_other_evict_3y", "bf_buyout_3y", "bldg_buyout_3y", "permits_3y", "remodel_3y", "unit_removal_10y",
         "log_permit_cost_3y", "cell_sales_3y", "supervisor_district"]
P = dict(objective="tweedie", tweedie_variance_power=1.3, learning_rate=0.03, num_leaves=15, min_data_in_leaf=300,
         feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1, seed=0)


def fit_predict(ctx, rounds=600):
    tr = ctx.labels(3)
    tr = tr[tr.t >= 2013]  # buyout data starts 2015-03
    m = lgb.train(P, lgb.Dataset(tr[FEATS], tr.y_val, categorical_feature=["supervisor_district"]), rounds)
    return m.predict(ctx.score[FEATS])
