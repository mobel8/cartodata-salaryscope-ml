from salaryscope import features, regressor


def test_fit_and_predict_intervals(synth_offers):
    bundle = regressor.fit_final(synth_offers)
    x = features.build_features(synth_offers.head(8))
    out = regressor.predict(bundle, x)
    assert (out["low"] <= out["point"]).all()
    assert (out["point"] <= out["high"]).all()
    assert out["point"].notna().all()
    assert (out["point"] > 0).all()


def test_lightgbm_beats_trivial_baseline(synth_offers):
    metrics = regressor.evaluate(synth_offers)
    mae_lgbm = metrics["models"]["lightgbm"]["mae"]
    mae_median = metrics["models"]["median_global"]["mae"]
    # Sur des données où le salaire dépend des features, l'arbre doit battre la médiane.
    assert mae_lgbm <= mae_median
    assert metrics["n_groups"] >= 5
