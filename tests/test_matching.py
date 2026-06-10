from salaryscope import matching


def test_nearest_returns_k(synth_offers):
    idx = matching.build_index(synth_offers)
    res = matching.nearest(idx, "data scientist", k=3)
    assert len(res) == 3
    assert "similarite" in res[0]
    assert 0.0 <= res[0]["similarite"] <= 1.0


def test_market_percentile():
    dist = [10, 20, 30, 40]
    assert matching.market_percentile(dist, 25) == 50.0
    assert matching.market_percentile(dist, 5) == 0.0
    assert matching.market_percentile(dist, 100) == 100.0


def test_tension(synth_offers):
    t = matching.tension(synth_offers, departement="75")
    assert t["n_offres"] >= 0
    assert 0 <= t["part_marche_pct"] <= 100
