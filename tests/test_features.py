import pandas as pd

from salaryscope import features


def test_build_features_columns(synth_offers):
    x = features.build_features(synth_offers)
    for col in ("title", "is_senior", "is_junior_titre", "n_skills", "skill_Python"):
        assert col in x.columns
    assert x["skill_Python"].dtype == bool


def test_seniority_detection():
    df = pd.DataFrame({"title": ["Lead Data Scientist", "Data Analyst Junior", "Data Engineer"]})
    x = features.build_features(df)
    assert x["is_senior"].tolist() == [True, False, False]
    assert x["is_junior_titre"].tolist() == [False, True, False]


def test_skill_flags():
    df = pd.DataFrame({"title": ["x"], "skills": [["Python", "SQL"]]})
    x = features.build_features(df)
    assert bool(x["skill_Python"].iloc[0]) is True
    assert bool(x["skill_Spark"].iloc[0]) is False
    assert int(x["n_skills"].iloc[0]) == 2


def test_groups(synth_offers):
    g = features.groups(synth_offers)
    assert g.nunique() <= 12
