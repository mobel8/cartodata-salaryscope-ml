from salaryscope import classifier


def test_train_and_predict(synth_offers):
    out = classifier.train_classifier(synth_offers)
    assert 0.0 <= out["metrics"]["f1_macro"] <= 1.0
    assert 0.0 <= out["metrics"]["f1_micro"] <= 1.0
    assert out["metrics"]["n_classes"] >= 2
    fam = classifier.predict_famille(out["model"], "data scientist machine learning")
    assert fam and "famille" in fam[0] and "proba" in fam[0]
