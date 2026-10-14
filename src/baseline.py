"""Baseline TF-IDF + régression logistique (jalon m1).

Usage : python -m src.baseline --config configs/base.yaml
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import load_ledgar
from src.metrics import report
from src.utils import load_config, save_json, set_seed


def main():
    cfg = load_config("configs/base.yaml")
    set_seed(cfg["seed"])
    ds, names = load_ledgar(cfg)
    rd = cfg["results_dir"]

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)
    Xtr = vec.fit_transform(ds["train"]["text"])  # fit sur le train uniquement
    Xva = vec.transform(ds["validation"]["text"])

    # Choix de C et du class_weight sur la VALIDATION, jamais sur le test
    best = None
    for C in (1.0, 5.0, 10.0):
        for cw in (None, "balanced"):
            clf = LogisticRegression(C=C, class_weight=cw, max_iter=2000)
            clf.fit(Xtr, ds["train"]["label"])
            m = report(ds["validation"]["label"], clf.predict(Xva), len(names))
            print(f"C={C} class_weight={cw} -> {m}")
            if best is None or m["macro_f1"] > best[0]["macro_f1"]:
                best = (m, clf, {"C": C, "class_weight": cw})

    val_metrics, clf, params = best
    Xte = vec.transform(ds["test"]["text"])
    y_pred = clf.predict(Xte)
    test_metrics = report(ds["test"]["label"], y_pred, len(names))
    save_json({"params": params, "validation": val_metrics, "test": test_metrics},
              f"{rd}/metrics_baseline.json")
    pd.DataFrame({"y_true": ds["test"]["label"], "y_pred": y_pred}).to_csv(
        f"{rd}/preds_baseline.csv", index=False)


if __name__ == "__main__":
    main()
