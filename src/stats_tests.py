"""Comparaisons appariées sur le même test set (S06) : McNemar et bootstrap macro-F1.

Usage : python -m src.stats_tests --a results/preds_finetune.csv --b results/preds_lora.csv
"""
import argparse

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2

from src.metrics import macro_f1


def mcnemar(y_true, pred_a, pred_b) -> dict:
    ok_a, ok_b = pred_a == y_true, pred_b == y_true
    b = int(np.sum(ok_a & ~ok_b))  # seul A a raison
    c = int(np.sum(~ok_a & ok_b))  # seul B a raison
    if b + c < 25:  # test binomial exact
        p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
        return {"b": b, "c": c, "test": "exact binomial", "p_value": float(p)}
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # correction de continuité
    return {"b": b, "c": c, "test": "chi2", "chi2": float(stat),
            "p_value": float(chi2.sf(stat, df=1))}


def paired_bootstrap_f1(y_true, pred_a, pred_b, n_classes, n_boot=1000, seed=42) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)  # même rééchantillonnage pour A et B
        diffs[i] = (macro_f1(y_true[idx], pred_a[idx], n_classes)
                    - macro_f1(y_true[idx], pred_b[idx], n_classes))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"delta_macro_f1": float(macro_f1(y_true, pred_a, n_classes)
                                    - macro_f1(y_true, pred_b, n_classes)),
            "ci95": [float(lo), float(hi)], "significant": bool(lo > 0 or hi < 0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--n_classes", type=int, default=100)
    args = ap.parse_args()
    A, B = pd.read_csv(args.a), pd.read_csv(args.b)
    assert (A.y_true.values == B.y_true.values).all(), "runs must share the same test set"
    y, pa, pb = A.y_true.values, A.y_pred.values, B.y_pred.values
    print("McNemar   :", mcnemar(y, pa, pb))
    print("Bootstrap :", paired_bootstrap_f1(y, pa, pb, args.n_classes))


if __name__ == "__main__":
    main()
