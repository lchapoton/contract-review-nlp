"""Les métriques maison doivent égaler scikit-learn (TP01, partie B)."""
import numpy as np
from sklearn.metrics import f1_score

from src.metrics import accuracy, confusion_matrix, macro_f1
from src.stats_tests import mcnemar


def test_exercise3_session1():
    # Matrice de l'exercice 3 de la séance 1 (termination / payment / confidentiality)
    C = np.array([[44, 4, 2], [2, 6, 2], [1, 1, 8]])
    y_true = np.repeat(np.arange(3), C.sum(1))
    y_pred = np.concatenate([np.repeat(np.arange(3), row) for row in C])
    assert (confusion_matrix(y_true, y_pred, 3) == C).all()
    assert abs(accuracy(y_true, y_pred) - 58 / 70) < 1e-9
    assert abs(macro_f1(y_true, y_pred, 3) - f1_score(y_true, y_pred, average="macro")) < 1e-9


def test_macro_f1_random_matches_sklearn():
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 20, 500), rng.integers(0, 20, 500)
    assert abs(macro_f1(y, p, 20) - f1_score(y, p, average="macro")) < 1e-9


def test_mcnemar_exercise5_session6():
    # b = 60, c = 40 -> chi2 = 39^2 / 100 = 3.61 < 3.84 : non significatif
    y = np.zeros(1000, dtype=int)
    a = np.ones(1000, dtype=int)
    b = np.ones(1000, dtype=int)
    a[:830], b[:830] = 0, 0          # 830 bons pour les deux
    a[830:890] = 0                   # 60 bons seulement pour A
    b[890:930] = 0                   # 40 bons seulement pour B
    res = mcnemar(y, a, b)
    assert res["b"] == 60 and res["c"] == 40
    assert abs(res["chi2"] - 3.61) < 1e-9 and res["p_value"] > 0.05
