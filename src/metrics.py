"""Métriques codées à la main (S01) : matrice de confusion, précision, rappel, F1."""
import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    C = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(C, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return C


def per_class_prf(C: np.ndarray):
    tp = np.diag(C).astype(float)
    pred = C.sum(axis=0)
    true = C.sum(axis=1)
    p = np.divide(tp, pred, out=np.zeros_like(tp), where=pred > 0)
    r = np.divide(tp, true, out=np.zeros_like(tp), where=true > 0)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return p, r, f1


def macro_f1(y_true, y_pred, n_classes: int) -> float:
    """Moyenne des F1 par classe, sur les classes présentes dans y_true ou y_pred
    (même convention que sklearn f1_score(average='macro'))."""
    C = confusion_matrix(y_true, y_pred, n_classes)
    _, _, f1 = per_class_prf(C)
    present = (C.sum(0) + C.sum(1)) > 0
    return float(f1[present].mean())


def accuracy(y_true, y_pred) -> float:
    return float((np.asarray(y_true) == np.asarray(y_pred)).mean())


def report(y_true, y_pred, n_classes: int) -> dict:
    return {"accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred, n_classes)}


def most_confused(C: np.ndarray, names, top: int = 10):
    """Paires (vraie, prédite) les plus fréquentes hors diagonale."""
    off = C.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(off.ravel())[::-1][:top], off.shape))[0]
    return [(names[i], names[j], int(off[i, j])) for i, j in idx]
