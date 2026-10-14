"""Chargement de LEDGAR + exploration (jalon m1).

Usage : python -m src.data --config configs/base.yaml
"""
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

from src.utils import load_config, save_json, set_seed


def load_ledgar(cfg: dict):
    """Renvoie (dataset dict train/validation/test, liste des noms de classes)."""
    ds = load_dataset(cfg["dataset"], cfg["subset"])
    n = cfg.get("train_subsample")
    if n:
        ds["train"] = ds["train"].shuffle(seed=cfg["seed"]).select(range(n))
    label_names = ds["train"].features["label"].names
    return ds, label_names


def main():
    cfg = load_config("configs/base.yaml")
    set_seed(cfg["seed"])
    ds, names = load_ledgar(cfg)
    rd = cfg["results_dir"]

    counts = np.bincount(ds["train"]["label"], minlength=len(names))
    order = np.argsort(counts)[::-1]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.bar(range(len(names)), counts[order], color="#1a7f72")
    ax.set_yscale("log")
    ax.set_xlabel("classe (triée par fréquence)")
    ax.set_ylabel("nb de clauses (log)")
    ax.set_title("LEDGAR train : distribution des 100 types de clauses")
    fig.tight_layout()
    fig.savefig(f"{rd}/class_distribution.png", dpi=150)

    lengths = np.array([len(t.split()) for t in ds["train"]["text"]])
    stats = {
        "sizes": {s: len(ds[s]) for s in ds},
        "n_classes": len(names),
        "most_frequent": {names[i]: int(counts[i]) for i in order[:5]},
        "least_frequent": {names[i]: int(counts[i]) for i in order[-5:]},
        "imbalance_ratio": float(counts.max() / max(counts.min(), 1)),
        "words_per_clause": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "p95": float(np.percentile(lengths, 95)),
            "max": int(lengths.max()),
        },
    }
    save_json(stats, f"{rd}/eda_stats.json")
    print(stats)


if __name__ == "__main__":
    main()
