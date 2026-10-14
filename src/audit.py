"""Audit de biais et de robustesse du classifieur fine-tuné (S07).

1. Performance par sous-groupe : fréquence de la classe, longueur de la clause.
2. Tests contrefactuels : on change un nom de partie / une juridiction / un genre
   sans changer le type de clause -> taux de prédictions qui changent ("flip rate").
3. Calibration : confiance des bonnes vs mauvaises prédictions, seuil d'abstention.

Usage : python -m src.audit --config configs/finetune.yaml
"""
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data import load_ledgar
from src.metrics import macro_f1
from src.utils import load_config, save_json, set_seed

# (nom, motif regex, remplacement) : la sémantique juridique du type de clause est inchangée
PERTURBATIONS = [
    ("jurisdiction_NY_to_France", r"State of New York|New York", "France"),
    ("jurisdiction_Delaware_to_Germany", r"State of Delaware|Delaware", "Germany"),
    ("party_Company_to_AcmeSAS", r"\bthe Company\b", "Acme SAS"),
    ("gender_he_to_she", r"\bhe\b", "she"),
    ("gender_his_to_her", r"\bhis\b", "her"),
    ("currency_USD_to_EUR", r"\$\s?(\d)", r"EUR \1"),
]


@torch.no_grad()
def predict(model, tok, texts, max_length, bs=64):
    device = next(model.parameters()).device
    probs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt").to(device)
        probs.append(torch.softmax(model(**enc).logits, -1).cpu())
    return torch.cat(probs).numpy()


def main():
    cfg = load_config("configs/finetune.yaml")
    set_seed(cfg["seed"])
    rd = cfg["results_dir"]
    ds, names = load_ledgar(cfg)
    n = len(names)
    path = f"{cfg['outputs_dir']}/{cfg['run_name']}/best"
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    test = ds["test"]
    texts, y = list(test["text"]), np.array(test["label"])
    probs = predict(model, tok, texts, cfg["max_length"])
    pred, conf = probs.argmax(1), probs.max(1)
    out = {}

    # 1. Sous-groupes
    freq = np.bincount(ds["train"]["label"], minlength=n)[y]
    q1, q2 = np.percentile(np.bincount(ds["train"]["label"], minlength=n), [33, 66])
    n_tok = np.array([len(tok(t)["input_ids"]) for t in texts])
    groups = {
        "rare_classes": freq <= q1, "mid_classes": (freq > q1) & (freq <= q2),
        "frequent_classes": freq > q2,
        "short(<64tok)": n_tok < 64, "medium": (n_tok >= 64) & (n_tok <= cfg["max_length"]),
        "truncated(>max_len)": n_tok > cfg["max_length"],
    }
    out["subgroups"] = {g: {"n": int(m.sum()),
                            "accuracy": float((pred[m] == y[m]).mean()) if m.any() else None,
                            "macro_f1": macro_f1(y[m], pred[m], n) if m.any() else None}
                        for g, m in groups.items()}

    # 2. Contrefactuels
    cf = {}
    for name, pattern, repl in PERTURBATIONS:
        idx = [i for i, t in enumerate(texts) if re.search(pattern, t)]
        if not idx:
            continue
        new = [re.sub(pattern, repl, texts[i]) for i in idx]
        new_pred = predict(model, tok, new, cfg["max_length"]).argmax(1)
        cf[name] = {"n_affected": len(idx),
                    "flip_rate": float((new_pred != pred[idx]).mean()),
                    "acc_before": float((pred[idx] == y[idx]).mean()),
                    "acc_after": float((new_pred == y[idx]).mean())}
    out["counterfactuals"] = cf

    # 3. Calibration et abstention
    correct = pred == y
    out["calibration"] = {"mean_conf_correct": float(conf[correct].mean()),
                          "mean_conf_wrong": float(conf[~correct].mean())}
    out["abstention"] = [{"threshold": t, "coverage": float((conf >= t).mean()),
                          "accuracy_kept": float(correct[conf >= t].mean())}
                         for t in (0.5, 0.7, 0.9)]

    save_json(out, f"{rd}/audit.json")
    pd.DataFrame(out["subgroups"]).T.to_csv(f"{rd}/audit_subgroups.csv")


if __name__ == "__main__":
    main()
