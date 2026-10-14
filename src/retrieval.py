"""Index de recherche sémantique sur la banque de clauses (jalon m2).

- Banque de clauses = train set LEDGAR, encodé une fois et normalisé.
- Recherche = cosinus (produit scalaire de vecteurs unitaires), top-k.
- Évaluation = Precision@k et MRR, une clause du test servant de requête et
  un voisin étant "pertinent" s'il a le même type de clause.

Usage : python -m src.retrieval --config configs/retrieval.yaml
"""
import re

import numpy as np
from sentence_transformers import SentenceTransformer

from src.data import load_ledgar
from src.utils import load_config, save_json, set_seed


class ClauseIndex:
    def __init__(self, encoder_name: str, batch_size: int = 128):
        self.model = SentenceTransformer(encoder_name)
        self.batch_size = batch_size
        self.emb = None
        self.texts, self.labels = [], []

    def build(self, texts, labels=None):
        self.texts, self.labels = list(texts), list(labels) if labels is not None else []
        self.emb = self.encode(self.texts)
        return self

    def encode(self, texts) -> np.ndarray:
        return self.model.encode(list(texts), batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=True)

    def search(self, query: str, k: int = 5, label_filter=None, regex: str | None = None):
        """Recherche hybride : sémantique + filtre de classe + filtre regex (ex. délai)."""
        scores = self.emb @ self.encode([query])[0]
        order = np.argsort(-scores)
        hits = []
        for i in order:
            if label_filter is not None and self.labels[i] != label_filter:
                continue
            if regex and not re.search(regex, self.texts[i], flags=re.I):
                continue
            hits.append({"rank": len(hits) + 1, "score": float(scores[i]),
                         "label": self.labels[i] if self.labels else None,
                         "text": self.texts[i]})
            if len(hits) == k:
                break
        return hits


def evaluate(index: ClauseIndex, q_texts, q_labels, k_values=(1, 5, 10)) -> dict:
    Q = index.encode(q_texts)
    bank_labels = np.asarray(index.labels)
    kmax = max(k_values)
    out = {f"precision@{k}": [] for k in k_values}
    rr = []
    for start in range(0, len(Q), 512):  # par blocs pour limiter la mémoire
        S = Q[start:start + 512] @ index.emb.T
        top = np.argsort(-S, axis=1)[:, :kmax]
        rel = bank_labels[top] == np.asarray(q_labels[start:start + 512])[:, None]
        for k in k_values:
            out[f"precision@{k}"].extend(rel[:, :k].mean(1).tolist())
        first = np.where(rel.any(1), rel.argmax(1) + 1, np.inf)
        rr.extend((1.0 / first).tolist())
    res = {k: float(np.mean(v)) for k, v in out.items()}
    res[f"mrr@{kmax}"] = float(np.mean(rr))
    return res


def main():
    cfg = load_config("configs/retrieval.yaml")
    set_seed(cfg["seed"])
    ds, names = load_ledgar(cfg)
    index = ClauseIndex(cfg["encoder"], cfg["batch_size"]).build(
        ds["train"]["text"], ds["train"]["label"])
    np.save(f"{cfg['outputs_dir']}/bank_embeddings.npy", index.emb)

    test = ds["test"].shuffle(seed=cfg["seed"]).select(range(cfg["eval_queries"]))
    metrics = evaluate(index, test["text"], test["label"], cfg["k_values"])
    save_json({"encoder": cfg["encoder"], **metrics},
              f"{cfg['results_dir']}/metrics_retrieval.json")

    # Exemples qualitatifs pour les slides
    termination = names.index("Terminations") if "Terminations" in names else None
    for q, lab, rx in [
        ("What happens if there is an earthquake?", None, None),
        ("show me termination clauses with a 30-day notice period", termination,
         r"\b(30|thirty)\s*(\(30\)\s*)?days?\b"),
    ]:
        print(f"\nQ: {q}")
        for h in index.search(q, k=3, label_filter=lab, regex=rx):
            print(f"  [{h['score']:.2f}] {names[h['label']]}: {h['text'][:150]}...")


if __name__ == "__main__":
    main()
