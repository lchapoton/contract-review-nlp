"""Classifieur (Bi)LSTM sur LEDGAR (jalon m3), d'après le TP04.

Usage : python -m src.lstm --config configs/lstm.yaml
"""
import re
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from src.data import load_ledgar
from src.metrics import confusion_matrix, most_confused, per_class_prf, report
from src.utils import load_config, save_json, set_seed

PAD, UNK = 0, 1


def tokenize(text: str):
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocab(texts, min_freq: int):
    counts = Counter(w for t in texts for w in tokenize(t))  # train uniquement
    itos = ["<pad>", "<unk>"] + [w for w, c in counts.most_common() if c >= min_freq]
    return {w: i for i, w in enumerate(itos)}


def encode(text, stoi, max_len):
    ids = [stoi.get(w, UNK) for w in tokenize(text)][:max_len]
    return ids or [UNK]


def make_collate(stoi, max_len):
    def collate(batch):
        ids = [torch.tensor(encode(b["text"], stoi, max_len)) for b in batch]
        lengths = torch.tensor([len(x) for x in ids])
        padded = nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=PAD)
        return padded, lengths, torch.tensor([b["label"] for b in batch])
    return collate


class LSTMClassifier(nn.Module):
    def __init__(self, V, d, H, C, bidirectional=True):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=PAD)
        self.lstm = nn.LSTM(d, H, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(H * (2 if bidirectional else 1), C)

    def forward(self, ids, lengths):
        packed = pack_padded_sequence(self.emb(ids), lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)  # les pads ne sont jamais lus
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[-2], h_n[-1]], dim=1) if self.lstm.bidirectional else h_n[-1]
        return self.fc(h)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for ids, lengths, _ in loader:
        preds.append(model(ids.to(device), lengths).argmax(1).cpu())
    return torch.cat(preds).numpy()


def main():
    cfg = load_config("configs/lstm.yaml")
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds, names = load_ledgar(cfg)
    stoi = build_vocab(ds["train"]["text"], cfg["min_freq"])
    collate = make_collate(stoi, cfg["max_len"])
    mk = lambda split, shuffle: DataLoader(ds[split], batch_size=cfg["batch_size"],  # noqa: E731
                                           shuffle=shuffle, collate_fn=collate)
    train_dl, val_dl, test_dl = mk("train", True), mk("validation", False), mk("test", False)

    model = LSTMClassifier(len(stoi), cfg["emb_dim"], cfg["hidden"], len(names),
                           cfg["bidirectional"]).to(device)
    print(f"vocab={len(stoi)}  params={sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.CrossEntropyLoss()

    best_f1, ckpt = -1.0, f"{cfg['outputs_dir']}/lstm_best.pt"
    for epoch in range(cfg["epochs"]):
        model.train()
        for ids, lengths, y in train_dl:
            loss = loss_fn(model(ids.to(device), lengths), y.to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_norm"])
            opt.step()
        m = report(ds["validation"]["label"], predict(model, val_dl, device), len(names))
        print(f"epoch {epoch + 1}: loss={loss.item():.3f} val={m}")
        if m["macro_f1"] > best_f1:  # early stopping sur la validation
            best_f1 = m["macro_f1"]
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt))
    y_true, y_pred = np.array(ds["test"]["label"]), predict(model, test_dl, device)
    C = confusion_matrix(y_true, y_pred, len(names))
    p, r, f1 = per_class_prf(C)
    pd.DataFrame({"class": names, "precision": p, "recall": r, "f1": f1,
                  "support": C.sum(1)}).sort_values("f1").to_csv(
        f"{cfg['results_dir']}/per_class_lstm.csv", index=False)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        f"{cfg['results_dir']}/preds_lstm.csv", index=False)
    save_json({"test": report(y_true, y_pred, len(names)),
               "most_confused": most_confused(C, names)},
              f"{cfg['results_dir']}/metrics_lstm.json")


if __name__ == "__main__":
    main()
