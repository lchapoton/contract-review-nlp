# Contract Review NLP

NLP core of a contract-review assistant: **classify every clause** into its type, **retrieve similar clauses** from a clause bank, and **document where the system can (and cannot) be trusted**.

Final project · *Deep Learning for Sequential Data and NLP* (MSA-DAT07-01), Albert School × Mines Paris - PSL.

## Overview

| Component | Approach | Code |
|---|---|---|
| Data | LEDGAR (LexGLUE): ~80k SEC contract clauses, 100 clause types, highly imbalanced | `src/data.py` |
| Baseline | TF-IDF (1-2 grams) + logistic regression | `src/baseline.py` |
| Sequential model | BiLSTM with packed sequences | `src/lstm.py` |
| Main classifier | Fine-tuned RoBERTa-base (full and LoRA) | `src/finetune.py` |
| Clause retrieval | Sentence-transformer index, cosine top-k, hybrid filters | `src/retrieval.py` |
| Statistical comparison | McNemar + paired bootstrap on macro-F1 | `src/stats_tests.py` |
| Audit | Subgroups, counterfactuals, calibration / abstention | `src/audit.py` |

**Main metric: macro-F1.** With 100 imbalanced classes, accuracy is dominated by frequent clause types.

## Pre-registered hypothesis

> Written **before** running the experiments (milestone m4).

1. A fine-tuned RoBERTa-base beats the TF-IDF baseline on test macro-F1. The gain is tested with a paired bootstrap (95% CI excludes 0).
2. LoRA (r=8, W_Q and W_V) reaches a macro-F1 within 2 points of full fine-tuning, while training less than 2% of the parameters. The difference is tested with McNemar.

## Results

> Filled in from `results/*.json` once the runs are done.

| Model | Trainable params | Train time | Accuracy | Macro-F1 |
|---|---|---|---|---|
| TF-IDF + LogReg | – | | | |
| BiLSTM | | | | |
| RoBERTa full fine-tuning | | | | |
| RoBERTa LoRA (r=8) | | | | |

| Retrieval (MiniLM-L6) | P@1 | P@5 | P@10 | MRR@10 |
|---|---|---|---|---|
| | | | | |

## Reproduce

```bash
git clone https://github.com/<user>/contract-review-nlp.git
cd contract-review-nlp
python -m venv .venv && source .venv/bin/activate
make install

make eda        # m1 · class distribution + stats
make baseline   # m1 · TF-IDF baseline
make retrieval  # m2 · retrieval index + P@k / MRR
make lstm       # m3 · BiLSTM + per-class error analysis
make finetune   # m4 · full fine-tuning
make lora       # m4 · LoRA
make compare    # m4 · McNemar + paired bootstrap
make audit      # bias / robustness audit
```

Every script reads its hyperparameters from `configs/*.yaml`, fixes the seed (42) and writes its outputs to `results/`. Checkpoints go to `outputs/` and are not versioned.

**On Google Colab (GPU):**
```python
!git clone https://github.com/<user>/contract-review-nlp.git
%cd contract-review-nlp
!pip install -q -r requirements.txt
!python -m src.finetune --config configs/finetune.yaml
```

To iterate quickly, set `train_subsample: 10000` in the config.

## Evaluation protocol

- The splits are the official LexGLUE train / validation / test splits.
- Every choice (hyperparameters, best epoch, LoRA rank) is made on **validation**.
- The **test** set is used once per final model.
- Model comparisons are **paired**, on the same test set.

## Repository structure

```
├── configs/          # one YAML per experiment
├── src/              # all reusable code (python -m src.<module>)
├── tests/            # unit tests: metrics vs scikit-learn, McNemar
├── notebooks/        # exploration and figures only
├── results/          # metrics (JSON), predictions (CSV), figures: versioned
├── outputs/          # checkpoints and embeddings: NOT versioned
└── reports/          # model card
```

## Limitations and intended use

See [`reports/model_card.md`](reports/model_card.md). In short:

- This is a triage aid that requires human review, not a legal decision tool.
- It covers English, US-style (SEC) contracts only.

## License

MIT
