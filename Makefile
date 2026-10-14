.PHONY: install lint test eda baseline retrieval lstm finetune lora compare audit

install:
	pip install -r requirements.txt

lint:
	ruff check src tests

test:
	pytest -q

eda:        ; python -m src.data --config configs/base.yaml
baseline:   ; python -m src.baseline --config configs/base.yaml
retrieval:  ; python -m src.retrieval --config configs/retrieval.yaml
lstm:       ; python -m src.lstm --config configs/lstm.yaml
finetune:   ; python -m src.finetune --config configs/finetune.yaml
lora:       ; python -m src.finetune --config configs/lora.yaml
compare:    ; python -m src.stats_tests --a results/preds_finetune.csv --b results/preds_lora.csv
audit:      ; python -m src.audit --config configs/finetune.yaml
