"""Fine-tuning complet ou LoRA d'un encodeur BERT-class (jalon m4).

Usage :
  python -m src.finetune --config configs/finetune.yaml   # fine-tuning complet
  python -m src.finetune --config configs/lora.yaml       # LoRA (peft)
"""
import time

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from src.data import load_ledgar
from src.metrics import report
from src.utils import load_config, save_json, set_seed


def add_lora(model, lora_cfg: dict):
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(task_type=TaskType.SEQ_CLS, r=lora_cfg["r"],
                        lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
                        target_modules=lora_cfg["target_modules"])
    # TaskType.SEQ_CLS garde la tête de classification entièrement entraînable
    return get_peft_model(model, config)


def main():
    cfg = load_config("configs/finetune.yaml")
    set_seed(cfg["seed"])
    ds, names = load_ledgar(cfg)
    n = len(names)
    run, rd = cfg["run_name"], cfg["results_dir"]

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])  # toujours le tokenizer du modèle
    ds = ds.map(lambda b: tok(b["text"], truncation=True, max_length=cfg["max_length"]),
                batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=n,
        id2label=dict(enumerate(names)), label2id={name: i for i, name in enumerate(names)})
    if cfg.get("lora"):
        model = add_lora(model, cfg["lora"])
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f} %)")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return report(labels, logits.argmax(-1), n)

    args = TrainingArguments(
        output_dir=f"{cfg['outputs_dir']}/{run}",
        learning_rate=cfg["lr"], num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        warmup_ratio=cfg["warmup_ratio"], weight_decay=cfg["weight_decay"],
        lr_scheduler_type="linear", max_grad_norm=1.0,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="macro_f1",
        fp16=cfg["fp16"] and torch.cuda.is_available(),
        seed=cfg["seed"], report_to="none", logging_steps=100,
    )
    trainer = Trainer(model=model, args=args, train_dataset=ds["train"],
                      eval_dataset=ds["validation"], processing_class=tok,
                      data_collator=DataCollatorWithPadding(tok),
                      compute_metrics=compute_metrics)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.train()
    train_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None

    # Test : utilisé UNE fois, avec le meilleur checkpoint choisi sur la validation
    out = trainer.predict(ds["test"])
    probs = torch.softmax(torch.tensor(out.predictions), -1).numpy()
    y_pred = probs.argmax(1)
    y_true = np.array(ds["test"]["label"])
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "confidence": probs.max(1)}).to_csv(
        f"{rd}/preds_{run}.csv", index=False)
    save_json({
        "model": cfg["model_name"], "lora": cfg.get("lora"),
        "trainable_params": trainable, "total_params": total,
        "train_time_min": round(train_time / 60, 1), "peak_gpu_mem_gb": peak_mem,
        "validation": trainer.evaluate(), "test": report(y_true, y_pred, n),
    }, f"{rd}/metrics_{run}.json")
    trainer.save_model(f"{cfg['outputs_dir']}/{run}/best")


if __name__ == "__main__":
    main()
