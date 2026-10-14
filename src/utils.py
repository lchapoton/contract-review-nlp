"""Utilitaires partagés : config, seed, sauvegarde des résultats."""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml


def load_config(default: str) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for key in ("results_dir", "outputs_dir"):
        Path(cfg.get(key, key.split("_")[0])).mkdir(parents=True, exist_ok=True)
    return cfg


def set_seed(seed: int) -> None:
    """Fixe tous les générateurs aléatoires pour des résultats reproductibles."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"-> {path}")
