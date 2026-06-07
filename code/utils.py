from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str = "auto") -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_trainable_checkpoint(model: torch.nn.Module, path: str | Path) -> None:
    trainable_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith("qformer.") or key.startswith("language_projection.")
    }
    torch.save(trainable_state, path)


def load_trainable_checkpoint(model: torch.nn.Module, path: str | Path, device: torch.device) -> None:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=False)
