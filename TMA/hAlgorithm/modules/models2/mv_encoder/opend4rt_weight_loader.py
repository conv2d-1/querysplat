"""Load Open-d4rt encoder weights into TMA VideoMAE backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "module", "encoder"):
            value = payload.get(key)
            if isinstance(value, dict) and value:
                if all(torch.is_tensor(v) for v in value.values()):
                    return value
                nested = _unwrap_state_dict(value)
                if nested:
                    return nested
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    return {}


def extract_encoder_state_dict(payload: Any, *, prefix: str = "encoder.") -> dict[str, torch.Tensor]:
    """Extract encoder tensors from an Open-d4rt checkpoint payload."""
    state = _unwrap_state_dict(payload)
    if not state:
        return {}

    if any(k.startswith(prefix) for k in state):
        return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

    if prefix.rstrip(".") in state and isinstance(state[prefix.rstrip(".")], dict):
        inner = state[prefix.rstrip(".")]
        return {k: v for k, v in inner.items() if torch.is_tensor(v)}

    return {k: v for k, v in state.items() if torch.is_tensor(v)}


def load_opend4rt_encoder_checkpoint(
    module: torch.nn.Module,
    ckpt_path: str | Path,
    *,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
) -> tuple[list[str], list[str]]:
    """Load encoder weights from ``opend4rt.ckpt`` or a pre-extracted encoder-only file."""
    path = Path(ckpt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Open-d4rt encoder checkpoint not found: {path}")

    payload = torch.load(path, map_location=map_location, weights_only=False)
    state = extract_encoder_state_dict(payload)
    if not state:
        raise RuntimeError(f"No encoder tensors found in checkpoint: {path}")

    missing, unexpected = module.load_state_dict(state, strict=strict)
    return missing, unexpected
