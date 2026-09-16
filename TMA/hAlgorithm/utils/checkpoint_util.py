"""Checkpoint loading helpers for architecture changes (e.g. new output heads)."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


def _can_expand_leading_dim(ckpt: torch.Tensor, target: torch.Tensor) -> bool:
    """Allow growing the first dim (Linear out_features / 1-D bias)."""
    if ckpt.ndim != target.ndim or ckpt.ndim not in (1, 2):
        return False
    if ckpt.shape[0] >= target.shape[0]:
        return False
    if ckpt.ndim == 2:
        return ckpt.shape[1] == target.shape[1]
    return True


def _expand_leading_dim(ckpt: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    expanded = target.clone()
    expanded[: ckpt.shape[0]] = ckpt
    return expanded


def adapt_state_dict_for_model(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    expand_output_dims: bool = True,
) -> Tuple[Dict[str, torch.Tensor], List[str], List[str], List[Tuple[str, torch.Size, torch.Size]]]:
    """Filter / expand *state_dict* so it can be loaded into *model*.

    Returns:
        adapted_state_dict, expanded_keys, skipped_keys, shape_mismatch_keys
    """
    model_sd = model.state_dict()
    adapted: Dict[str, torch.Tensor] = {}
    expanded_keys: List[str] = []
    skipped_keys: List[str] = []
    shape_mismatch_keys: List[Tuple[str, torch.Size, torch.Size]] = []

    for key, value in state_dict.items():
        if key not in model_sd:
            continue

        target = model_sd[key]
        if value.shape == target.shape:
            adapted[key] = value
        elif expand_output_dims and _can_expand_leading_dim(value, target):
            adapted[key] = _expand_leading_dim(value, target)
            expanded_keys.append(key)
        else:
            shape_mismatch_keys.append((key, value.shape, target.shape))
            skipped_keys.append(key)

    return adapted, expanded_keys, skipped_keys, shape_mismatch_keys


def load_state_dict_compatible(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    strict: bool = False,
    expand_output_dims: bool = True,
    log_prefix: str = "",
):
    """Load *state_dict* into *model*, skipping incompatible keys.

    When *expand_output_dims* is True, tensors whose first dimension grew
  (typical case: appending a new head output) copy the overlapping prefix and
    keep the model's random init for the new rows.
    """
    adapted, expanded_keys, skipped_keys, shape_mismatch_keys = adapt_state_dict_for_model(
        model,
        state_dict,
        expand_output_dims=expand_output_dims,
    )
    result = model.load_state_dict(adapted, strict=strict)

    prefix = f"{log_prefix}: " if log_prefix else ""
    logging.info("%sLoaded %d / %d checkpoint keys", prefix, len(adapted), len(state_dict))
    if expanded_keys:
        logging.info("%sexpanded_keys (%d): %s", prefix, len(expanded_keys), expanded_keys)
    if skipped_keys:
        logging.info("%sskipped_keys (%d): %s", prefix, len(skipped_keys), skipped_keys)
    if shape_mismatch_keys:
        for key, ckpt_shape, model_shape in shape_mismatch_keys[:20]:
            logging.info(
                "%sskipped shape mismatch: %s ckpt=%s model=%s",
                prefix,
                key,
                tuple(ckpt_shape),
                tuple(model_shape),
            )
        if len(shape_mismatch_keys) > 20:
            logging.info("%s... and %d more shape mismatches", prefix, len(shape_mismatch_keys) - 20)
    logging.info("%sunexpected_keys: %s", prefix, result.unexpected_keys)
    logging.info("%smissing_keys: %s", prefix, result.missing_keys)
    return result
