#!/usr/bin/env python3
"""Extract Open-d4rt encoder weights from a full ``opend4rt.ckpt`` checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from hAlgorithm.modules.models2.mv_encoder.opend4rt_weight_loader import extract_encoder_state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(
            "/mnt/home/tcchen/workspace/Projects/Open-d4rt/checkpoints/"
            "OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt.ckpt"
        ),
        help="Full Open-d4rt checkpoint path.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(
            "/mnt/nasTeam/AI/Autoresearch/Algorithm/Pretrain/OpenD4RT/"
            "OpenD4RT_32CLIP_9Dataset_NoAUG/opend4rt_encoder.pth"
        ),
        help="Output encoder-only checkpoint path.",
    )
    args = parser.parse_args()

    if not args.src.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {args.src}")

    payload = torch.load(args.src, map_location="cpu", weights_only=False)
    state = extract_encoder_state_dict(payload)
    if not state:
        raise RuntimeError(f"No encoder tensors found in {args.src}")

    # ``encoder`` key: ``OpenD4RTVideoEncoder.pretrain`` / weight loader.
    # ``fuse_encoder.encoder.*`` keys: ``trainer.load_from`` on MVQuery6 (strict=False).
    out_payload = {"encoder": state}
    out_payload.update({f"fuse_encoder.encoder.{k}": v for k, v in state.items()})

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_payload, args.dst)
    num_elems = sum(v.numel() for v in state.values())
    print(f"Saved {len(state)} encoder tensors ({num_elems:,} elems) to {args.dst}")


if __name__ == "__main__":
    main()
