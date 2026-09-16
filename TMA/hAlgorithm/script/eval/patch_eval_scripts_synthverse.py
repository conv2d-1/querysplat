#!/usr/bin/env python3
"""Add --coord-convention and synthverse shell support to baseline DP eval scripts."""

from __future__ import annotations

import re
from pathlib import Path

WORKSPACE = Path("/mnt/home/tcchen/workspace")

EVAL_SCRIPTS = [
    WORKSPACE / "Projects/4RC/scripts/eval_worldtrack_4rc.py",
    WORKSPACE / "Projects/Any4D/scripts/eval_worldtrack_any4d.py",
    WORKSPACE / "Projects/vdpm/scripts/eval_worldtrack_vdpm.py",
    WORKSPACE / "Projects/Open-d4rt/eval_worldtrack_opend4rt.py",
    WORKSPACE / "Projects/SpaTrackerV2/evaluation/eval_worldtrack_spatrackerv2.py",
    WORKSPACE / "Projects/St4RTrack/evaluation/eval_worldtrack_st4rtrack.py",
    WORKSPACE / "TraceAnything/scripts/eval_worldtrack_traceanything.py",
]

SHELL_SCRIPTS = [
    WORKSPACE / "Projects/4RC/scripts/run_eval_worldtrack_4rc.sh",
    WORKSPACE / "Projects/Any4D/scripts/run_eval_worldtrack_any4d.sh",
    WORKSPACE / "Projects/vdpm/scripts/run_eval_worldtrack_vdpm.sh",
    WORKSPACE / "Projects/Open-d4rt/run_eval_worldtrack_opend4rt.sh",
    WORKSPACE / "Projects/SpaTrackerV2/scripts/run_eval_worldtrack_spatrackerv2.sh",
    WORKSPACE / "Projects/St4RTrack/scripts/run_eval_worldtrack_st4rtrack.sh",
    WORKSPACE / "TraceAnything/scripts/run_eval_worldtrack_traceanything.sh",
]

SYNTHVERSE_SHELL_BLOCK = '''
SYNTHVERSE_JSON="${SYNTHVERSE_JSON:-/mnt/nasTeam2/AI/datasets/TMD/SynthVerse/converted/train_mf_with_tracking_subset50_64f_q430.json}"
SYNTHVERSE_DYNAMIC_JSON="${SYNTHVERSE_DYNAMIC_JSON:-/mnt/nasTeam2/AI/datasets/TMD/SynthVerse/converted/train_mf_with_tracking_subset50_64f_dynamic.json}"
DYNAMIC_QUERY_THRESHOLD="${DYNAMIC_QUERY_THRESHOLD:-}"
COORD_CONVENTION="${COORD_CONVENTION:-opend4rt}"
'''

SYNTHVERSE_CASE = '''  synthverse|synthverse_dynamic)
    DATA_ROOT="${DATA_ROOT:-/mnt/nasTeam2/AI/datasets/TMD}"
    COORD_CONVENTION="${COORD_CONVENTION:-pointodyssey}"
    DYNAMIC_QUERY_THRESHOLD="${DYNAMIC_QUERY_THRESHOLD:-0.01}"
    JSON_ARGS+=(--json "$SYNTHVERSE_DYNAMIC_JSON" --subset-name synthverse_subset50_dynamic)
    ;;
'''


def patch_eval_py(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "--coord-convention" in text:
        print(f"skip eval: {path}")
        return
    text = text.replace(
        '    p.add_argument("--resume", action="store_true")\n    return p.parse_args()',
        '    p.add_argument("--resume", action="store_true")\n'
        '    p.add_argument("--coord-convention", default="opend4rt")\n'
        '    return p.parse_args()',
        1,
    )
    text = text.replace(
        "    resume: bool,\n    output_dir: Path,",
        "    resume: bool,\n    coord_convention: str,\n    output_dir: Path,",
        1,
    )
    text = re.sub(
        r"filter_sequences_with_frame0_queries\(\s*json_path, seq_names, data_root=data_root, num_frames=num_frames\s*\)",
        "filter_sequences_with_frame0_queries(\n        json_path, seq_names, data_root=data_root, num_frames=num_frames, coord_convention=coord_convention\n    )",
        text,
        count=1,
    )
    text = re.sub(
        r"load_sequence_from_json\(json_path, name, data_root=data_root, num_frames=num_frames, load_rgb=True\)",
        "load_sequence_from_json(json_path, name, data_root=data_root, num_frames=num_frames, load_rgb=True, coord_convention=coord_convention)",
        text,
        count=1,
    )
    text = re.sub(
        r"load_sequence_from_json\(\s*json_path,\s*name,\s*data_root=data_root,\s*num_frames=num_frames,\s*load_rgb=True,\s*\)",
        "load_sequence_from_json(json_path, name, data_root=data_root, num_frames=num_frames, load_rgb=True, coord_convention=coord_convention)",
        text,
        count=1,
    )
    # pass coord from main evaluate_subset calls
    text = re.sub(
        r"(evaluate_subset\(\s*[^)]*resume=args\.resume,\s*)",
        r"\1coord_convention=args.coord_convention,\n            ",
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")
    print(f"patched eval: {path}")


def patch_shell(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "synthverse)" in text:
        print(f"skip shell: {path}")
        return
    if 'SUBSET="${SUBSET:-adt}"' in text and "SYNTHVERSE_JSON" not in text:
        text = text.replace('SUBSET="${SUBSET:-adt}"\n', 'SUBSET="${SUBSET:-adt}"\n' + SYNTHVERSE_SHELL_BLOCK)
    text = text.replace(
        "  ds) JSON_ARGS+=(--json \"$DS_JSON\" --subset-name ds_mini) ;;\n  *)",
        "  ds) JSON_ARGS+=(--json \"$DS_JSON\" --subset-name ds_mini) ;;\n" + SYNTHVERSE_CASE + "  *)",
        1,
    )
    text = text.replace(
        '    echo "Unknown SUBSET=$SUBSET"',
        '    echo "Unknown SUBSET=$SUBSET (use adt, pstudio, po, ds, synthverse)"',
        1,
    )
    if "--coord-convention" not in text:
        text = text.replace(
            "  --output-dir \"$OUTPUT_DIR\" \\\n",
            "  --output-dir \"$OUTPUT_DIR\" \\\n  --coord-convention \"$COORD_CONVENTION\" \\\n",
            1,
        )
    path.write_text(text, encoding="utf-8")
    print(f"patched shell: {path}")


def main() -> None:
    for p in EVAL_SCRIPTS:
        if p.is_file():
            patch_eval_py(p)
    for p in SHELL_SCRIPTS:
        if p.is_file():
            patch_shell(p)


if __name__ == "__main__":
    main()
