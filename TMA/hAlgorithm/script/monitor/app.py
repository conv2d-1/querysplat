#!/usr/bin/env python3
"""Training Log Monitor - Real-time web dashboard for monitoring training experiments."""

import ast
import math
import os
import re
import json
import time
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request, send_file

app = Flask(__name__, static_folder="static")
app.json.sort_keys = False

RESULTS_DIR = os.environ.get(
    "RESULTS_DIR",
    os.path.join(os.getcwd(), "results"),
)


def parse_train_line(line):
    """Parse a single training log line into structured data."""
    match = re.search(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+).*?iter(\d+),\s*(.*)", line
    )
    if not match:
        return None
    timestamp_str, iteration, metrics_str = match.groups()
    iteration = int(iteration)
    metrics = {}
    for kv in metrics_str.split(", "):
        kv = kv.strip()
        if ":" not in kv:
            continue
        k, v = kv.split(":", 1)
        k = k.strip()
        v = v.strip()
        try:
            metrics[k] = float(v.rstrip("M"))
        except ValueError:
            metrics[k] = v
    clean = {}
    for k, v in metrics.items():
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            clean[k] = None
        else:
            clean[k] = v
    return {"iter": iteration, "timestamp": timestamp_str, **clean}


def parse_timing_line(line):
    """Parse a timing line like: iter:1.44, mean:1.44, total:0:35:33, eta:19:27:58
    Also handles: total:1 day, 0:35:33, eta:1 day, 21:24:22
    """
    match = re.search(
        r"iter:([\d.]+),\s*mean:([\d.]+),\s*total:(.+),\s*eta:(.+)$", line
    )
    if not match:
        return None
    tail = line[line.index("total:"):]
    parts = tail.split(", eta:")
    if len(parts) != 2:
        return None
    total_part = parts[0].replace("total:", "").strip()
    eta_part = parts[1].strip()
    return {
        "iter_time": float(match.group(1)),
        "mean_time": float(match.group(2)),
        "total": total_part,
        "eta": eta_part,
    }


def parse_log_file(log_path, last_n=None, since_iter=None):
    """Parse a logging.log file and return structured training data."""
    records = []
    timing = None
    if not os.path.exists(log_path):
        return records, timing

    with open(log_path, "r") as f:
        lines = f.readlines()

    if last_n:
        lines = lines[-last_n:]

    i = 0
    while i < len(lines):
        line = lines[i]
        if "moge_trainer.py" in line and "iter" in line and "loss:" in line:
            rec = parse_train_line(line)
            if rec and (since_iter is None or rec["iter"] > since_iter):
                records.append(rec)
        elif "moge_trainer.py" in line and "eta:" in line:
            t = parse_timing_line(line)
            if t:
                timing = t
        i += 1

    return records, timing


def _parse_timing_only(log_path):
    """Fast path: only read the tail of the log to extract timing info."""
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, "r") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 500000))
            lines = f.readlines()
        for line in reversed(lines):
            if "eta:" in line and "total:" in line:
                t = parse_timing_line(line)
                if t:
                    return t
    except Exception:
        pass
    return None


def parse_eval_file(eval_path):
    """Parse an evaluation result file."""
    if not os.path.exists(eval_path):
        return None

    result = {}
    with open(eval_path, "r") as f:
        content = f.read()

    iter_match = re.search(r"Iter:\s*(\d+)", content)
    if iter_match:
        result["iter"] = int(iter_match.group(1))

    dataset_match = re.search(r"on dataset (\w+)", content)
    if dataset_match:
        result["dataset"] = dataset_match.group(1)

    format_match = re.search(r"Format Text:\s*\n(.+)", content)
    format_vals = None
    if format_match:
        fmt = format_match.group(1).strip()
        result["format_text"] = fmt
        # Try to parse numeric vector (works well for hypersim)
        try:
            format_vals = [float(x) for x in re.split(r"\s+", fmt)]
        except ValueError:
            format_vals = None

    # Generic parser: detect separator → names → values pattern
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for separator lines (all dashes and spaces)
        if re.match(r"^[-\s]+$", line) and len(line) >= 3:
            # Next line should be metric names
            if i + 2 < len(lines):
                names_line = lines[i + 1].strip()
                vals_line = lines[i + 2].strip()
                # Skip if names_line is also a separator or empty
                if names_line and not re.match(r"^[-\s]+$", names_line):
                    names = names_line.split()
                    vals = vals_line.split()
                    if len(names) == len(vals) and len(names) > 0:
                        for name, val in zip(names, vals):
                            try:
                                v = float(val)
                                if not (math.isnan(v) or math.isinf(v)):
                                    result.setdefault(name, v)
                                else:
                                    result.setdefault(name, None)
                            except ValueError:
                                pass
                        i += 3
                        continue
        # Also handle single-metric blocks (name on one line, value on next)
        elif (
            line
            and not re.match(r"^[-\s]+$", line)
            and not line.startswith("Iter:")
            and not line.startswith("Evaluation")
            and not line.startswith("Format Text")
            and " " not in line
        ):
            if i + 1 < len(lines):
                val_line = lines[i + 1].strip()
                if val_line and " " not in val_line:
                    try:
                        v = float(val_line)
                        if not (math.isnan(v) or math.isinf(v)):
                            result.setdefault(line, v)
                        else:
                            result.setdefault(line, None)
                    except ValueError:
                        pass
        i += 1

    return result


def discover_experiments():
    """Discover all experiments in the results directory (recursive).

    An experiment directory is identified by containing a 'logs/' or 'evaluation/' subdirectory.
    The group is the relative path from RESULTS_DIR to the experiment's parent.
    """
    experiments = []
    if not os.path.exists(RESULTS_DIR):
        return experiments

    def _is_experiment(path):
        return os.path.isdir(os.path.join(path, "logs")) or os.path.isdir(
            os.path.join(path, "evaluation")
        )

    def _scan(dir_path, max_depth=5):
        if max_depth <= 0:
            return
        try:
            entries = sorted(os.listdir(dir_path))
        except PermissionError:
            return
        for entry in entries:
            entry_path = os.path.join(dir_path, entry)
            if not os.path.isdir(entry_path):
                continue
            if _is_experiment(entry_path):
                rel = os.path.relpath(entry_path, RESULTS_DIR)
                group = os.path.dirname(rel)
                name = os.path.basename(rel)
                if not group:
                    group = "(root)"
                log_path = os.path.join(entry_path, "logs", "logging.log")
                has_log = os.path.exists(log_path)
                eval_dir = os.path.join(entry_path, "evaluation")
                eval_files = []
                if os.path.isdir(eval_dir):
                    for root, dirs, files in os.walk(eval_dir):
                        for f in files:
                            if f.endswith(".txt"):
                                eval_files.append(
                                    os.path.relpath(os.path.join(root, f), eval_dir)
                                )
                log_size = os.path.getsize(log_path) if has_log else 0
                log_mtime = os.path.getmtime(log_path) if has_log else 0
                experiments.append(
                    {
                        "group": group,
                        "name": name,
                        "path": entry_path,
                        "has_log": has_log,
                        "log_size": log_size,
                        "log_mtime": log_mtime,
                        "eval_files": sorted(eval_files),
                    }
                )
            else:
                _scan(entry_path, max_depth - 1)

    _scan(RESULTS_DIR)
    return experiments


# ============ Visualization Functions ============

def discover_vis_experiments():
    """Discover all experiments with visualization directory."""
    experiments = []
    if not os.path.exists(RESULTS_DIR):
        return experiments

    def _has_visualization(path):
        return os.path.isdir(os.path.join(path, "visualization"))

    def _scan(dir_path, max_depth=4):
        if max_depth <= 0:
            return
        try:
            entries = sorted(os.listdir(dir_path))
        except PermissionError:
            return
        for entry in entries:
            entry_path = os.path.join(dir_path, entry)
            if not os.path.isdir(entry_path):
                continue
            if _has_visualization(entry_path):
                rel = os.path.relpath(entry_path, RESULTS_DIR)
                group = os.path.dirname(rel)
                name = os.path.basename(rel)
                if not group:
                    group = "(root)"
                vis_path = os.path.join(entry_path, "visualization")
                vis_mtime = os.path.getmtime(vis_path) if os.path.exists(vis_path) else 0
                experiments.append({
                    "group": group,
                    "name": name,
                    "path": entry_path,
                    "vis_mtime": vis_mtime,
                })
            else:
                _scan(entry_path, max_depth - 1)

    _scan(RESULTS_DIR)
    return experiments


def discover_iterations(vis_dir):
    """Discover all iterations in a visualization directory."""
    iterations = []
    if not os.path.exists(vis_dir):
        return iterations
    
    for entry in sorted(os.listdir(vis_dir)):
        if entry.startswith("iter_"):
            iterations.append(entry)
    return iterations


def _detect_dataset_prefix(base_dir):
    """Check if a directory has dataset subdirectories (multi-dataset structure)."""
    if not os.path.isdir(base_dir):
        return False
    for entry in os.listdir(base_dir):
        if entry == "glb":
            continue
        if os.path.isdir(os.path.join(base_dir, entry)):
            return True
    return False


def _get_dataset_prefix(file_root, base_dir):
    """Extract dataset prefix from file path relative to base_dir."""
    rel = os.path.relpath(file_root, base_dir)
    parts = rel.replace("\\", "/").split("/")
    if parts and parts[0] != ".":
        return parts[0]
    return ""


def discover_samples(vis_dir, iteration=None):
    """Discover all samples in a visualization directory."""
    samples = {}
    if not os.path.exists(vis_dir):
        return samples

    gt_dir = os.path.join(vis_dir, "gt")
    multi_ds = _detect_dataset_prefix(gt_dir)

    if os.path.exists(gt_dir):
        for root, dirs, files in os.walk(gt_dir):
            ds_prefix = _get_dataset_prefix(root, gt_dir) if multi_ds else ""
            for f in files:
                match = re.search(r"_(\d{6})\.(jpg|ply)$", f)
                if match:
                    raw_id = match.group(1)
                    sample_id = f"{ds_prefix}/{raw_id}" if ds_prefix else raw_id
                    if sample_id not in samples:
                        samples[sample_id] = {"gt": {}, "pred": {}, "frames": {}}
                    fpath = os.path.join(root, f)
                    rel_path = os.path.relpath(fpath, vis_dir)
                    if f.startswith("merge_rgb"):
                        samples[sample_id]["gt"]["merge_rgb"] = rel_path
                    elif f.startswith("points_gt"):
                        samples[sample_id]["gt"]["points_gt"] = rel_path
                    elif f.startswith("glb_points_gt"):
                        samples[sample_id]["gt"]["glb_points_gt"] = rel_path

    iter_dirs = []
    for entry in os.listdir(vis_dir):
        if entry.startswith("iter_"):
            iter_dirs.append(entry)

    if not iter_dirs:
        return samples

    iter_dirs.sort()
    if iteration and iteration in iter_dirs:
        target_iter = iteration
    else:
        target_iter = iter_dirs[-1]

    iter_dir = os.path.join(vis_dir, target_iter)
    iter_multi_ds = _detect_dataset_prefix(iter_dir) if not multi_ds else multi_ds

    for root, dirs, files in os.walk(iter_dir):
        ds_prefix = _get_dataset_prefix(root, iter_dir) if (multi_ds or iter_multi_ds) else ""
        for f in files:
            match = re.search(r"_(\d{6})\.(jpg|ply)$", f)
            if match:
                raw_id = match.group(1)
                sample_id = f"{ds_prefix}/{raw_id}" if ds_prefix else raw_id
                if sample_id not in samples:
                    samples[sample_id] = {"gt": {}, "pred": {}, "frames": {}}
                fpath = os.path.join(root, f)
                rel_path = os.path.relpath(fpath, vis_dir)

                frame_match = re.match(r"f(\d{3})_v(\d{3})_(.+)_\d{6}\.(jpg|ply)$", f)
                if frame_match:
                    frame_id = frame_match.group(1)
                    file_type = frame_match.group(3)
                    if frame_id not in samples[sample_id]["frames"]:
                        samples[sample_id]["frames"][frame_id] = {}
                    samples[sample_id]["frames"][frame_id][file_type] = rel_path
                elif f.startswith("glb_points"):
                    samples[sample_id]["pred"]["glb_point"] = rel_path
                elif f.startswith("lcl2glb_points"):
                    samples[sample_id]["pred"]["lcl2glb_point"] = rel_path
                elif "depth" in f:
                    samples[sample_id]["pred"]["depth"] = rel_path
                elif "error" in f:
                    samples[sample_id]["pred"]["error"] = rel_path
                elif "conf_mask" in f:
                    samples[sample_id]["pred"]["conf_mask"] = rel_path
                elif "conf" in f and "mask" not in f:
                    samples[sample_id]["pred"]["conf"] = rel_path
                elif "filtered_point" in f:
                    samples[sample_id]["pred"]["filtered_point"] = rel_path
                elif "point" in f and "filtered" not in f:
                    samples[sample_id]["pred"]["point"] = rel_path

    # For single-view experiments (only f000 frame), copy frame point clouds to pred
    for sample_id, sample_data in samples.items():
        frames = sample_data.get("frames", {})
        pred = sample_data.get("pred", {})
        # If only one frame (f000) and pred doesn't have point cloud, use frame's
        if len(frames) == 1 and "000" in frames:
            frame_data = frames["000"]
            if "point" not in pred and "point" in frame_data:
                pred["point"] = frame_data["point"]
            if "filtered_point" not in pred and "filtered_point" in frame_data:
                pred["filtered_point"] = frame_data["filtered_point"]

    return samples


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/experiments")
def api_experiments():
    exps = discover_experiments()
    return jsonify(exps)


def _build_experiment_detail(exp):
    """Build detail info for a single experiment (used in parallel)."""
    group = exp["group"]
    name = exp["name"]
    path = exp["path"]

    info = {
        "group": group,
        "name": name,
        "path": path,
        "has_log": exp["has_log"],
        "log_size": exp["log_size"],
        "start_time": None,
        "last_time": None,
        "elapsed": None,
        "eta": None,
        "current_iter": None,
        "total_iter": None,
        "status": "unknown",
        "num_gpus": None,
        "gpu_model": None,
    }

    if not exp["has_log"]:
        return info

    log_path = os.path.join(path, "logs", "logging.log")
    try:
        with open(log_path, "r") as f:
            first_line = f.readline()
            f.seek(0)
            first_chunk = f.read(300000)
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 100000))
            lines = f.readlines()
            last_line = lines[-1] if lines else ""

        time_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"

        start_match = re.search(time_pattern, first_line)
        if start_match:
            info["start_time"] = start_match.group(1)

        last_match = re.search(time_pattern, last_line)
        if last_match:
            info["last_time"] = last_match.group(1)

        total_match = re.search(r"Total optimization steps\s*=\s*(\d+)", first_chunk)
        if total_match:
            info["total_iter"] = int(total_match.group(1))
        else:
            max_iter_match = re.search(
                r"['\"]?max_iter['\"]?\s*[:=]\s*(\d+)", first_chunk
            )
            if max_iter_match:
                info["total_iter"] = int(max_iter_match.group(1))

        num_proc_match = re.search(
            r"['\"]?num_processes['\"]?\s*[:=]\s*(\d+)", first_chunk
        )
        if num_proc_match:
            info["num_gpus"] = int(num_proc_match.group(1))

        gpu_match = re.search(
            r"_CudaDeviceProperties\(name=['\"]([^'\"]+)['\"]", first_chunk
        )
        if gpu_match:
            info["gpu_model"] = gpu_match.group(1)

        for line in reversed(lines[-500:]):
            if info["current_iter"] is None:
                iter_match = re.search(r"- train - iter(\d+)[,\s]", line)
                if iter_match:
                    info["current_iter"] = int(iter_match.group(1))

            if info["elapsed"] is None:
                total_time_match = re.search(
                    r"total:([\d\s]+days?,\s*)?(\d+:\d+:\d+)", line
                )
                if total_time_match:
                    days_part = total_time_match.group(1)
                    time_part = total_time_match.group(2)
                    if days_part:
                        days_num = re.search(r"(\d+)", days_part)
                        if days_num:
                            info["elapsed"] = f"{days_num.group(1)}d {time_part}"
                        else:
                            info["elapsed"] = time_part
                    else:
                        info["elapsed"] = time_part

            if info["eta"] is None:
                eta_match = re.search(
                    r"eta:([\d\s]+days?,\s*)?(\d+:\d+:\d+)", line
                )
                if eta_match:
                    days_part = eta_match.group(1)
                    time_part = eta_match.group(2)
                    if days_part:
                        days_num = re.search(r"(\d+)", days_part)
                        if days_num:
                            info["eta"] = f"{days_num.group(1)}d {time_part}"
                        else:
                            info["eta"] = time_part
                    else:
                        info["eta"] = time_part

            if info["current_iter"] and info["elapsed"] and info["eta"]:
                break

        last_mtime = exp["log_mtime"]
        is_active = last_mtime and (time.time() - last_mtime) < 120
        is_finished = (
            info["current_iter"]
            and info["total_iter"]
            and info["current_iter"] >= info["total_iter"]
        )

        eta_is_zero = False
        if info["eta"]:
            try:
                eta_str = info["eta"]
                if "days" in eta_str or "day" in eta_str:
                    eta_is_zero = False
                else:
                    eta_parts = eta_str.split(":")
                    eta_is_zero = all(int(p.strip()) == 0 for p in eta_parts)
            except:
                eta_is_zero = False

        if is_active:
            info["status"] = "running"
        elif is_finished or eta_is_zero:
            info["status"] = "completed"
        elif info["eta"] and not eta_is_zero:
            info["status"] = "error"
        else:
            info["status"] = "stopped"

    except Exception:
        info["status"] = "error"

    return info


@app.route("/api/experiment_details")
def api_experiment_details():
    """Get detailed information about all experiments including timing."""
    from concurrent.futures import ThreadPoolExecutor

    exps = discover_experiments()
    with ThreadPoolExecutor(max_workers=min(len(exps) or 1, 16)) as pool:
        details = list(pool.map(_build_experiment_detail, exps))
    return jsonify(details)


@app.route("/api/backup_file")
def api_backup_file():
    """Get backup.py content for an experiment."""
    group = request.args.get("group", "")
    name = request.args.get("name", "")
    
    if group == "(root)":
        group = ""
    
    exp_dir = os.path.join(RESULTS_DIR, group, name)
    
    # Find *_backup.py file in experiment directory
    backup_path = None
    if os.path.isdir(exp_dir):
        for f in os.listdir(exp_dir):
            if f.endswith("_backup.py"):
                backup_path = os.path.join(exp_dir, f)
                break
    
    if not backup_path or not os.path.exists(backup_path):
        return jsonify({"content": "backup.py not found", "exists": False})
    
    try:
        with open(backup_path, "r") as f:
            content = f.read()
        return jsonify({"content": content, "exists": True, "filename": os.path.basename(backup_path)})
    except Exception as e:
        return jsonify({"content": f"Error reading file: {str(e)}", "exists": False})


@app.route("/api/training_data")
def api_training_data():
    group = request.args.get("group", "")
    name = request.args.get("name", "")
    since_iter = request.args.get("since_iter", None)
    sample = request.args.get("sample", None)

    if since_iter is not None:
        since_iter = int(since_iter)
    if sample is not None:
        sample = int(sample)

    log_path = os.path.join(RESULTS_DIR, group, name, "logs", "logging.log")
    records, timing = parse_log_file(log_path, since_iter=since_iter)

    if sample and len(records) > sample:
        step = len(records) // sample
        records = records[::step]

    return jsonify({"records": records, "timing": timing})


@app.route("/api/eval_data")
def api_eval_data():
    group = request.args.get("group", "")
    name = request.args.get("name", "")
    eval_dir = os.path.join(RESULTS_DIR, group, name, "evaluation")

    results = []
    if os.path.isdir(eval_dir):
        for root, dirs, files in os.walk(eval_dir):
            for f in sorted(files):
                if f.endswith(".txt"):
                    fpath = os.path.join(root, f)
                    parsed = parse_eval_file(fpath)
                    if parsed:
                        parsed["file"] = os.path.relpath(fpath, eval_dir)
                        results.append(parsed)

    return jsonify(results)


@app.route("/api/compare_batch", methods=["POST"])
def api_compare_batch():
    """Batch fetch training timing + eval data for multiple experiments in one request."""
    payload = request.get_json(force=True)
    experiments = payload.get("experiments", [])
    include_train = payload.get("include_train", False)
    sample = payload.get("sample", 1000)

    from concurrent.futures import ThreadPoolExecutor

    def process_one(exp):
        group = exp.get("group", "")
        name = exp.get("name", "")
        key = f"{group}/{name}"
        result = {"key": key, "group": group, "name": name}

        log_path = os.path.join(RESULTS_DIR, group, name, "logs", "logging.log")
        if include_train:
            records, timing = parse_log_file(log_path)
            if sample and len(records) > sample:
                step = len(records) // sample
                records = records[::step]
            result["train"] = {"records": records, "timing": timing}
        else:
            timing = _parse_timing_only(log_path)
            result["train"] = {"records": [], "timing": timing}

        # Eval data
        eval_dir = os.path.join(RESULTS_DIR, group, name, "evaluation")
        eval_results = []
        if os.path.isdir(eval_dir):
            for root, dirs, files in os.walk(eval_dir):
                for f in sorted(files):
                    if f.endswith(".txt"):
                        fpath = os.path.join(root, f)
                        parsed = parse_eval_file(fpath)
                        if parsed:
                            parsed["file"] = os.path.relpath(fpath, eval_dir)
                            eval_results.append(parsed)
        result["eval"] = eval_results
        return result

    with ThreadPoolExecutor(max_workers=min(len(experiments), 8)) as pool:
        results = list(pool.map(process_one, experiments))

    return jsonify(results)


@app.route("/api/log_tail")
def api_log_tail():
    """Get the last N lines of a log file for live viewing."""
    group = request.args.get("group", "")
    name = request.args.get("name", "")
    n = int(request.args.get("n", 50))
    log_path = os.path.join(RESULTS_DIR, group, name, "logs", "logging.log")

    if not os.path.exists(log_path):
        return jsonify({"lines": [], "total_lines": 0})

    with open(log_path, "r") as f:
        lines = f.readlines()

    total = len(lines)
    tail = lines[-n:]
    return jsonify({"lines": [l.rstrip() for l in tail], "total_lines": total})


@app.route("/api/model_info")
def api_model_info():
    """Get model architecture info from model.log."""
    group = request.args.get("group", "")
    name = request.args.get("name", "")
    model_path = os.path.join(RESULTS_DIR, group, name, "logs", "model.log")

    if not os.path.exists(model_path):
        return jsonify({"content": ""})

    with open(model_path, "r") as f:
        content = f.read()

    return jsonify({"content": content})


@app.route("/api/config")
def api_config():
    """Get experiment config."""
    group = request.args.get("group", "")
    name = request.args.get("name", "")
    exp_path = os.path.join(RESULTS_DIR, group, name)

    config_content = ""
    for f in os.listdir(exp_path):
        if f.endswith(".py") and "backup" not in f:
            with open(os.path.join(exp_path, f), "r") as fh:
                config_content = fh.read()
            break

    return jsonify({"content": config_content})


# ============ Visualization Routes ============

@app.route("/vis")
def vis_index():
    return send_from_directory(app.static_folder, "vis.html")


@app.route("/api/vis/experiments")
def api_vis_experiments():
    exps = discover_vis_experiments()
    return jsonify(exps)


@app.route("/api/vis/iterations")
def api_vis_iterations():
    group = request.args.get("group", "")
    name = request.args.get("name", "")
    
    if group == "(root)":
        group = ""
    
    vis_dir = os.path.join(RESULTS_DIR, group, name, "visualization")
    iterations = discover_iterations(vis_dir)
    return jsonify(iterations)


@app.route("/api/vis/samples")
def api_vis_samples():
    group = request.args.get("group", "")
    name = request.args.get("name", "")
    iteration = request.args.get("iteration", "")
    
    if group == "(root)":
        group = ""
    
    vis_dir = os.path.join(RESULTS_DIR, group, name, "visualization")
    samples = discover_samples(vis_dir, iteration if iteration else None)
    
    result = []
    for sample_id in sorted(samples.keys()):
        result.append({
            "id": sample_id,
            "gt": samples[sample_id]["gt"],
            "pred": samples[sample_id]["pred"],
            "frames": samples[sample_id].get("frames", {}),
        })
    return jsonify(result)


@app.route("/api/vis/file")
def api_vis_file():
    """Serve a file from an experiment's visualization directory."""
    group = request.args.get("group", "")
    name = request.args.get("name", "")
    filepath = request.args.get("path", "")
    
    if group == "(root)":
        group = ""
    
    full_path = os.path.join(RESULTS_DIR, group, name, "visualization", filepath)
    if not os.path.exists(full_path):
        return jsonify({"error": "File not found"}), 404

    if filepath.endswith(".ply"):
        return send_file(full_path, mimetype="application/octet-stream")
    elif filepath.endswith(".jpg") or filepath.endswith(".jpeg"):
        return send_file(full_path, mimetype="image/jpeg")
    elif filepath.endswith(".png"):
        return send_file(full_path, mimetype="image/png")
    else:
        return send_file(full_path)


# ============ Config Parsing & Analysis Report ============


def _ast_node_to_value(node):
    """Convert an AST node to a Python value, falling back to source repr."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        pass
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict":
        d = {}
        for kw in node.keywords:
            if kw.arg:
                d[kw.arg] = _ast_node_to_value(kw.value)
        return d
    if isinstance(node, ast.List):
        return [_ast_node_to_value(el) for el in node.elts]
    if isinstance(node, ast.Name):
        return f"<var:{node.id}>"
    return ast.dump(node)


def _flatten_dict(d, prefix=""):
    """Flatten a nested dict into dot-notation keys."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
        if isinstance(v, dict):
            out.update(_flatten_dict(v, key))
        else:
            out[key] = v
    return out


def _parse_config_to_dict(filepath):
    """Parse a Python config file into a flat dict using AST (safe, no exec)."""
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r") as f:
        source = f.read()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    scope = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    scope[target.id] = _ast_node_to_value(node.value)
                elif isinstance(target, ast.Tuple):
                    vals = node.value
                    if isinstance(vals, (ast.Tuple, ast.List)) and len(vals.elts) == len(target.elts):
                        for t, v in zip(target.elts, vals.elts):
                            if isinstance(t, ast.Name):
                                scope[t.id] = _ast_node_to_value(v)
    return _flatten_dict(scope)


def _find_backup_path(group, name):
    """Find the *_backup.py file for an experiment."""
    exp_dir = os.path.join(RESULTS_DIR, group, name)
    if not os.path.isdir(exp_dir):
        return None
    for f in os.listdir(exp_dir):
        if f.endswith("_backup.py"):
            return os.path.join(exp_dir, f)
    return None


def _diff_configs(configs, baseline_idx=0):
    """Compare N flat config dicts against baseline; return only differing params."""
    if not configs:
        return {}
    baseline = configs[baseline_idx]
    all_keys = set()
    for c in configs:
        all_keys.update(c.keys())

    diffs = {}
    for k in sorted(all_keys):
        vals = [c.get(k) for c in configs]
        base_val = vals[baseline_idx]
        if any(_val_repr(v) != _val_repr(base_val) for v in vals):
            diffs[k] = vals
    return diffs


def _val_repr(v):
    """Stable string repr for comparison."""
    if isinstance(v, float):
        return f"{v:.10g}"
    return str(v)


def _val_display(v):
    """Format a value for display in the report."""
    if v is None:
        return "-"
    if isinstance(v, float):
        if abs(v) < 1e-3 or abs(v) > 1e5:
            return f"{v:.4e}"
        return f"{v:.6g}"
    if isinstance(v, list) and len(str(v)) > 80:
        return f"[list, len={len(v)}]"
    s = str(v)
    if len(s) > 100:
        return s[:97] + "..."
    return s


def _generate_analysis_report(experiments, baseline_idx=0):
    """Generate a Markdown analysis report for given experiments."""
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime

    n = len(experiments)
    names = [e["name"] for e in experiments]
    short_names = [n.split("/")[-1] if "/" in n else n for n in names]

    # --- Fetch experiment details ---
    exp_objs = []
    for e in experiments:
        exp_objs.append({"group": e["group"], "name": e["name"],
                         "path": os.path.join(RESULTS_DIR, e["group"], e["name"]),
                         "has_log": os.path.exists(os.path.join(RESULTS_DIR, e["group"], e["name"], "logs", "logging.log")),
                         "log_size": 0, "log_mtime": 0})
        lp = os.path.join(RESULTS_DIR, e["group"], e["name"], "logs", "logging.log")
        if os.path.exists(lp):
            exp_objs[-1]["log_size"] = os.path.getsize(lp)
            exp_objs[-1]["log_mtime"] = os.path.getmtime(lp)

    with ThreadPoolExecutor(max_workers=min(n, 8)) as pool:
        details = list(pool.map(_build_experiment_detail, exp_objs))

    # --- Parse configs ---
    configs = []
    for e in experiments:
        bp = _find_backup_path(e["group"], e["name"])
        configs.append(_parse_config_to_dict(bp) if bp else {})

    diffs = _diff_configs(configs, baseline_idx)

    # --- Fetch eval data ---
    all_evals = []
    for e in experiments:
        eval_dir = os.path.join(RESULTS_DIR, e["group"], e["name"], "evaluation")
        evals = []
        if os.path.isdir(eval_dir):
            for root, dirs, files in os.walk(eval_dir):
                for f in sorted(files):
                    if f.endswith(".txt"):
                        parsed = parse_eval_file(os.path.join(root, f))
                        if parsed:
                            parsed["file"] = f
                            evals.append(parsed)
        all_evals.append(evals)

    # --- Build Markdown ---
    md = []
    md.append(f"# Experiment Analysis Report")
    md.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n")
    md.append(f"**Baseline**: {short_names[baseline_idx]}\n")

    # Section 1: Overview
    md.append("## 1. Experiment Overview\n")
    md.append("| # | Experiment | Status | Progress | Elapsed | ETA |")
    md.append("|---|-----------|--------|----------|---------|-----|")
    for i, d in enumerate(details):
        marker = " **(baseline)**" if i == baseline_idx else ""
        cur = d.get("current_iter") or "?"
        tot = d.get("total_iter") or "?"
        progress = f"{cur}/{tot}" if cur != "?" and tot != "?" else str(cur)
        md.append(f"| {i+1} | {short_names[i]}{marker} | {d.get('status','-')} | {progress} | {d.get('elapsed','-')} | {d.get('eta','-')} |")
    md.append("")

    # Section 2: Parameter Differences
    md.append("## 2. Parameter Differences\n")
    if not diffs:
        md.append("*No parameter differences found (or backup.py not available).*\n")
    else:
        groups = {}
        for k, vals in diffs.items():
            cat = k.split(".")[0] if "." in k else "top-level"
            if cat not in groups:
                groups[cat] = []
            groups[cat].append((k, vals))

        for cat in sorted(groups.keys()):
            md.append(f"### {cat}\n")
            header = "| Parameter | " + " | ".join(short_names) + " |"
            sep = "|---|" + "|".join(["---"] * n) + "|"
            md.append(header)
            md.append(sep)
            for k, vals in groups[cat]:
                row = f"| `{k}` |"
                for i, v in enumerate(vals):
                    cell = _val_display(v)
                    if i == baseline_idx:
                        cell = f"**{cell}**"
                    row += f" {cell} |"
                md.append(row)
            md.append("")

    # Section 3: Evaluation Metrics
    md.append("## 3. Evaluation Metrics\n")
    skip_keys = {"file", "format_text", "dataset"}
    ds_map = {}
    for i, evals in enumerate(all_evals):
        for ev in evals:
            fname = ev.get("file", "")
            if "-latest." in fname:
                ds = re.sub(r"^eval-|-latest\.txt$", "", fname)
                if ds not in ds_map:
                    ds_map[ds] = {}
                ds_map[ds][i] = ev

    if not ds_map:
        md.append("*No evaluation data available.*\n")
    else:
        for ds in sorted(ds_map.keys()):
            md.append(f"### Dataset: {ds}\n")
            exp_evals = ds_map[ds]
            metric_keys = []
            seen = set()
            for idx in sorted(exp_evals.keys()):
                for k in exp_evals[idx].keys():
                    if k not in skip_keys and k not in seen and isinstance(exp_evals[idx][k], (int, float)):
                        seen.add(k)
                        metric_keys.append(k)

            present = sorted(exp_evals.keys())
            header = "| Metric | " + " | ".join(short_names[i] for i in present) + " |"
            sep = "|---|" + "|".join(["---"] * len(present)) + "|"
            md.append(header)
            md.append(sep)
            for k in metric_keys:
                row = f"| `{k}` |"
                for i in present:
                    v = exp_evals[i].get(k)
                    if isinstance(v, (int, float)):
                        row += f" {v:.5f} |"
                    else:
                        row += " - |"
                md.append(row)
            md.append("")

    # Section 4: Analysis Summary
    md.append("## 4. Analysis Summary\n")
    base_evals = {}
    for ds, exp_map in ds_map.items():
        if baseline_idx in exp_map:
            base_evals[ds] = exp_map[baseline_idx]

    for i, e in enumerate(experiments):
        if i == baseline_idx:
            continue
        md.append(f"### {short_names[i]} vs baseline\n")
        param_changes = []
        for k, vals in diffs.items():
            bv = vals[baseline_idx]
            ev = vals[i]
            if _val_repr(bv) != _val_repr(ev):
                param_changes.append(f"`{k}`: {_val_display(bv)} → {_val_display(ev)}")
        if param_changes:
            md.append("**Parameter changes:**")
            for pc in param_changes:
                md.append(f"- {pc}")
            md.append("")

        metric_deltas = []
        for ds in sorted(ds_map.keys()):
            if baseline_idx not in ds_map[ds] or i not in ds_map[ds]:
                continue
            base_ev = ds_map[ds][baseline_idx]
            cur_ev = ds_map[ds][i]
            for k in base_ev:
                if k in skip_keys or not isinstance(base_ev.get(k), (int, float)):
                    continue
                bv = base_ev[k]
                cv = cur_ev.get(k)
                if cv is None or not isinstance(cv, (int, float)):
                    continue
                if bv != 0:
                    pct = (cv - bv) / abs(bv) * 100
                    direction = "+" if pct > 0 else ""
                    metric_deltas.append(f"[{ds}] `{k}`: {bv:.5f} → {cv:.5f} ({direction}{pct:.1f}%)")
        if metric_deltas:
            md.append("**Metric changes:**")
            for delta in metric_deltas:
                md.append(f"- {delta}")
            md.append("")

        if not param_changes and not metric_deltas:
            md.append("*No significant differences.*\n")

    return "\n".join(md)


@app.route("/api/analysis_report", methods=["POST"])
def api_analysis_report():
    payload = request.get_json(force=True)
    experiments = payload.get("experiments", [])
    baseline_idx = payload.get("baseline_index", 0)
    if len(experiments) < 2:
        return jsonify({"error": "Need at least 2 experiments"}), 400
    try:
        md = _generate_analysis_report(experiments, baseline_idx)
        title = f"report_{time.strftime('%Y%m%d_%H%M%S')}"
        return jsonify({"markdown": md, "title": title})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ============ Feishu Notification ============

FEISHU_WEBHOOK_URL = os.environ.get("FEISHU_WEBHOOK_URL", "")

import threading
import urllib.request

_exp_status_cache = {}
_feishu_lock = threading.Lock()


def feishu_send(title, content, color="green"):
    """Send a message to Feishu via webhook."""
    if not FEISHU_WEBHOOK_URL:
        return
    color_map = {
        "green": "green",
        "red": "red",
        "orange": "orange",
        "blue": "blue",
    }
    tag = color_map.get(color, "green")
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": title},
                "template": tag,
            },
            "elements": [
                {
                    "tag": "markdown",
                    "content": content,
                }
            ],
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        FEISHU_WEBHOOK_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"[Feishu] Failed to send: {e}")


def _detect_status(exp_detail):
    """Return a normalized status tuple: (status, current_iter, total_iter, elapsed)."""
    return (
        exp_detail.get("status", "unknown"),
        exp_detail.get("current_iter"),
        exp_detail.get("total_iter"),
        exp_detail.get("elapsed"),
    )


def _check_status_changes():
    """Scan all experiments and detect status transitions."""
    global _exp_status_cache
    from concurrent.futures import ThreadPoolExecutor

    exps = discover_experiments()
    with ThreadPoolExecutor(max_workers=min(len(exps) or 1, 16)) as pool:
        details = list(pool.map(_build_experiment_detail, exps))

    new_cache = {}
    for d in details:
        key = f"{d['group']}/{d['name']}"
        new_status = _detect_status(d)
        new_cache[key] = new_status

        if key not in _exp_status_cache:
            continue

        old_status = _exp_status_cache[key]
        old_s = old_status[0]
        new_s = new_status[0]

        if old_s == new_s:
            continue

        name = d["name"]
        group = d["group"]
        cur_iter = d.get("current_iter", "?")
        total_iter = d.get("total_iter", "?")
        elapsed = d.get("elapsed", "?")
        progress = (
            f"{cur_iter}/{total_iter}"
            if cur_iter and total_iter
            else str(cur_iter or "?")
        )

        if new_s == "completed" and old_s == "running":
            feishu_send(
                "✅ Training Completed",
                f"**{name}**\n"
                f"- Group: {group}\n"
                f"- Progress: {progress}\n"
                f"- Elapsed: {elapsed}",
                color="green",
            )
        elif new_s == "error" and old_s == "running":
            feishu_send(
                "❌ Training Failed",
                f"**{name}**\n"
                f"- Group: {group}\n"
                f"- Stopped at: {progress}\n"
                f"- Elapsed: {elapsed}",
                color="red",
            )
        elif new_s == "stopped" and old_s == "running":
            feishu_send(
                "⚠️ Training Stopped",
                f"**{name}**\n"
                f"- Group: {group}\n"
                f"- Stopped at: {progress}\n"
                f"- Elapsed: {elapsed}",
                color="orange",
            )

    with _feishu_lock:
        _exp_status_cache = new_cache


def _push_running_tasks():
    """Send a summary of all running tasks to Feishu."""
    from concurrent.futures import ThreadPoolExecutor
    exps = discover_experiments()
    if not exps:
        return
    with ThreadPoolExecutor(max_workers=min(len(exps), 16)) as pool:
        details = list(pool.map(_build_experiment_detail, exps))
    running = [d for d in details if d.get("status") == "running"]
    if not running:
        return
    lines = []
    for d in running:
        name = d["name"]
        cur = d.get("current_iter") or "?"
        tot = d.get("total_iter") or "?"
        elapsed = d.get("elapsed") or "-"
        eta = d.get("eta") or "-"
        progress = f"{cur}/{tot}" if cur != "?" and tot != "?" else str(cur)
        pct = ""
        if isinstance(cur, int) and isinstance(tot, int) and tot > 0:
            pct = f" ({cur * 100 // tot}%)"
        lines.append(f"🟢 **{name}**")
        lines.append(f"　　Progress: {progress}{pct} | Elapsed: {elapsed} | ETA: {eta}")
    feishu_send(
        f"🏃 Running Tasks: {len(running)}",
        "\n".join(lines),
        color="blue",
    )


_feishu_interval = 0
_FEISHU_STATUS_CHECK_INTERVAL = 60


def _feishu_monitor_loop(interval=0):
    """Background thread: always checks status changes every 60s, optionally pushes running tasks."""
    global _feishu_interval
    _feishu_interval = interval
    print(f"[Feishu] Monitor started (running_push={interval}s, status_check={_FEISHU_STATUS_CHECK_INTERVAL}s)")
    _check_status_changes()
    last_running_push = time.time()
    while True:
        time.sleep(_FEISHU_STATUS_CHECK_INTERVAL)
        try:
            _check_status_changes()
        except Exception as e:
            print(f"[Feishu] Monitor error: {e}")
        if _feishu_interval > 0 and (time.time() - last_running_push) >= _feishu_interval:
            try:
                _push_running_tasks()
            except Exception as e:
                print(f"[Feishu] Running push error: {e}")
            last_running_push = time.time()


@app.route("/api/feishu/config", methods=["GET"])
def feishu_config_get():
    return jsonify({
        "webhook_url": FEISHU_WEBHOOK_URL,
        "interval": _feishu_interval,
    })


@app.route("/api/feishu/config", methods=["POST"])
def feishu_config_set():
    global FEISHU_WEBHOOK_URL, _feishu_interval
    data = request.get_json(force=True)
    if "webhook_url" in data:
        FEISHU_WEBHOOK_URL = data["webhook_url"].strip()
    if "interval" in data:
        val = int(data["interval"])
        if val >= 0:
            _feishu_interval = val
    print(f"[Feishu] Config updated: webhook={'***' + FEISHU_WEBHOOK_URL[-12:] if FEISHU_WEBHOOK_URL else '(empty)'}, interval={_feishu_interval}s")
    if _feishu_interval > 0 and FEISHU_WEBHOOK_URL:
        threading.Thread(target=_push_running_tasks, daemon=True).start()
    return jsonify({"ok": True, "webhook_url": FEISHU_WEBHOOK_URL, "interval": _feishu_interval})


@app.route("/api/feishu/test", methods=["POST"])
def feishu_test():
    if not FEISHU_WEBHOOK_URL:
        return jsonify({"ok": False, "error": "No webhook URL configured"})
    try:
        feishu_send("🔔 Test Notification", "Monitor is connected successfully!", color="blue")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--feishu-webhook", type=str, default=None,
                        help="Feishu webhook URL for notifications")
    parser.add_argument("--feishu-interval", type=int, default=0,
                        help="Feishu running push interval in seconds (0=disabled)")
    args = parser.parse_args()

    if args.results_dir:
        RESULTS_DIR = args.results_dir
    if args.feishu_webhook:
        FEISHU_WEBHOOK_URL = args.feishu_webhook

    print(f"Monitoring results directory: {RESULTS_DIR}")
    print(f"Starting server at http://{args.host}:{args.port}")

    t = threading.Thread(
        target=_feishu_monitor_loop,
        args=(args.feishu_interval,),
        daemon=True,
    )
    t.start()
    if FEISHU_WEBHOOK_URL:
        print(f"[Feishu] Notifications enabled (interval={args.feishu_interval}s)")
    else:
        print("[Feishu] Notifications idle, configure via Manage > Feishu button")

    app.run(host=args.host, port=args.port, debug=True, threaded=True)
