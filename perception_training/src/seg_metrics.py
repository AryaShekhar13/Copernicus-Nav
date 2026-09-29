"""Torch-free metric + table helpers (unit-testable without a GPU).

Definitions intentionally mirror the project's evaluate.py so numbers stay comparable with
the handover tables: void (ignore_index) GT pixels are excluded; IoU = TP/(TP+FP+FN) with FP
counted only on non-void GT pixels; mIoU = mean over non-void classes that have GT pixels.
"""
import json
import os
import numpy as np

FOCUS_CLASSES = ["water", "mud", "puddle", "person"]


def metrics_from_confusion(conf, ignore_index=0):
    """conf[i, j] = #pixels with GT class i predicted as class j (void GT already excluded)."""
    conf = np.asarray(conf, dtype=np.float64)
    tp, gt, pred = np.diag(conf), conf.sum(1), conf.sum(0)
    union = gt + pred - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, np.nan)
        recall = np.where(gt > 0, tp / gt, np.nan)
        precision = np.where(pred > 0, tp / pred, np.nan)
    present = [c for c in range(len(gt)) if gt[c] > 0 and c != ignore_index]
    return {"iou": iou, "recall": recall, "precision": precision, "gt_px": gt, "present": present}


def summarize_split(conf, ignore_index, names_by_id, hazard_ids, ece_bins=None):
    m = metrics_from_confusion(conf, ignore_index)
    present = m["present"]
    haz = [c for c in present if c in hazard_ids]
    non = [c for c in present if c not in hazard_ids]
    mean = lambda ids: float(np.nanmean(m["iou"][ids])) if ids else float("nan")
    ids = [c for c in sorted(names_by_id) if c != ignore_index]
    out = {
        "miou": mean(present), "hazard_miou": mean(haz), "nonhazard_miou": mean(non),
        "n_present": len(present),
        "iou": {names_by_id[c]: _f(m["iou"][c]) for c in ids},
        "recall": {names_by_id[c]: _f(m["recall"][c]) for c in ids},
        "precision": {names_by_id[c]: _f(m["precision"][c]) for c in ids},
        "gt_px": {names_by_id[c]: int(m["gt_px"][c]) for c in ids},
    }
    if ece_bins is not None:
        out["ece"] = ece_from_bins(*ece_bins)
    return out


def ece_from_bins(count, correct_sum, conf_sum):
    count, correct_sum, conf_sum = (np.asarray(a, dtype=np.float64) for a in (count, correct_sum, conf_sum))
    total = count.sum()
    if total == 0:
        return float("nan")
    ok = count > 0
    acc, conf = correct_sum[ok] / count[ok], conf_sum[ok] / count[ok]
    return float(((count[ok] / total) * np.abs(acc - conf)).sum())


def _f(x):
    x = float(x)
    return None if np.isnan(x) else x


# ------------------------------------------------------------------ tables
def _fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def load_results(results_dir):
    out = []
    for fn in sorted(os.listdir(results_dir)):
        if fn.endswith(".json"):
            with open(os.path.join(results_dir, fn)) as f:
                out.append(json.load(f))
    return out


def build_tables(results, split="test", baseline_label=None):
    """Returns (main_markdown, per_class_markdown, csv_rows)."""
    results = [r for r in results if split in r["splits"]]
    if not results:
        return "(no results)", "", []
    base = next((r for r in results if r["label"] == baseline_label), None) \
        or next((r for r in results if r["model"] == "terrainseg"), results[0])
    other = "val" if split == "test" else "test"

    head = ["Model", "Params (M)", "mIoU", "ΔmIoU", "Haz. mIoU"] + \
           [f"{c} rec." for c in FOCUS_CLASSES] + \
           [f"{split} loss", "val loss" if split == "test" else "test loss",
            "GPU ms fp32", "GPU FPS fp16", "CPU ONNX ms", "Infer MB", "Train MB", "Best ep."]
    rows, csv_rows = [], []
    for r in results:
        s, sp = r["splits"][split], r.get("speed", {})
        o = r["splits"].get(other, {})
        tr = r.get("train", {})
        row = [r["label"], _fmt(r.get("params_m"), 2), _fmt(s["miou"]),
               _fmt(s["miou"] - base["splits"][split]["miou"]) if r is not base else "—",
               _fmt(s["hazard_miou"])] + \
              [_fmt(s["recall"].get(c)) for c in FOCUS_CLASSES] + \
              [_fmt(s.get("loss"), 4), _fmt(o.get("loss"), 4),
               _fmt(sp.get("gpu_fp32_ms"), 1), _fmt(sp.get("gpu_fp16_fps"), 0),
               _fmt(sp.get("cpu_onnx_ms"), 0), _fmt(sp.get("infer_mem_mb"), 0),
               _fmt(sp.get("train_mem_mb"), 0), str(tr.get("best_epoch", "n/a"))]
        rows.append(row)
        csv_rows.append(dict(zip(head, row)))

    md = "| " + " | ".join(head) + " |\n|" + "|".join(["---"] * len(head)) + "|\n"
    md += "\n".join("| " + " | ".join(r) + " |" for r in rows)

    names = list(base["splits"][split]["iou"].keys())
    pc = f"**Per-class IoU ({split})**\n\n| class | " + " | ".join(r["label"] for r in results) + " |\n"
    pc += "|---|" + "---|" * len(results) + "\n"
    for n in names:
        pc += f"| {n} | " + " | ".join(_fmt(r["splits"][split]["iou"].get(n)) for r in results) + " |\n"
    pc += f"\n**Per-class recall ({split})**\n\n| class | " + " | ".join(r["label"] for r in results) + " |\n"
    pc += "|---|" + "---|" * len(results) + "\n"
    for n in names:
        pc += f"| {n} | " + " | ".join(_fmt(r["splits"][split]["recall"].get(n)) for r in results) + " |\n"
    return md, pc, csv_rows
