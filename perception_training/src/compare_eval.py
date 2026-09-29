"""Evaluate + benchmark one checkpoint, or aggregate all results into the comparison table.

  python src/compare_eval.py run   --label segformer_b0_s0 --model segformer_b0 \
         --ckpt /kaggle/working/checkpoints/segformer_b0/best.pt --out /kaggle/working/results
  python src/compare_eval.py table --results /kaggle/working/results

Metrics use the same definitions as the project's evaluate.py (see seg_metrics.py); loss is the
same weighted CrossEntropy(ignore_index=0) as training, averaged per batch like train.py's val loss.
Latency is model-forward only (no pre/post-processing), batch size 1, at training.yaml's input_size.
"""
import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import RellisDataset
from models_zoo import build_model
from preprocessing import build_class_remap
from seg_metrics import summarize_split, load_results, build_tables
from train import build_criterion


@torch.no_grad()
def run_split(model, loader, criterion, device, num_classes, ignore_index, n_bins=15, max_batches=None):
    model.eval()
    conf = torch.zeros(num_classes * num_classes, dtype=torch.long, device=device)
    b_cnt = torch.zeros(n_bins, dtype=torch.float64, device=device)
    b_cor = torch.zeros(n_bins, dtype=torch.float64, device=device)
    b_conf = torch.zeros(n_bins, dtype=torch.float64, device=device)
    losses = []
    for i, (images, masks) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        losses.append(criterion(logits, masks).item())
        confs, preds = torch.softmax(logits, dim=1).max(dim=1)
        valid = masks != ignore_index
        t, p, c = masks[valid], preds[valid], confs[valid].double()
        conf += torch.bincount(t * num_classes + p, minlength=num_classes ** 2)
        idx = (torch.ceil(c * n_bins) - 1).clamp(0, n_bins - 1).long()   # (lo, hi] bins, as evaluate.py
        b_cnt += torch.bincount(idx, minlength=n_bins).double()
        b_cor += torch.bincount(idx, weights=(p == t).double(), minlength=n_bins)
        b_conf += torch.bincount(idx, weights=c, minlength=n_bins)
    conf = conf.view(num_classes, num_classes).cpu().numpy()
    return float(np.mean(losses)), conf, (b_cnt.cpu().numpy(), b_cor.cpu().numpy(), b_conf.cpu().numpy())


def bench_gpu(model, size, amp, warmup=20, iters=100):
    x = torch.randn(1, 3, *size, device="cuda")
    model.eval()
    times = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); model(x); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    t = np.array(times)
    return {"ms": float(t.mean()), "p95_ms": float(np.percentile(t, 95)), "fps": float(1000.0 / t.mean())}


def bench_onnx_cpu(model, size, onnx_path, warmup=5, iters=30):
    """Export exactly like the deployed path (opset 17, legacy exporter), time ORT CPUExecutionProvider."""
    import onnxruntime as ort
    m = copy.deepcopy(model).cpu().eval()
    x = torch.randn(1, 3, *size)
    os.makedirs(os.path.dirname(onnx_path) or ".", exist_ok=True)
    kw = dict(opset_version=17, input_names=["input"], output_names=["logits"])
    try:
        torch.onnx.export(m, x, onnx_path, dynamo=False, **kw)
    except TypeError:            # older torch without the `dynamo` kwarg
        torch.onnx.export(m, x, onnx_path, **kw)
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inp = {sess.get_inputs()[0].name: x.numpy()}
    with torch.no_grad():
        diff = float(np.abs(m(x).numpy() - sess.run(None, inp)[0]).max())
    for _ in range(warmup):
        sess.run(None, inp)
    t = []
    for _ in range(iters):
        t0 = time.perf_counter(); sess.run(None, inp); t.append((time.perf_counter() - t0) * 1000)
    return {"cpu_onnx_ms": float(np.mean(t)), "cpu_onnx_p95_ms": float(np.percentile(t, 95)),
            "onnx_max_abs_diff": diff, "onnx_mb": os.path.getsize(onnx_path) / 2 ** 20,
            "cpu_threads": os.cpu_count()}


def peak_mem_train_mb(model_cpu, criterion, batch_size, size, num_classes, lr):
    m = copy.deepcopy(model_cpu).cuda().train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    x = torch.randn(batch_size, 3, *size, device="cuda")
    y = torch.randint(1, num_classes, (batch_size, *size), device="cuda")
    for _ in range(3):                      # 3 steps so Adam state is allocated
        opt.zero_grad(); criterion(m(x), y).backward(); opt.step()
    torch.cuda.synchronize()
    mb = torch.cuda.max_memory_allocated() / 2 ** 20
    del m, opt
    torch.cuda.empty_cache()
    return mb


def count_gflops(model, size, device):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        model.eval()
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(torch.randn(1, 3, *size, device=device))
        return fc.get_total_flops() / 1e9          # FLOPs = 2 x MACs
    except Exception as e:
        print("FLOP counting unavailable:", e)
        return None


def cmd_run(a):
    with open(a.training_config) as f:
        tcfg = yaml.safe_load(f)
    with open(a.classes_config) as f:
        ccfg = yaml.safe_load(f)
    num_classes, ignore_index = ccfg["num_classes"], ccfg.get("ignore_index", 0)
    size = tuple(tcfg["input_size"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    remap, _ = build_class_remap(a.classes_config)
    names = {remap[c["index"]]: c["name"] for c in ccfg["classes"]}
    hazard_ids = {remap[c["index"]] for c in ccfg["classes"] if c.get("hazardous")}

    ckpt = torch.load(a.ckpt, map_location="cpu")
    extra = ckpt.get("extra", {}) or {}
    model = build_model(a.model, num_classes, pretrained=False,
                        hf_id=a.hf_id or extra.get("hf_id"), dl_encoder=a.dl_encoder or extra.get("dl_encoder"))
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()
    criterion = build_criterion(tcfg, a.training_config, device)

    res = {"label": a.label, "model": a.model, "ckpt": a.ckpt, "ckpt_epoch": ckpt.get("epoch"),
           "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
           "gflops": count_gflops(model, size, device), "splits": {}, "speed": {}}

    for split in a.splits:
        ds = RellisDataset(a.dataset_config, split=split, classes_config_path=a.classes_config,
                           training_config_path=a.training_config)
        loader = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=tcfg["num_workers"])
        loss, conf, ece_bins = run_split(model, loader, criterion, device, num_classes, ignore_index,
                                         max_batches=a.max_batches)
        s = summarize_split(conf, ignore_index, names, hazard_ids, ece_bins)
        s["loss"], s["n_frames"] = loss, len(ds)
        res["splits"][split] = s
        print(f"[{a.label}] {split}: mIoU {s['miou']:.4f} (n={s['n_present']})  hazard mIoU {s['hazard_miou']:.4f}  "
              f"loss {loss:.4f}  ECE {s['ece']:.4f}")
        print("   recall: " + "  ".join(f"{c}={s['recall'].get(c)}" for c in ("water", "mud", "puddle", "person")))

    sp = res["speed"]
    if device == "cuda":
        sp["gpu_name"] = torch.cuda.get_device_name(0)
        for tag, amp in (("fp32", False), ("fp16", True)):
            r = bench_gpu(model, size, amp)
            sp.update({f"gpu_{tag}_ms": r["ms"], f"gpu_{tag}_p95_ms": r["p95_ms"], f"gpu_{tag}_fps": r["fps"]})
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(torch.randn(1, 3, *size, device="cuda"))
        sp["infer_mem_mb"] = torch.cuda.max_memory_allocated() / 2 ** 20   # weights + activations, bs=1, fp32
        model_cpu = model.cpu()
        torch.cuda.empty_cache()
        sp["train_mem_mb"] = peak_mem_train_mb(model_cpu, criterion, tcfg["batch_size"], size,
                                               num_classes, tcfg["learning_rate"])   # fwd+bwd+Adam, fp32
        sp["train_mem_batch"] = tcfg["batch_size"]
    if not a.skip_onnx:
        try:
            sp.update(bench_onnx_cpu(model, size, os.path.join(a.out, "onnx", f"{a.label}.onnx")))
        except Exception as e:                                       # keep going; record why
            sp["onnx_error"] = repr(e)
            print("ONNX export/benchmark failed:", repr(e))

    hist_path = os.path.join(os.path.dirname(a.ckpt), "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            h = json.load(f)
        best = min(h, key=lambda r: r["val_loss"])
        res["train"] = {"epochs_run": len(h), "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
                        "train_minutes": sum(r["epoch_seconds"] for r in h) / 60,
                        "history": h}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.label}.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"[{a.label}] saved -> {os.path.join(a.out, a.label + '.json')}")


def cmd_table(a):
    results = load_results(a.results)
    md, pc, rows = build_tables(results, split=a.split, baseline_label=a.baseline)
    out_md = os.path.join(a.results, f"comparison_{a.split}.md")
    with open(out_md, "w") as f:
        f.write(md + "\n\n" + pc)
    if rows:
        import csv
        with open(os.path.join(a.results, f"comparison_{a.split}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(md + "\n\n" + pc)
    print("saved:", out_md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--label", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--ckpt", required=True)
    r.add_argument("--out", default="/kaggle/working/results")
    r.add_argument("--splits", nargs="+", default=["val", "test"])
    r.add_argument("--hf-id", default=None)
    r.add_argument("--dl-encoder", default=None)
    r.add_argument("--max-batches", type=int, default=None, help="smoke tests only")
    r.add_argument("--skip-onnx", action="store_true")
    r.add_argument("--dataset-config", default="configs/dataset.yaml")
    r.add_argument("--classes-config", default="configs/classes.yaml")
    r.add_argument("--training-config", default="configs/training.yaml")
    r.set_defaults(fn=cmd_run)
    t = sub.add_parser("table")
    t.add_argument("--results", default="/kaggle/working/results")
    t.add_argument("--split", default="test")
    t.add_argument("--baseline", default=None, help="label to compare against")
    t.set_defaults(fn=cmd_table)
    args = ap.parse_args()
    args.fn(args)
