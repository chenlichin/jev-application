"""Compare classic Kaggle baselines with zero-shot Jev inference on five Kaggle problems.

Usage:
    python run_benchmark.py --mode baseline               # no API key needed
    python run_benchmark.py --mode dry-run                # write example Jev requests, no API calls
    TYPESAFE_API_KEY=... python run_benchmark.py --mode both --jev-n 200
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from jev_bench.datasets import ROOT, split
from jev_bench.jev_runner import request_payload, run_jev
from jev_bench.tasks import TASKS, Task, positive_scores

RESULTS_DIR = ROOT / "results"


def metrics(task: Task, y_true: Any, y_pred: Any, y_score: Any | None) -> dict[str, float]:
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=task.labels)),
    }
    if task.kind == "binary":
        out["f1_positive"] = float(f1_score(y_true, y_pred, pos_label=1))
        if y_score is not None:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out


def run_baseline(task: Task, data) -> dict[str, Any]:
    model = task.make_baseline()
    start = time.perf_counter()
    model.fit(task.baseline_features(data.train), data.train["label"])
    fit_s = time.perf_counter() - start
    result = {"model": task.baseline_desc, "fit_seconds": round(fit_s, 2)}
    for part, frame in [("full_test", data.test), ("jev_subset", data.jev_test)]:
        X = task.baseline_features(frame)
        score = positive_scores(model, X) if task.kind == "binary" else None
        result[part] = metrics(task, frame["label"], model.predict(X), score)
    return result


def run_jev_eval(task: Task, data, model: str, concurrency: int) -> dict[str, Any]:
    records = asyncio.run(run_jev(task, data.jev_test, model, concurrency))
    ok = [(i, r) for i, r in enumerate(records) if "error" not in r]
    errors = [r["error"] for r in records if "error" in r]
    result: dict[str, Any] = {"model": model, "errors": len(errors)}
    if errors:
        result["error_examples"] = sorted(set(errors))[:3]
    if not ok:
        return result
    idx = [i for i, _ in ok]
    decoded = [task.decode(r["answer"]) for _, r in ok]
    y_true = data.jev_test["label"].iloc[idx]
    y_pred = [d[0] for d in decoded]
    y_score = [d[1] for d in decoded] if task.kind == "binary" else None
    result["jev_subset"] = metrics(task, y_true, y_pred, y_score)
    lat = [r["latency_s"] for _, r in ok if "latency_s" in r]
    if lat:
        result["latency_p50_s"] = round(float(np.median(lat)), 3)
        result["latency_p95_s"] = round(float(np.percentile(lat, 95)), 3)
    tokens = [r.get("usage", {}).get("input_tokens") for _, r in ok]
    tokens = [t for t in tokens if t is not None]
    if tokens:
        result["input_tokens_total"] = int(sum(tokens))
    if "confidence" in ok[0][1]["answer"]:
        conf = np.array([r["answer"]["confidence"] for _, r in ok])
        correct = np.array([p == t for p, t in zip(y_pred, y_true)])
        result["accuracy_when_confidence_ge_0.8"] = float(correct[conf >= 0.8].mean()) if (conf >= 0.8).any() else None
        result["share_confidence_ge_0.8"] = float((conf >= 0.8).mean())
    return result


def write_summary() -> None:
    """Summarise every task that has a results file, not only the tasks run this time."""
    all_results = {name: json.loads(p.read_text()) for name in TASKS if (p := RESULTS_DIR / f"{name}.json").exists()}
    lines = [
        "# Kaggle × Jev benchmark results",
        "",
        "Same stratified test subset for both methods (`jev_subset`). The baseline also reports the full 20% test split.",
        "",
        "| Task | Metric | Kaggle baseline (subset) | Jev zero-shot (subset) | Baseline (full test) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, res in all_results.items():
        task = TASKS[name]
        keys = ["accuracy", "macro_f1"] + (["roc_auc"] if task.kind == "binary" else [])
        for k in keys:
            b = res.get("baseline", {}).get("jev_subset", {}).get(k)
            j = res.get("jev", {}).get("jev_subset", {}).get(k)
            f = res.get("baseline", {}).get("full_test", {}).get(k)
            fmt = lambda v: "—" if v is None else f"{v:.4f}"
            lines.append(f"| {name} | {k} | {fmt(b)} | {fmt(j)} | {fmt(f)} |")
    lines.append("")
    for name, res in all_results.items():
        if "jev" in res:
            jev = {k: v for k, v in res["jev"].items() if k != "jev_subset"}
            lines.append(f"- **{name}** Jev details: `{json.dumps(jev, ensure_ascii=False)}`")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--mode", choices=["baseline", "jev", "both", "dry-run"], default="both")
    parser.add_argument("--jev-n", type=int, default=200, help="test rows per task sent to Jev (stratified); 0 = all")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    jev_n = None if args.jev_n == 0 else args.jev_n
    for name in args.tasks:
        task = TASKS[name]
        data = split(name, jev_n)
        print(f"\n=== {name}: train={len(data.train)} test={len(data.test)} jev_subset={len(data.jev_test)}")
        path = RESULTS_DIR / f"{name}.json"
        res: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
        res["kaggle"] = task.kaggle
        if args.mode == "dry-run":
            example = request_payload(task, data.jev_test.iloc[0], args.model)
            (RESULTS_DIR / f"{name}_request_example.json").write_text(json.dumps(example, indent=2, ensure_ascii=False))
            print(json.dumps(example, ensure_ascii=False)[:600])
            continue
        if args.mode in ("baseline", "both"):
            res["baseline"] = run_baseline(task, data)
            print("[baseline]", json.dumps(res["baseline"]["jev_subset"]))
        if args.mode in ("jev", "both"):
            res["jev"] = run_jev_eval(task, data, args.model, args.concurrency)
            print("[jev]", json.dumps(res["jev"].get("jev_subset"), ensure_ascii=False), "errors:", res["jev"]["errors"])
        path.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    if args.mode != "dry-run":
        write_summary()
        print(f"\nWrote {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
