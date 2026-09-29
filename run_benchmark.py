"""Compare classic Kaggle baselines with zero-shot Jev inference on five Kaggle problems.

Usage:
    python run_benchmark.py --mode baseline               # no API key needed
    python run_benchmark.py --mode dry-run                # write example Jev requests, no API calls
    TYPESAFE_API_KEY=... python run_benchmark.py --mode both --jev-n 200
    TYPESAFE_API_KEY=... python run_benchmark.py --mode jev --shots 0 3   # zero-shot and 3 examples per class
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
from jev_bench.few_shot import few_shot_questions, select_shots
from jev_bench.jev_runner import request_payload, run_jev
from jev_bench.tasks import TASKS, Task, positive_scores

RESULTS_DIR = ROOT / "results"


def jev_key(shots: int) -> str:
    """Results key for a Jev run; zero-shot keeps the original "jev" key."""
    return "jev" if shots == 0 else f"jev_{shots}shot"


def jev_methods(all_results: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """(results key, column label) for every Jev variant present in any task, zero-shot first."""
    shots = sorted({0 if k == "jev" else int(k[4:-4]) for res in all_results.values() for k in res if k.startswith("jev")})
    return [(jev_key(k), "Jev 0-shot" if k == 0 else f"Jev {k}-shot") for k in shots]


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


def run_jev_eval(task: Task, data, model: str, concurrency: int, shots: int) -> dict[str, Any]:
    questions = task.questions
    shot_rows = None
    if shots:
        shot_rows = select_shots(task, data.train, shots)
        questions = few_shot_questions(task, shot_rows)
    records = asyncio.run(run_jev(task, data.jev_test, model, concurrency, questions))
    ok = [(i, r) for i, r in enumerate(records) if "error" not in r]
    errors = [r["error"] for r in records if "error" in r]
    result: dict[str, Any] = {"model": model, "shots_per_class": shots, "errors": len(errors)}
    if shot_rows is not None:
        # Training-row indices, so the exact examples can be reproduced and checked against the test set.
        result["shot_train_rows"] = [int(i) for i in shot_rows.index]
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


def describe_pipeline(task: Task, data) -> dict[str, Any]:
    """What the summary table shows about a task: its data, the Jev question, and one example state."""
    question = task.questions["label"]
    if question.type == "noul":
        answers = "yes / no"
    else:
        answers = " / ".join(question.criteria)
    return {
        "problem": task.problem,
        "n_jev_subset": len(data.jev_test),
        "n_full_test": len(data.test),
        "n_train": len(data.train),
        "question_type": question.type.capitalize(),
        "question": question.instructions,
        "answers": answers,
        "state_example": task.to_state(data.jev_test.iloc[0]),
        "state_example_label": str(data.jev_test["label"].iloc[0]),
    }


def _truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _cell(text: str) -> str:
    """Make text safe inside one Markdown table cell."""
    return " ".join(text.split()).replace("|", "\\|")


def pipeline_table(all_results: dict[str, dict[str, Any]]) -> list[str]:
    """Plain Markdown table; each metric column is prefixed with its group ("Accuracy · Kaggle")."""
    fmt = lambda v: "—" if v is None else f"{v:.3f}"
    methods = jev_methods(all_results)
    acc_cols = ["Kaggle", *(label for _, label in methods), "Kaggle full test"]
    f1_cols = ["Kaggle", *(label for _, label in methods)]
    header = ["Dataset", "n", "Problem", "Jev question", "State (first row)", "True label"]
    header += [f"Accuracy · {c}" for c in acc_cols] + [f"Macro-F1 · {c}" for c in f1_cols]
    rows = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * 6 + ["---:"] * (len(header) - 6)) + " |"]
    for name, res in all_results.items():
        p = res.get("pipeline")
        if p is None:
            continue
        base, full = res.get("baseline", {}).get("jev_subset", {}), res.get("baseline", {}).get("full_test", {})
        jevs = [res.get(key, {}).get("jev_subset", {}) for key, _ in methods]
        # Backticks would end the code span early, so they are dropped from the example state.
        state = _truncate(json.dumps(p["state_example"], ensure_ascii=False).replace("`", ""), 110)
        question = f"**{p['question_type']}**: {p['question']} ({p['answers']})"
        cells = [f"[{name}]({res['kaggle']})", str(p["n_jev_subset"]), p["problem"], question, f"`{state}`", p["state_example_label"]]
        cells += [fmt(base.get("accuracy")), *(fmt(j.get("accuracy")) for j in jevs), fmt(full.get("accuracy"))]
        cells += [fmt(base.get("macro_f1")), *(fmt(j.get("macro_f1")) for j in jevs)]
        rows.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    return rows


def write_summary() -> None:
    """Summarise every task that has a results file, not only the tasks run this time."""
    all_results = {name: json.loads(p.read_text()) for name in TASKS if (p := RESULTS_DIR / f"{name}.json").exists()}
    methods = jev_methods(all_results)
    lines = [
        "# Kaggle × Jev benchmark results",
        "",
        "## Pipeline overview",
        "",
        "`n` is the stratified test subset both methods are scored on; \"Kaggle full test\" is the whole 20% test split.",
        "Kaggle columns are the classic Kaggle solution trained on the 80% train split. Jev 0-shot never sees training",
        "labels; Jev k-shot adds k labeled training examples per class to each answer's criteria (never test rows).",
        "Row counts for every split are in `results/<task>.json` under `pipeline`.",
        "",
        *pipeline_table(all_results),
        "",
        "## All metrics",
        "",
        "Same stratified test subset for every method (`jev_subset`). The baseline also reports the full 20% test split.",
        "",
        "| Task | Metric | Kaggle baseline (subset) | "
        + " | ".join(f"{label} (subset)" for _, label in methods)
        + " | Baseline (full test) |",
        "| --- | --- | --- | " + " | ".join("---" for _ in methods) + " | --- |",
    ]
    fmt = lambda v: "—" if v is None else f"{v:.4f}"
    for name, res in all_results.items():
        task = TASKS[name]
        keys = ["accuracy", "macro_f1"] + (["roc_auc"] if task.kind == "binary" else [])
        for k in keys:
            b = res.get("baseline", {}).get("jev_subset", {}).get(k)
            js = [res.get(key, {}).get("jev_subset", {}).get(k) for key, _ in methods]
            f = res.get("baseline", {}).get("full_test", {}).get(k)
            lines.append(f"| {name} | {k} | {fmt(b)} | " + " | ".join(fmt(j) for j in js) + f" | {fmt(f)} |")
    lines.append("")
    for name, res in all_results.items():
        for key, label in methods:
            if key in res:
                details = {k: v for k, v in res[key].items() if k not in ("jev_subset", "shot_train_rows")}
                lines.append(f"- **{name}** {label} details: `{json.dumps(details, ensure_ascii=False)}`")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--mode", choices=["baseline", "jev", "both", "dry-run"], default="both")
    parser.add_argument("--jev-n", type=int, default=200, help="test rows per task sent to Jev (stratified); 0 = all")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--shots", type=int, nargs="+", default=[0], help="labeled examples per class; e.g. --shots 0 3")
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
        res["pipeline"] = describe_pipeline(task, data)
        if args.mode == "dry-run":
            for shots in args.shots:
                questions = few_shot_questions(task, select_shots(task, data.train, shots)) if shots else None
                example = request_payload(task, data.jev_test.iloc[0], args.model, questions)
                suffix = "" if shots == 0 else f"_{shots}shot"
                out = RESULTS_DIR / f"{name}_request_example{suffix}.json"
                out.write_text(json.dumps(example, indent=2, ensure_ascii=False))
                print(f"[dry-run] wrote {out.name} ({len(json.dumps(example))} chars)")
            continue
        if args.mode in ("baseline", "both"):
            res["baseline"] = run_baseline(task, data)
            print("[baseline]", json.dumps(res["baseline"]["jev_subset"]))
        if args.mode in ("jev", "both"):
            for shots in args.shots:
                key = jev_key(shots)
                res[key] = run_jev_eval(task, data, args.model, args.concurrency, shots)
                print(f"[{key}]", json.dumps(res[key].get("jev_subset"), ensure_ascii=False), "errors:", res[key]["errors"])
        path.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    if args.mode != "dry-run":
        write_summary()
        print(f"\nWrote {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
