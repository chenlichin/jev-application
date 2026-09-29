"""Compare classic Kaggle baselines with zero-shot Jev inference on five Kaggle problems.

Usage:
    python run_benchmark.py --mode baseline               # no API key needed
    python run_benchmark.py --mode dry-run                # write example Jev requests, no API calls
    TYPESAFE_API_KEY=... python run_benchmark.py --mode both --jev-n 200
    TYPESAFE_API_KEY=... python run_benchmark.py --mode jev --shots 0 3   # zero-shot and 3 examples per class
    TYPESAFE_API_KEY=... python run_benchmark.py --mode jev --shots cluster cluster-random

Shot variants: an integer k draws k random training examples per class; "cluster" takes one
example per (label, k-means cluster) cell; "cluster-random" is its control, with the same
per-label counts drawn at random.
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
from jev_bench.cluster_shots import select_cluster_shots, select_matched_random
from jev_bench.few_shot import few_shot_questions, select_shots
from jev_bench.jev_runner import request_payload, run_jev
from jev_bench.tasks import TASKS, Task, positive_scores

RESULTS_DIR = ROOT / "results"


CLUSTER_VARIANTS = {"cluster": ("jev_cluster", "Jev cluster-shot"), "cluster-random": ("jev_cluster_random", "Jev matched random")}


def jev_key(variant: str) -> str:
    """Results key for a Jev run; zero-shot keeps the original "jev" key."""
    if variant in CLUSTER_VARIANTS:
        return CLUSTER_VARIANTS[variant][0]
    return "jev" if int(variant) == 0 else f"jev_{int(variant)}shot"


def jev_methods(all_results: dict[str, dict[str, Any]]) -> list[tuple[str, str]]:
    """(results key, column label) for every Jev variant present in any task, in a fixed order."""
    present = {k for res in all_results.values() for k in res if k.startswith("jev")}
    shot_keys = sorted((k for k in present if k == "jev" or k.endswith("shot") and k[4:-4].isdigit()),
                       key=lambda k: 0 if k == "jev" else int(k[4:-4]))
    methods = [(k, "Jev 0-shot" if k == "jev" else f"Jev {k[4:-4]}-shot") for k in shot_keys]
    return methods + [(key, label) for key, label in CLUSTER_VARIANTS.values() if key in present]


class ShotPlanner:
    """Builds each variant's examples for one task; the cluster selection is computed once and reused."""

    def __init__(self, task: Task, data) -> None:
        self.task, self.data = task, data
        self._cluster: tuple[Any, dict[str, Any]] | None = None

    def cluster(self) -> tuple[Any, dict[str, Any]]:
        if self._cluster is None:
            self._cluster = select_cluster_shots(self.task, self.data.train)
        return self._cluster

    def shots(self, variant: str) -> tuple[Any | None, dict[str, Any]]:
        """(training rows used as examples or None for zero-shot, extra info for the results file)."""
        if variant == "cluster":
            rows, info = self.cluster()
            return rows, {"clustering": info}
        if variant == "cluster-random":
            return select_matched_random(self.task, self.data.train, self.cluster()[0]), {}
        k = int(variant)
        return (select_shots(self.task, self.data.train, k) if k else None), {}


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


def run_jev_eval(task: Task, data, model: str, concurrency: int, variant: str, planner: ShotPlanner) -> dict[str, Any]:
    shot_rows, extra = planner.shots(variant)
    questions = task.questions if shot_rows is None else few_shot_questions(task, shot_rows)
    records = asyncio.run(run_jev(task, data.jev_test, model, concurrency, questions))
    ok = [(i, r) for i, r in enumerate(records) if "error" not in r]
    errors = [r["error"] for r in records if "error" in r]
    result: dict[str, Any] = {"model": model, "shots": variant, "errors": len(errors)}
    if shot_rows is not None:
        result["examples_per_label"] = {str(k): int(v) for k, v in shot_rows["label"].value_counts().sort_index().items()}
        # Training-row indices, so the exact examples can be reproduced and checked against the test set.
        result["shot_train_rows"] = [int(i) for i in shot_rows.index]
    result.update(extra)
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


def _underline(text: str) -> str:
    """Underline with a combining low line (U+0332) after each character.

    Markdown has no underline syntax and some viewers do not render HTML such as <ins>,
    so the underline is carried by the text itself.
    """
    return "".join(ch + "\u0332" for ch in text)


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(_cell(c) for c in cells) + " |"


def _any_subset(all_results: dict[str, dict[str, Any]]) -> bool:
    """Whether any task was scored on a sample of its test split rather than all of it."""
    return any(
        (p := res.get("pipeline")) is not None and p["n_jev_subset"] < p["n_full_test"] for res in all_results.values()
    )


def overview_table(all_results: dict[str, dict[str, Any]]) -> list[str]:
    """Per task: dataset, size, problem, the Jev question, and the first state sent to Jev."""
    rows = [
        _row(["Dataset", "n", "Train", "Problem", "Jev question", "State (first row)", "True label"]),
        "| --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    for name, res in all_results.items():
        p = res.get("pipeline")
        if p is None:
            continue
        # Backticks would end the code span early, so they are dropped from the example state.
        state = _truncate(json.dumps(p["state_example"], ensure_ascii=False).replace("`", ""), 110)
        question = f"**{p['question_type']}**: {p['question']} ({p['answers']})"
        rows.append(
            _row([f"[{name}]({res['kaggle']})", str(p["n_jev_subset"]), str(p["n_train"]), p["problem"], question,
                  f"`{state}`", p["state_example_label"]])
        )
    return rows


def metric_table(all_results: dict[str, dict[str, Any]], metric: str) -> list[str]:
    """One metric across every method; the best score on the shared subset is bold."""
    methods = jev_methods(all_results)
    # Once every task is scored on its whole test split, the full-test column would repeat "Kaggle".
    show_full = _any_subset(all_results)
    header = ["Dataset", "n", "Kaggle", *(label for _, label in methods)] + (["Kaggle full test"] if show_full else [])
    rows = [_row(header), "| --- | " + " | ".join(["---:"] * (len(header) - 1)) + " |"]
    for name, res in all_results.items():
        p = res.get("pipeline")
        base = res.get("baseline", {}).get("jev_subset", {}).get(metric)
        if p is None or base is None:  # ROC AUC exists only for the binary tasks
            continue
        scores = [base, *(res.get(key, {}).get("jev_subset", {}).get(metric) for key, _ in methods)]
        # Compared at the displayed precision, so scores that print the same are marked the same.
        best = max(round(v, 3) for v in scores if v is not None)
        cells = ["—" if v is None else f"**{_underline(f'{v:.3f}')}**" if round(v, 3) == best else f"{v:.3f}" for v in scores]
        full = res.get("baseline", {}).get("full_test", {}).get(metric)
        extra = ["—" if full is None else f"{full:.3f}"] if show_full else []
        rows.append(_row([name, str(p["n_jev_subset"]), *cells, *extra]))
    return rows


def write_summary() -> None:
    """Summarise every task that has a results file, not only the tasks run this time."""
    all_results = {name: json.loads(p.read_text()) for name in TASKS if (p := RESULTS_DIR / f"{name}.json").exists()}
    methods = jev_methods(all_results)
    if _any_subset(all_results):
        scope = ("`n` is the stratified test subset every method is scored on; \"Kaggle full test\" scores the same "
                 "Kaggle model on the whole 20% test split, to show whether the subset is representative.")
    else:
        scope = "Every method is scored on the whole 20% test split (`n` rows), never seen in training."
    lines = [
        "# Kaggle × Jev benchmark results",
        "",
        scope,
        "Kaggle is the classic Kaggle solution trained on the 80% train split. Jev 0-shot never sees training labels;",
        "Jev k-shot adds k random labeled training examples per class to each answer's criteria; Jev cluster-shot takes",
        "one example per (label, k-means cluster) cell; Jev matched random is its control with the same per-label counts.",
        "Examples always come from the train split. Bold and underline mark the best score on the shared subset.",
        "",
        "## 1. Summary",
        "",
        *overview_table(all_results),
        "",
        "## 2. Accuracy",
        "",
        *metric_table(all_results, "accuracy"),
        "",
        "## 3. Macro-F1",
        "",
        *metric_table(all_results, "macro_f1"),
        "",
        "## 4. ROC AUC (binary tasks)",
        "",
        "Uses the Kaggle model's positive-class score and Jev's `noul` probability.",
        "",
        *metric_table(all_results, "roc_auc"),
        "",
        "## Run details",
        "",
    ]
    for name, res in all_results.items():
        for key, label in methods:
            if key in res:
                details = {k: v for k, v in res[key].items() if k not in ("jev_subset", "shot_train_rows", "clustering")}
                lines.append(f"- **{name}** {label}: `{json.dumps(details, ensure_ascii=False)}`")
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    parser.add_argument("--mode", choices=["baseline", "jev", "both", "dry-run"], default="both")
    parser.add_argument("--jev-n", type=int, default=200, help="test rows per task sent to Jev (stratified); 0 = all")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--shots", nargs="+", default=["0"], help="shot variants: integers (examples per class), cluster, cluster-random"
    )
    args = parser.parse_args()
    for variant in args.shots:
        if variant not in CLUSTER_VARIANTS and not variant.isdigit():
            parser.error(f"--shots: unknown variant {variant!r}")

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
        planner = ShotPlanner(task, data)
        if args.mode == "dry-run":
            for variant in args.shots:
                shot_rows, _ = planner.shots(variant)
                questions = None if shot_rows is None else few_shot_questions(task, shot_rows)
                example = request_payload(task, data.jev_test.iloc[0], args.model, questions)
                suffix = jev_key(variant)[3:]
                out = RESULTS_DIR / f"{name}_request_example{suffix}.json"
                out.write_text(json.dumps(example, indent=2, ensure_ascii=False))
                print(f"[dry-run] wrote {out.name} ({len(json.dumps(example))} chars)")
            continue
        if args.mode in ("baseline", "both"):
            res["baseline"] = run_baseline(task, data)
            print("[baseline]", json.dumps(res["baseline"]["jev_subset"]))
        if args.mode in ("jev", "both"):
            for variant in args.shots:
                key = jev_key(variant)
                res[key] = run_jev_eval(task, data, args.model, args.concurrency, variant, planner)
                print(f"[{key}]", json.dumps(res[key].get("jev_subset"), ensure_ascii=False), "errors:", res[key]["errors"])
        path.write_text(json.dumps(res, indent=2, ensure_ascii=False))

    if args.mode != "dry-run":
        write_summary()
        print(f"\nWrote {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
