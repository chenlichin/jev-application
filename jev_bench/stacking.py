"""Stack Jev's probabilities with the Kaggle model's scores.

Jev's answer becomes a feature next to the Kaggle model's own score, in a second-stage model::

    stage 1  Kaggle model  -> out-of-fold scores on train (5-fold), scores on test (fit on all train)
    Jev      same question -> probabilities on a stratified train sample and on the scored test rows
    stage 2  logistic regression on [Kaggle score, Jev probability], fit on the train sample

Out-of-fold scores keep stage 2 honest: each training row's Kaggle score comes from a model that
did not see that row, just as the test scores come from a model that did not see the test rows.
Jev is only called on a sample of the training rows (``STACK_TRAIN_N``), since stage 2 has few
inputs and does not need 40,000 IMDB rows to fit. Rows used as few-shot examples are excluded
from that sample, so Jev never scores a row it was shown with its label.
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from jev_bench.datasets import SEED
from jev_bench.few_shot import few_shot_questions
from jev_bench.jev_runner import run_jev
from jev_bench.tasks import Task

STACK_TRAIN_N = 2000
EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _baseline_scores(task: Task, model: Any, X: Any) -> np.ndarray:
    """Stage-1 scores as a 2-D array: positive-class logit for binary, one column per label otherwise."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return _logit(proba[:, 1:2]) if task.kind == "binary" else _logit(proba)
    scores = model.decision_function(X)
    return scores.reshape(-1, 1) if scores.ndim == 1 else scores


def _oof_scores(task: Task, train: pd.DataFrame) -> np.ndarray:
    method = "predict_proba" if hasattr(task.make_baseline(), "predict_proba") else "decision_function"
    raw = cross_val_predict(
        task.make_baseline(),
        task.baseline_features(train),
        train["label"],
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        method=method,
    )
    if method == "predict_proba":
        return _logit(raw[:, 1:2]) if task.kind == "binary" else _logit(raw)
    return raw.reshape(-1, 1) if raw.ndim == 1 else raw


def _jev_features(task: Task, records: list[dict[str, Any]]) -> tuple[np.ndarray, int]:
    """Jev probabilities as logits (one column for Noul, one per label for Choice); failures get the prior."""
    rows, missing = [], 0
    for r in records:
        if "error" in r:
            missing += 1
            rows.append([0.5] if task.kind == "binary" else [1 / len(task.labels)] * len(task.labels))
        elif task.kind == "binary":
            rows.append([r["answer"]["noul"]])
        else:
            rows.append([r["answer"]["probabilities"].get(str(label), 0.0) for label in task.labels])
    return _logit(np.array(rows, dtype=float)), missing


def run_stacking(
    task: Task, data, model: str, concurrency: int, shot_rows: pd.DataFrame | None
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray | None]]]:
    """Fit stage 2 and return (info, predictions).

    ``predictions`` maps "stacked" and the control "kaggle_only" (the same stage 2 fit on the Kaggle
    score alone, so any change from re-fitting the decision threshold is not credited to Jev) to
    (test predictions, positive-class test scores or None).
    """
    questions = task.questions if shot_rows is None else few_shot_questions(task, shot_rows)

    oof = _oof_scores(task, data.train)
    eligible = data.train.index if shot_rows is None else data.train.index.difference(shot_rows.index)
    if len(eligible) > STACK_TRAIN_N:
        sample_idx, _ = train_test_split(
            eligible, train_size=STACK_TRAIN_N, random_state=SEED, stratify=data.train.loc[eligible, "label"]
        )
    else:
        sample_idx = eligible
    sample = data.train.loc[sample_idx]
    positions = data.train.index.get_indexer(sample_idx)

    train_records = asyncio.run(run_jev(task, sample, model, concurrency, questions))
    test_records = asyncio.run(run_jev(task, data.jev_test, model, concurrency, questions))
    jev_train, miss_train = _jev_features(task, train_records)
    jev_test, miss_test = _jev_features(task, test_records)

    base = task.make_baseline().fit(task.baseline_features(data.train), data.train["label"])
    base_test = _baseline_scores(task, base, task.baseline_features(data.jev_test))

    def fit_predict(X_train: np.ndarray, X_test: np.ndarray) -> tuple[Any, tuple[np.ndarray, np.ndarray | None]]:
        meta = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X_train, sample["label"])
        score = meta.predict_proba(X_test)[:, 1] if task.kind == "binary" else None
        return meta, (meta.predict(X_test), score)

    meta, stacked = fit_predict(np.hstack([oof[positions], jev_train]), np.hstack([base_test, jev_test]))
    _, kaggle_only = fit_predict(oof[positions], base_test)

    info: dict[str, Any] = {"stage2_train_rows": len(sample), "jev_missing": miss_train + miss_test}
    if task.kind == "binary":
        # Standardised coefficients: how much stage 2 leans on each input.
        coef = meta[-1].coef_[0]
        info["stage2_weight"] = {"kaggle": round(float(coef[0]), 3), "jev": round(float(coef[1]), 3)}
    return info, {"stacked": stacked, "kaggle_only": kaggle_only}
