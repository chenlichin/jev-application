"""Few-shot variants of each task's Jev question.

Labeled examples are drawn from the training split only (never the test rows) and attached
to the criteria of the answer they illustrate, so each outcome is defined by a description
plus concrete cases::

    criteria = {"true":  {"description": "...", "examples": [state, state, ...]},
                "false": {"description": "...", "examples": [...]}}

The instructions stay unchanged, so a k-shot run differs from the zero-shot run only by the
examples. The same examples are used for every test row, which keeps results comparable.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from typesafe_sdk import Choice, Noul

from jev_bench.datasets import SEED
from jev_bench.tasks import Task

# Examples longer than this (as JSON) are skipped when shorter ones exist, so a few long
# reviews or articles do not multiply the request size.
MAX_EXAMPLE_CHARS = 700
# Text fields of an example are cut to this length when no short example exists (BBC articles
# are almost all longer); the opening of an article or review carries most of its label signal.
MAX_EXAMPLE_TEXT = 600


def example_len(task: Task, row: pd.Series) -> int:
    return len(json.dumps(task.to_state(row), ensure_ascii=False))


def select_shots(task: Task, train: pd.DataFrame, k: int) -> pd.DataFrame:
    """``k`` training rows per label, preferring examples short enough to keep requests small."""
    picked = []
    for label in task.labels:
        rows = train[train["label"] == label]
        short = rows[[example_len(task, r) <= MAX_EXAMPLE_CHARS for _, r in rows.iterrows()]]
        pool = short if len(short) >= k else rows
        picked.append(pool.sample(n=min(k, len(pool)), random_state=SEED))
    return pd.concat(picked)


def _shorten(value: Any) -> Any:
    """Cut every string inside an example state to ``MAX_EXAMPLE_TEXT`` characters."""
    if isinstance(value, str):
        return value if len(value) <= MAX_EXAMPLE_TEXT else value[:MAX_EXAMPLE_TEXT].rstrip() + " …"
    if isinstance(value, dict):
        return {k: _shorten(v) for k, v in value.items()}
    return value


def _criterion_key(task: Task, label: Any) -> str:
    """Criteria key that describes ``label``: Noul uses "true"/"false", Choice uses the label itself."""
    if task.questions["label"].type == "noul":
        return "true" if label == 1 else "false"
    return str(label)


def few_shot_questions(task: Task, shots: pd.DataFrame) -> dict[str, Any]:
    """The task's questions with ``shots`` attached as examples to each outcome's criteria."""
    question = task.questions["label"]
    examples: dict[str, list[Any]] = {}
    for _, row in shots.iterrows():
        examples.setdefault(_criterion_key(task, row["label"]), []).append(_shorten(task.to_state(row)))
    criteria = {
        key: {"description": description, "examples": examples.get(key, [])}
        for key, description in dict(question.criteria).items()
    }
    cls = Noul if question.type == "noul" else Choice
    return {"label": cls(instructions=question.instructions, criteria=criteria)}
