"""Run Jev over a task's test rows with bounded concurrency and an on-disk answer cache.

Answers are cached in ``.cache/jev/<task>.jsonl`` keyed by a hash of (model, state, questions),
so re-running the benchmark or changing only the metrics costs no extra API calls.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from typesafe_sdk import AsyncTypeSafeClient, RetryPolicy, TypeSafeError

from jev_bench.datasets import ROOT
from jev_bench.tasks import Task

CACHE_DIR = ROOT / ".cache" / "jev"


def _wire_questions(task: Task) -> dict[str, Any]:
    return {name: q.model_dump(mode="json") for name, q in task.questions.items()}


def request_payload(task: Task, row: pd.Series, model: str) -> dict[str, Any]:
    return {"model": model, "state": task.to_state(row), "questions": _wire_questions(task)}


def _key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                cache[record["key"]] = record
    return cache


async def run_jev(task: Task, rows: pd.DataFrame, model: str, concurrency: int) -> list[dict[str, Any]]:
    """Return one record per row: ``{"answer", "latency_s", "usage"}`` or ``{"error"}``."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{task.name}.jsonl"
    cache = _load_cache(cache_path)
    payloads = [request_payload(task, row, model) for _, row in rows.iterrows()]
    keys = [_key(p) for p in payloads]
    todo = [i for i, k in enumerate(keys) if k not in cache]
    print(f"[jev] {task.name}: {len(rows) - len(todo)} cached, {len(todo)} to request (model={model})")

    if todo:
        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        done = 0
        async with AsyncTypeSafeClient(model=model, timeout=60.0, retry=RetryPolicy(max_retries=4)) as client:
            with cache_path.open("a") as fh:

                async def one(i: int) -> None:
                    nonlocal done
                    p = payloads[i]
                    async with sem:
                        start = time.perf_counter()
                        try:
                            resp = await client.system_one(p["state"], task.questions)
                            record = {
                                "key": keys[i],
                                "answer": resp.answers["label"].model_dump(mode="json"),
                                "model": resp.model,
                                "latency_s": time.perf_counter() - start,
                                "usage": resp.usage.model_dump(mode="json"),
                            }
                        except TypeSafeError as err:  # recorded, not cached, so a rerun retries it
                            record = {"key": keys[i], "error": f"{type(err).__name__}: {err}"}
                    async with lock:
                        if "error" not in record:
                            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                            fh.flush()
                        cache[keys[i]] = record
                        done += 1
                        if done % 25 == 0 or done == len(todo):
                            print(f"[jev] {task.name}: {done}/{len(todo)}")

                await asyncio.gather(*(one(i) for i in todo))

    return [cache[k] for k in keys]
