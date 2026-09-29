"""Cluster-stratified few-shot selection.

Random k-per-class sampling mostly draws the typical member of each class. Here the training
rows are first grouped with k-means (the number of clusters is chosen by silhouette score), then
every (label, cluster) cell with enough rows contributes one representative example.
The cluster count is at least the number of labels. A label
that spreads over several clusters therefore gets one example per sub-pattern, including the
minority cells, such as a first-class woman who died on the Titanic, that sit near the label
boundary and that random sampling rarely draws.

``select_matched_random`` draws the same number of examples per label at random, as a control
that separates the effect of diversity from the effect of simply showing more examples.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler

from jev_bench.datasets import SEED
from jev_bench.few_shot import MAX_EXAMPLE_CHARS, example_len
from jev_bench.tasks import Task

# Cluster counts tried, starting at the number of labels: with fewer clusters than labels the
# clusters cannot separate the labels (Iris scores best at k=2, setosa vs. the rest).
MAX_K = 8
# A cell needs at least this many rows to count as a pattern rather than a single outlier.
MIN_CELL_SIZE = 3
SILHOUETTE_SAMPLE = 3000


def cluster_features(task: Task, train: pd.DataFrame) -> np.ndarray:
    """Numeric vectors for k-means, built from the same inputs the Kaggle baseline uses."""
    X = task.baseline_features(train)
    if isinstance(X, pd.Series):  # text: TF-IDF reduced to a dense, length-normalised space
        pipe = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50_000, stop_words="english"),
            TruncatedSVD(n_components=100, random_state=SEED),
            Normalizer(),
        )
        return pipe.fit_transform(X)
    pre = ColumnTransformer(
        [
            ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), make_column_selector(dtype_include="number")),
            (
                "cat",
                make_pipeline(
                    SimpleImputer(strategy="most_frequent"),
                    OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False),
                ),
                make_column_selector(dtype_exclude="number"),
            ),
        ]
    )
    return pre.fit_transform(X)


def choose_k(X: np.ndarray, min_k: int) -> tuple[int, dict[int, float]]:
    """The cluster count in ``min_k..MAX_K`` with the best silhouette score."""
    scores = {}
    for k in range(max(2, min_k), max(MAX_K, min_k) + 1):
        labels = KMeans(n_clusters=k, n_init=4, random_state=SEED).fit_predict(X)
        sample = min(SILHOUETTE_SAMPLE, len(X))
        scores[k] = float(silhouette_score(X, labels, sample_size=sample, random_state=SEED))
    return max(scores, key=scores.get), scores


def select_cluster_shots(task: Task, train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One example per (label, k-means cluster) cell, plus a description of the clustering."""
    X = cluster_features(task, train)
    k, scores = choose_k(X, len(task.labels))
    clusters = KMeans(n_clusters=k, n_init=4, random_state=SEED).fit_predict(X)
    labels = train["label"].to_numpy()
    short = np.array([example_len(task, row) <= MAX_EXAMPLE_CHARS for _, row in train.iterrows()])

    picked, cells = [], []
    for label in task.labels:
        for c in range(k):
            in_cell = np.flatnonzero((labels == label) & (clusters == c))
            if len(in_cell) < MIN_CELL_SIZE:
                continue
            # Representative = row nearest the cell centroid, among short rows when the cell has any.
            candidates = in_cell[short[in_cell]] if short[in_cell].any() else in_cell
            centroid = X[in_cell].mean(axis=0)
            best = candidates[np.argmin(np.linalg.norm(X[candidates] - centroid, axis=1))]
            picked.append(best)
            cluster_size = int((clusters == c).sum())
            cells.append(
                {
                    "label": str(label),
                    "cluster": c,
                    "cell_size": len(in_cell),
                    "cluster_size": cluster_size,
                    # Below 0.5 the label is the minority of its cluster: a boundary pattern.
                    "label_share_of_cluster": round(len(in_cell) / cluster_size, 3),
                    "train_row": int(train.index[best]),
                }
            )
    info = {
        "k": k,
        "silhouette": {str(kk): round(v, 4) for kk, v in scores.items()},
        "min_cell_size": MIN_CELL_SIZE,
        "cells": cells,
    }
    return train.iloc[picked], info


def select_matched_random(task: Task, train: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Random examples with the same per-label counts as ``reference`` (the cluster selection)."""
    counts = reference["label"].value_counts()
    picked = []
    for label in task.labels:
        n = int(counts.get(label, 0))
        if n:
            rows = train[train["label"] == label]
            short = rows[[example_len(task, r) <= MAX_EXAMPLE_CHARS for _, r in rows.iterrows()]]
            pool = short if len(short) >= n else rows
            picked.append(pool.sample(n=n, random_state=SEED))
    return pd.concat(picked)
