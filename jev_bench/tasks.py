"""The five Kaggle tasks: a classic Kaggle baseline and a Jev (TypeSafe System One) formulation.

Jev is used zero-shot: it never sees training labels. Each task turns one row into ``state``
and asks one typed question (Noul for yes/no, Choice for multi-class). The answer is decoded
into a predicted label and, for binary tasks, the probability of the positive class.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC
from typesafe_sdk import Choice, Noul

# Long IMDB reviews are cut so a request stays small; most reviews are shorter than this.
MAX_TEXT_CHARS = 4000


@dataclass
class Task:
    name: str
    kaggle: str
    kind: str  # "binary" or "multiclass"
    labels: list[Any]
    make_baseline: Callable[[], Any]
    baseline_features: Callable[[pd.DataFrame], Any]
    baseline_desc: str
    to_state: Callable[[pd.Series], Any]
    questions: dict[str, Any]
    # Maps the answer for the "label" question to (predicted label, positive-class probability or None).
    decode: Callable[[dict[str, Any]], tuple[Any, float | None]]


def _decode_noul(answer: dict[str, Any]) -> tuple[int, float]:
    p = float(answer["noul"])
    return int(p >= 0.5), p


def _decode_choice(answer: dict[str, Any]) -> tuple[str, None]:
    return answer["choice"], None


# ---------------------------------------------------------------- Titanic


def _titanic_title(name: str) -> str:
    title = name.split(",")[1].split(".")[0].strip() if "," in name else ""
    return title or "Unknown"


def _titanic_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]].copy()
    out["Title"] = df["Name"].map(_titanic_title)
    out["HasCabin"] = df["Cabin"].notna().astype(int)
    out["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return out


def _titanic_baseline() -> Pipeline:
    cat = ["Sex", "Embarked", "Title"]
    pre = ColumnTransformer(
        [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), cat)],
        remainder="passthrough",
    )
    return make_pipeline(pre, HistGradientBoostingClassifier(random_state=42))


_EMBARKED = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}


def _nan_to_none(value: Any) -> Any:
    return None if pd.isna(value) else value


def _titanic_state(row: pd.Series) -> dict[str, Any]:
    # The full name is replaced by its title so the model judges the profile, not a memorised person.
    age = _nan_to_none(row["Age"])
    return {
        "passenger": {
            "ticket_class": {1: "1st (upper)", 2: "2nd (middle)", 3: "3rd (lower)"}[int(row["Pclass"])],
            "sex": row["Sex"],
            "title": _titanic_title(row["Name"]),
            "age_years": None if age is None else float(age),
            "siblings_or_spouses_aboard": int(row["SibSp"]),
            "parents_or_children_aboard": int(row["Parch"]),
            "fare_paid_gbp": round(float(row["Fare"]), 2),
            "has_recorded_cabin": bool(pd.notna(row["Cabin"])),
            "port_of_embarkation": _EMBARKED.get(row["Embarked"]) if pd.notna(row["Embarked"]) else None,
        }
    }


TITANIC = Task(
    name="titanic",
    kaggle="https://www.kaggle.com/competitions/titanic",
    kind="binary",
    labels=[0, 1],
    make_baseline=_titanic_baseline,
    baseline_features=_titanic_frame,
    baseline_desc="HistGradientBoosting + Title/FamilySize/HasCabin feature engineering",
    to_state=_titanic_state,
    questions={
        "label": Noul(
            instructions=(
                "`passenger` describes a person aboard the RMS Titanic when it sank on 15 April 1912. "
                "Based on this profile and what is known about who reached the lifeboats, did this passenger survive?"
            ),
            criteria={"true": "The passenger survived the sinking.", "false": "The passenger died in the sinking."},
        )
    },
    decode=_decode_noul,
)

# ---------------------------------------------------------------- SMS spam


def _text(df: pd.DataFrame) -> pd.Series:
    return df["text"]


SMS_SPAM = Task(
    name="sms_spam",
    kaggle="https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset",
    kind="binary",
    labels=[0, 1],
    make_baseline=lambda: make_pipeline(TfidfVectorizer(sublinear_tf=True), MultinomialNB(alpha=0.1)),
    baseline_features=_text,
    baseline_desc="TF-IDF + Multinomial Naive Bayes",
    to_state=lambda row: {"sms": row["text"]},
    questions={
        "label": Noul(
            instructions="Is `sms` a spam text message?",
            criteria={
                "true": "Unsolicited bulk or commercial message: prize or lottery claims, premium-rate numbers, "
                "marketing offers, subscription services, or scams.",
                "false": "An ordinary personal or transactional message between people who know each other, "
                "including informal chat with slang or abbreviations.",
            },
        )
    },
    decode=_decode_noul,
)

# ---------------------------------------------------------------- IMDB 50k


IMDB = Task(
    name="imdb",
    kaggle="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
    kind="binary",
    labels=[0, 1],
    make_baseline=lambda: make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True),
        LogisticRegression(C=4.0, max_iter=2000),
    ),
    baseline_features=_text,
    baseline_desc="TF-IDF (1-2 gram) + Logistic Regression",
    to_state=lambda row: {"review": row["text"][:MAX_TEXT_CHARS]},
    questions={
        "label": Noul(
            instructions="Is `review` a positive review overall, meaning the reviewer recommends the movie?",
            criteria={
                "true": "Overall positive: the reviewer liked the movie, even if they mention some flaws.",
                "false": "Overall negative: the reviewer disliked the movie, even if they mention some good points.",
            },
        )
    },
    decode=_decode_noul,
)

# ---------------------------------------------------------------- BBC News

BBC_LABELS = ["business", "entertainment", "politics", "sport", "tech"]

BBC_NEWS = Task(
    name="bbc_news",
    kaggle="https://www.kaggle.com/competitions/learn-ai-bbc",
    kind="multiclass",
    labels=BBC_LABELS,
    make_baseline=lambda: make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2), LinearSVC(C=1.0)),
    baseline_features=_text,
    baseline_desc="TF-IDF + Linear SVM",
    to_state=lambda row: {"article": row["text"][:MAX_TEXT_CHARS]},
    questions={
        "label": Choice(
            instructions="Which BBC News section does `article` belong to?",
            criteria={
                "business": "Companies, markets, economy, trade, finance, jobs and prices.",
                "entertainment": "Film, music, television, theatre, celebrities, awards and the arts.",
                "politics": "Government, parliament, parties, elections, ministers and public policy.",
                "sport": "Sports competitions, athletes, teams, matches and results.",
                "tech": "Technology, computing, the internet, gadgets, games, software and telecoms.",
            },
        )
    },
    decode=_decode_choice,
)

# ---------------------------------------------------------------- Iris

IRIS_FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

IRIS = Task(
    name="iris",
    kaggle="https://www.kaggle.com/datasets/uciml/iris",
    kind="multiclass",
    labels=["setosa", "versicolor", "virginica"],
    make_baseline=lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    baseline_features=lambda df: df[IRIS_FEATURES],
    baseline_desc="StandardScaler + Logistic Regression",
    to_state=lambda row: {"flower_measurements_cm": {f: float(row[f]) for f in IRIS_FEATURES}},
    questions={
        "label": Choice(
            instructions="`flower_measurements_cm` are measurements of one iris flower. Which species is it?",
            # No measurement ranges here: hand-written thresholds would be a rule written by us, not Jev's judgment.
            criteria={"setosa": "Iris setosa", "versicolor": "Iris versicolor", "virginica": "Iris virginica"},
        )
    },
    decode=_decode_choice,
)

TASKS: dict[str, Task] = {t.name: t for t in [TITANIC, SMS_SPAM, IMDB, BBC_NEWS, IRIS]}


def positive_scores(model: Any, X: Any) -> np.ndarray | None:
    """Positive-class scores for ROC AUC, if the baseline exposes them."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None
