"""Download, load, and split the five Kaggle datasets.

Kaggle itself needs an account and API token, so each dataset is fetched from a public
GitHub mirror of the same Kaggle file. Raw files are cached under ``data/raw``.
"""

from __future__ import annotations

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SEED = 42

SOURCES = {
    # Kaggle: competitions/titanic
    "titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    # Kaggle: uciml/sms-spam-collection-dataset
    "sms_spam": "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
    # Kaggle: lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
    "imdb": "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv",
    # Kaggle: competitions/learn-ai-bbc (BBC News Classification)
    "bbc_news": "https://raw.githubusercontent.com/mdsohaib/BBC-News-Classification/master/bbc-text.csv",
    # Kaggle: uciml/iris
    "iris": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
}


@dataclass
class Split:
    """Train/test frames with a ``label`` column, plus the test rows sent to Jev."""

    train: pd.DataFrame
    test: pd.DataFrame
    jev_test: pd.DataFrame


def fetch(name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = SOURCES[name]
    path = RAW_DIR / f"{name}{Path(url).suffix}"
    if not path.exists():
        print(f"[data] downloading {name} from {url}")
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)
    return path


def _clean_html(text: str) -> str:
    return re.sub(r"<br\s*/?>", " ", text).strip()


def load(name: str) -> pd.DataFrame:
    path = fetch(name)
    if name == "titanic":
        df = pd.read_csv(path)
        df["label"] = df["Survived"].astype(int)
    elif name == "sms_spam":
        df = pd.read_csv(path, sep="\t", header=None, names=["label_str", "text"])
        df["label"] = (df["label_str"] == "spam").astype(int)
    elif name == "imdb":
        df = pd.read_csv(path)
        df["text"] = df["review"].map(_clean_html)
        df["label"] = (df["sentiment"] == "positive").astype(int)
    elif name == "bbc_news":
        df = pd.read_csv(path).rename(columns={"category": "label"})
    elif name == "iris":
        df = pd.read_csv(path).rename(columns={"species": "label"})
    else:
        raise KeyError(name)
    return df.reset_index(drop=True)


def split(name: str, jev_n: int | None) -> Split:
    """Stratified 80/20 split; Jev gets a stratified sample of at most ``jev_n`` test rows."""
    df = load(name)
    train, test = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
    test = test.reset_index(drop=True)
    if jev_n is None or jev_n >= len(test):
        jev_test = test
    else:
        jev_test, _ = train_test_split(test, train_size=jev_n, random_state=SEED, stratify=test["label"])
    return Split(train.reset_index(drop=True), test, jev_test.reset_index(drop=True))
