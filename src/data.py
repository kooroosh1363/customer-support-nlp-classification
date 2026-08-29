from __future__ import annotations

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
EXPECTED_CLASSES = 77


def _load_csv(url: str, name: str) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / name
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(url)
        df.to_csv(path, index=False)
    required = {"text", "category"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing expected columns in {name}: {required - set(df.columns)}")
    if df[["text", "category"]].isna().any().any():
        raise ValueError(f"Null text/category values found in {name}")
    return df[["text", "category"]].copy()


def load_dataset(random_state: int = 42):
    official_train = _load_csv(TRAIN_URL, "banking77_train.csv")
    test = _load_csv(TEST_URL, "banking77_test.csv")

    classes = sorted(official_train["category"].unique().tolist())
    if len(classes) != EXPECTED_CLASSES:
        raise ValueError(f"Expected {EXPECTED_CLASSES} classes, found {len(classes)}")
    if set(test["category"].unique()) != set(classes):
        raise ValueError("Official test labels do not match training label set")

    train, val = train_test_split(
        official_train,
        test_size=0.20,
        random_state=random_state,
        stratify=official_train["category"],
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    duplicate_train_val = len(set(train["text"]).intersection(set(val["text"])))
    duplicate_train_test = len(set(train["text"]).intersection(set(test["text"])))

    audit = {
        "official_train_rows": int(len(official_train)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "classes": int(len(classes)),
        "train_min_class_rows": int(train["category"].value_counts().min()),
        "train_max_class_rows": int(train["category"].value_counts().max()),
        "train_val_exact_text_overlap": int(duplicate_train_val),
        "train_test_exact_text_overlap": int(duplicate_train_test),
        "split_policy": "official Banking77 test locked; official train stratified 80/20 into train/validation",
    }
    return train, val, test, audit, classes
