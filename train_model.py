#!/usr/bin/env python3
# train_model.py

import os
import sys
import glob
import joblib
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

def load_data(logs_pattern: str) -> pd.DataFrame:
    files = glob.glob(logs_pattern)
    if not files:
        print(f"Ошибка: не найдены файлы логов по паттерну {logs_pattern}", file=sys.stderr)
        sys.exit(1)

    dfs = []
    for fn in sorted(files):
        try:
            dfs.append(pd.read_json(fn))
        except ValueError as e:
            print(f"Warning: не удалось прочитать {fn}: {e}", file=sys.stderr)

    if not dfs:
        print(f"Ошибка: ни один файл не был загружен из {logs_pattern}", file=sys.stderr)
        sys.exit(1)

    data = pd.concat(dfs, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    return data

def prepare_features(df: pd.DataFrame):
    # Sort and compute delta
    df = df.sort_values(["run_id", "timestamp"])
    df["delta"] = (
        df.groupby("run_id")["timestamp"]
          .diff()
          .dt.total_seconds()
          .fillna(0)
    )
    # We'll use: 'delta', 'stage', 'status', 'message'
    X = df[["delta", "stage", "status", "message"]].copy()
    y = df["label"].astype(int)
    return X, y

def build_pipeline():
    # Numeric features
    num_features = ["delta"]
    num_transformer = StandardScaler()

    # Categorical features
    cat_features = ["stage", "status"]
    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    # Text feature
    text_feature = "message"
    text_transformer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1,2),
        stop_words="english"  # remove English stopwords—customize for Russian if needed
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features),
            ("txt", text_transformer, text_feature),
        ],
        remainder="drop"
    )

    # You can swap RandomForest for MLPClassifier, XGBClassifier, etc.
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", clf),
    ])

    return pipeline

def plot_confusion(cm, title):
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0,1])
    plt.yticks([0,1])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha="center", va="center")
    plt.tight_layout()

def main():
    parser = argparse.ArgumentParser(
        description="Train supervised model on CI/CD logs with text+CATEGORY+numeric features"
    )
    parser.add_argument(
        "--logs-dir", "-d", default="logs",
        help="директория с JSON-логами (по умолчанию 'logs')"
    )
    parser.add_argument(
        "--test-size", "-t", type=float, default=0.2,
        help="доля тестового набора"
    )
    args = parser.parse_args()

    pattern = os.path.join(args.logs_dir, "run_*.json")
    print(f"Loading logs from: {pattern}")
    df = load_data(pattern)
    X, y = prepare_features(df)
    print(f"Total records: {len(y)}, Anomaly rate: {y.mean():.2%}")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # Build & train
    pipe = build_pipeline()
    print("Training supervised classifier…")
    pipe.fit(X_train, y_train)

    # Predict & evaluate
    y_pred_train = pipe.predict(X_train)
    y_pred_test  = pipe.predict(X_test)

    print("\n=== TRAIN SET METRICS ===")
    cm_train = confusion_matrix(y_train, y_pred_train)
    print(cm_train)
    print(classification_report(y_train, y_pred_train, digits=4))

    print("\n=== TEST SET METRICS ===")
    cm_test = confusion_matrix(y_test, y_pred_test)
    print(cm_test)
    print(classification_report(y_test, y_pred_test, digits=4))

    # ROC & PR
    y_score = pipe.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    ap = average_precision_score(y_test, y_score)

    # Save model
    joblib.dump(pipe, "model.pkl")
    print("\nSupervised model saved as 'model.pkl'")

    # Visualization
    plot_confusion(cm_train, "Confusion Matrix — Train")
    plot_confusion(cm_test, "Confusion Matrix — Test")
    plt.show()

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.2f}")
    plt.title("Precision–Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
