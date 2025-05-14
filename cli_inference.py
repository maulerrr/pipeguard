#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import joblib
import sys
from train_model import feature_engineering
from openai_utils import describe_anomalies

def main():
    parser = argparse.ArgumentParser(
        description="CLI для обнаружения аномалий в CI/CD логах"
    )
    parser.add_argument(
        "input", help="путь к JSON-файлу логов"
    )
    parser.add_argument(
        "--model", "-m",
        default="model.pkl",
        help="файл с обученной моделью"
    )
    args = parser.parse_args()

    # 1. Загрузка и препроцессинг логов
    df = pd.read_json(args.input)
    X = feature_engineering(df)

    # 2. Загрузка модели и метаданных
    artifact = joblib.load(args.model)
    model, feature_cols = artifact["model"], artifact["features"]

    # 3. Приведение колонок под модель
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    # 4. Инференс
    preds = model.predict(X)
    df["anomaly"] = preds == -1

    # 5. Вывод результатов
    anomalies = df[df["anomaly"]]
    if anomalies.empty:
        print("Аномалий не обнаружено ✅")
        sys.exit(0)

    print(f"Найдено аномалий: {len(anomalies)}\n")
    for _, row in anomalies.iterrows():
        print(
            f"- run {row.run_id}, stage={row.stage}, "
            f"ts={row.timestamp}, msg={row.message}"
        )

    # 6. Описание аномалий через OpenAI
    simple_list = anomalies[
        ["run_id", "stage", "timestamp", "message"]
    ].to_dict(orient="records")
    print("\nГенерируем обзор аномалий с помощью OpenAI…")
    try:
        summary = describe_anomalies(simple_list)
        print("\nОБЗОР:")
        print(summary)
    except Exception as e:
        print(f"[Ошибка OpenAI] {e}")

    sys.exit(1)

if __name__ == "__main__":
    main()
