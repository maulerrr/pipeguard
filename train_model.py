#!/usr/bin/env python3
# train_model.py
import glob
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

def load_data(logs_pattern="logs/run_*.json"):
    files = glob.glob(logs_pattern)
    all_records = []
    for fn in files:
        df = pd.read_json(fn)
        all_records.append(df)
    data = pd.concat(all_records, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    return data

def feature_engineering(df: pd.DataFrame):
    df = df.sort_values(["run_id", "timestamp"])
    # дельта времени между записями
    df["delta"] = df.groupby("run_id")["timestamp"].diff().dt.total_seconds().fillna(0)
    # one-hot-кодирование этапов
    stages_ohe = pd.get_dummies(df["stage"], prefix="stage")
    # используем только delta + этапы
    X = pd.concat([df[["delta"]], stages_ohe], axis=1)
    return X

def main():
    print("Загрузка данных…")
    df = load_data()
    print(f"Всего записей: {len(df)}")
    X = feature_engineering(df)

    print("Тренировка IsolationForest…")
    model = IsolationForest(
        n_estimators=50,
        max_samples="auto",
        contamination=0.05,
        random_state=42
    )
    model.fit(X)

    # сохраняем модель + список колонок фич
    artifact = {
        "model": model,
        "features": X.columns.to_list()
    }
    joblib.dump(artifact, "model.pkl")
    print("Модель и метаданные сохранены в 'model.pkl'")

if __name__ == "__main__":
    main()
