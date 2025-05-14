#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import joblib
import openai_utils

@st.cache_resource
def load_model():
    artifact = joblib.load("model.pkl")
    return artifact["model"], artifact["features"]

def feature_engineering(df: pd.DataFrame):
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["run_id", "timestamp"])
    df["delta"] = (
        df.groupby("run_id")["timestamp"]
          .diff()
          .dt.total_seconds()
          .fillna(0)
    )
    stages_ohe = pd.get_dummies(df["stage"], prefix="stage")
    X = pd.concat([df[["delta"]], stages_ohe], axis=1)
    return X

st.title("Аномалии CI/CD логов")
model, feature_cols = load_model()

uploaded = st.file_uploader(
    "Загрузите JSON-файлы логов",
    type="json",
    accept_multiple_files=True
)

if uploaded:
    # 1. Загрузка и объединение данных
    dfs = [pd.read_json(f) for f in uploaded]
    data = pd.concat(dfs, ignore_index=True)

    # 2. Фичи и предсказания
    X = feature_engineering(data)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_cols]

    preds = model.predict(X)
    data["anomaly"] = preds == -1

    # 3. Отображение
    st.subheader("Результаты детекции")
    st.write("Всего записей:", len(data))
    st.write("Аномалий:", int(data["anomaly"].sum()))
    st.dataframe(data[data["anomaly"]])

    # 4. Генерация описания через OpenAI
    if data["anomaly"].any():
        if st.button("🔍 Описать аномалии (OpenAI)"):
            with st.spinner("Генерируем описание…"):
                simple_list = data[data["anomaly"]][
                    ["run_id", "stage", "timestamp", "message"]
                ].to_dict(orient="records")
                try:
                    summary = openai_utils.describe_anomalies(simple_list)
                    st.markdown("**Обзор аномалий:**")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Ошибка при вызове OpenAI: {e}")

    # 5. Скачать результат
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Скачать все записи с меткой",
        data=csv,
        file_name="results.csv",
        mime="text/csv"
    )
