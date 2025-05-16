#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import openai_utils

st.set_page_config(page_title="Аномалии CI/CD логов", layout="wide")
st.title("🔍 Аномалии CI/CD логов")

# Порог вероятности аномалии
threshold = st.sidebar.slider(
    "Порог вероятности аномалии",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01
)

uploaded = st.file_uploader(
    "Загрузите один или несколько JSON-файлов логов",
    type="json",
    accept_multiple_files=True
)

if uploaded:
    # 1. Чтение и объединение
    dfs = []
    for f in uploaded:
        try:
            dfs.append(pd.read_json(f))
        except ValueError:
            st.warning(f"Не удалось прочитать {f.name}")
    if not dfs:
        st.error("Нет корректных JSON-файлов")
        st.stop()

    data = pd.concat(dfs, ignore_index=True)
    st.sidebar.metric("Всего записей", len(data))

    # 2. Детекция
    # преобразуем DataFrame в список dict-ов
    records: List[Dict[str, Any]] = data.to_dict(orient="records")
    anomalies = openai_utils.detect_anomalies(records, threshold=threshold)
    n_anom = len(anomalies)
    st.sidebar.metric("Найдено аномалий", n_anom)

    # 3. Отображение таблицы аномалий
    if n_anom:
        df_anom = pd.DataFrame(anomalies)
        # упорядочим по вероятности
        df_anom = df_anom.sort_values("anomaly_prob", ascending=False)
        st.subheader("⚠️ Список потенциальных аномалий")
        st.dataframe(
            df_anom[[
                "anomaly_prob", "run_id", "stage", "status", "timestamp", "message"
            ]],
            use_container_width=True
        )

        # 4. Описание через OpenAI
        if st.button("📝 Описать аномалии"):
            with st.spinner("Генерируем обзор…"):
                try:
                    summary = openai_utils.describe_anomalies(anomalies)
                    st.markdown("**Обзор аномалий (2–3 предложения):**")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Ошибка при вызове OpenAI: {e}")
    else:
        st.info("Аномалий выше порога не найдено")

    # 5. Скачивание всего результата с метками и вероятностями
    # Добавим колонку probability в исходный DataFrame
    df_all = data.copy()
    # сначала инициализируем всем нулевыми
    df_all["anomaly_prob"] = 0.0
    # заполнить для найденных
    for rec in anomalies:
        # найдём строки, совпадающие по индексу и run_id+timestamp+stage
        mask = (
            (df_all["run_id"] == rec["run_id"]) &
            (df_all["timestamp"].astype(str) == str(rec["timestamp"])) &
            (df_all["stage"] == rec["stage"])
        )
        df_all.loc[mask, "anomaly_prob"] = rec["anomaly_prob"]

    csv = df_all.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Скачать результат (все записи с вероятностями)",
        data=csv,
        file_name="cicd_anomaly_results.csv",
        mime="text/csv"
    )
else:
    st.info("Загрузите JSON-файлы логов для анализа")
