# openai_utils.py

import os
import sys
import joblib
import pandas as pd

from typing import List, Dict, Any
from openai import OpenAI
import openai

# ——— 1) Load & validate OpenAI key ——————————————————————————————
_api_key = os.getenv("OPENAI_API_KEY")
if not _api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable not set")
client = OpenAI(api_key=_api_key)

# ——— 2) Lazy‐load the anomaly model pipeline ——————————————————————
_model: Any = None
def load_anomaly_model(path: str = "model.pkl") -> Any:
    global _model
    if _model is None:
        if not os.path.isfile(path):
            print(f"Ошибка: не найден файл модели по пути '{path}'", file=sys.stderr)
            sys.exit(1)
        _model = joblib.load(path)
    return _model

# ——— 3) Detect anomalies with probabilities ——————————————————————
def detect_anomalies(
    records: List[Dict[str, Any]],
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """
    records: список словарей с ключами 'run_id','stage','status','timestamp','message'
    threshold: минимальная вероятностная оценка аномалии (0–1)
    Возвращает: подсписок записей, где модель.predict_proba(...)[:,1] > threshold,
                каждая запись дополняется полем 'anomaly_prob'.
    """
    if not records:
        return []

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["run_id", "timestamp"])

    # replicate the same feature‐prep as your supervised pipeline expects:
    df["delta"] = (
        df.groupby("run_id")["timestamp"]
          .diff()
          .dt.total_seconds()
          .fillna(0)
    )

    X = df[["delta", "stage", "status", "message"]]
    model = load_anomaly_model()

    # predict_proba → second column is P(anomaly)
    probs = model.predict_proba(X)[:, 1]
    df["anomaly_prob"] = probs

    # filter by threshold
    anomalies = df[df["anomaly_prob"] > threshold]
    return anomalies.to_dict(orient="records")

# ——— 4) Describe anomalies via OpenAI ——————————————————————————
def describe_anomalies(
    anomalies: List[Dict[str, Any]],
    max_tokens: int = 256
) -> str:
    """
    anomalies: список от detect_anomalies, каждая запись имеет
               run_id, stage, status, timestamp, message, anomaly_prob
    Возвращает: строку с обзором от ChatGPT
    """
    if not anomalies:
        return "Аномалий не обнаружено."

    details = "\n".join(
        f"- [P={a['anomaly_prob']:.2f}] run {a['run_id']}, "
        f"stage='{a['stage']}', status={a['status']}, "
        f"time={a['timestamp']}, msg=\"{a['message']}\""
        for a in anomalies
    )

    prompt = (
        "Ты — ассистент по анализу CI/CD-логов. Ниже список потенциальных аномалий с их вероятностями:\n\n"
        f"{details}\n\n"
        "Составь краткий обзор (2–3 предложения), указав наиболее вероятные причины и рекомендации по устранению."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
    except openai.OpenAIError as e:
        return f"Ошибка при запросе к OpenAI: {e}"

    return resp.choices[0].message.content.strip()

# ——— Optional convenience: run detection + description —————————————————
def detect_and_describe(
    records: List[Dict[str, Any]],
    threshold: float = 0.5,
    max_tokens: int = 256
) -> str:
    """
    Быстрый путь: детектим аномалии и сразу генерируем их обзор.
    """
    ann = detect_anomalies(records, threshold=threshold)
    return describe_anomalies(ann, max_tokens=max_tokens)
