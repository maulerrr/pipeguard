import os
import sys
import joblib
import pandas as pd

from typing import List, Dict, Any

_client: Any = None

def _get_openai_client() -> Any:
    global _client
    if _client is None:
        from openai import OpenAI, OpenAIError
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OpenAI API key not set (pass via --openai-key or env OPENAI_API_KEY)")
        _client = OpenAI(api_key=key)
    return _client

_model: Any = None
def load_anomaly_model(path: str = "model.pkl") -> Any:
    global _model
    if _model is None:
        if not os.path.isfile(path):
            print(f"Ошибка: не найден файл модели по пути '{path}'", file=sys.stderr)
            sys.exit(1)
        _model = joblib.load(path)
    return _model

def detect_anomalies(
    records: List[Dict[str, Any]],
    threshold: float = 0.5
) -> List[Dict[str, Any]]:
    if not records:
        return []

    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["run_id", "timestamp"])
    df["delta"] = (
        df.groupby("run_id")["timestamp"]
          .diff()
          .dt.total_seconds()
          .fillna(0)
    )

    X = df[["delta", "stage", "status", "message"]]
    model = load_anomaly_model()
    probs = model.predict_proba(X)[:, 1]
    df["anomaly_prob"] = probs

    anomalies = df[df["anomaly_prob"] > threshold]
    return anomalies.to_dict(orient="records")


def describe_anomalies(
    anomalies: List[Dict[str, Any]],
    max_tokens: int = 256
) -> str:
    if not anomalies:
        return "Аномалий не обнаружено."
 
    client = _get_openai_client()

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
    except Exception as e:
        return f"Ошибка при запросе к OpenAI: {e}"

    return resp.choices[0].message.content.strip()

def detect_and_describe(
    records: List[Dict[str, Any]],
    threshold: float = 0.5,
    max_tokens: int = 256
) -> str:
    ann = detect_anomalies(records, threshold=threshold)
    return describe_anomalies(ann, max_tokens=max_tokens)
