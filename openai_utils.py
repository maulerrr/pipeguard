# openai_utils.py
import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def describe_anomalies(anomalies: list[dict], max_tokens: int = 256) -> str:
    """
    anomalies: список словарей с ключами 'run_id', 'stage', 'timestamp', 'message'
    Возвращает: сгенерированный OpenAI текст-описание
    """
    if not anomalies:
        return "Аномалий не обнаружено."

    # Формируем текстовый блок с аномалиями
    details = "\n".join(
        f"- run {a['run_id']}, stage '{a['stage']}', time={a['timestamp']}, msg='{a['message']}'"
        for a in anomalies
    )
    prompt = (
        "Ты — помощник по анализу CI/CD-логов. Ниже список аномалий, которые обнаружила модель:\n\n"
        f"{details}\n\n"
        "Составь краткий обзор (2–3 предложения) с возможными причинами и рекомендациями."
    )

    resp = openai.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()
