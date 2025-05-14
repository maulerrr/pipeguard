#!/usr/bin/env python3
# generate_logs.py
import os
import json
import random
import argparse
from datetime import datetime, timedelta

STAGES = ["checkout", "install", "build", "test", "deploy"]

def generate_run(run_id: int, start_ts: datetime, anomaly_prob: float):
    records = []
    ts = start_ts
    for stage in STAGES:
        ts += timedelta(seconds=random.randint(5, 30))
        is_anomaly = random.random() < anomaly_prob
        status = "ERROR" if is_anomaly else "INFO"
        msg = f"{stage} {'failed' if is_anomaly else 'succeeded'}"
        records.append({
            "run_id": run_id,
            "timestamp": ts.isoformat(),
            "stage": stage,
            "status": status,
            "message": msg,
            "label": int(is_anomaly)  # 1 = аномалия, 0 = норма
        })
    return records

def main():
    parser = argparse.ArgumentParser(description="Генератор синтетических CI/CD логов")
    parser.add_argument("--output-dir", "-o", default="logs", help="куда сохранить файлы")
    parser.add_argument("--runs", "-r", type=int, default=100, help="количество прогонов")
    parser.add_argument("--anomaly-prob", "-p", type=float, default=0.05, help="вероятность аномалии в шаге")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base_ts = datetime.now()

    for run_id in range(1, args.runs + 1):
        records = generate_run(run_id, base_ts, args.anomaly_prob)
        fname = os.path.join(args.output_dir, f"run_{run_id:03d}.json")
        with open(fname, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        base_ts += timedelta(minutes=5)  # следующий прогон чуть позже

    print(f"Сгенерировано {args.runs} файлов в '{args.output_dir}/'")

if __name__ == "__main__":
    main()
