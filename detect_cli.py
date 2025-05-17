import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from openai import OpenAI
import openai_utils
import subprocess

inp = Path(sys.argv[1])
if inp.suffix == ".log":
    tmp = Path("/tmp/all_logs.json")
    subprocess.run([
        sys.executable, "convert_workflow_logs.py",
        str(inp), str(tmp)
    ], check=True)
    
    sys.argv[1] = str(tmp)

def load_records(path: Path):
    try:
        df = pd.read_json(path)
        return df.to_dict(orient="records")
    except ValueError:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported format in {path}")

def override_openai_key(key: str):
    """Re-instantiate the OpenAI client inside openai_utils."""
    import os
    os.environ["OPENAI_API_KEY"] = key
    
    openai_utils.client = OpenAI(api_key=key)

def main():
    parser = argparse.ArgumentParser(
        description="CLI для обнаружения и обзора аномалий в CI/CD логах"
    )
    parser.add_argument(
        "input",
        help="путь к JSON-файлу логов"
    )
    parser.add_argument(
        "-k", "--openai-key",
        help="OpenAI API key (если не указано, берётся из env OPENAI_API_KEY)"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=float,
        default=0.5,
        help="порог вероятности аномалии (0–1, default=0.5)"
    )
    parser.add_argument(
        "-d", "--describe",
        action="store_true",
        help="сгенерировать обзор аномалий через OpenAI"
    )
    parser.add_argument(
        "-m", "--model",
        default="model_supervised.pkl",
        help="путь к файлу модели (default=model_supervised.pkl)"
    )
    args = parser.parse_args()

    if args.openai_key:
        override_openai_key(args.openai_key)

    path = Path(args.input)
    if not path.exists():
        print(f"Error: file {path} not found", file=sys.stderr)
        sys.exit(1)

    try:
        records = load_records(path)
    except Exception as e:
        print(f"Error reading {path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    anomalies = openai_utils.detect_anomalies(
        records,
        threshold=args.threshold
    )

    if not anomalies:
        print("Аномалий не обнаружено ✅")
        sys.exit(0)

    
    print(f"Найдено аномалий: {len(anomalies)} (threshold={args.threshold})\n")
    for a in anomalies:
        print(
            f"- P={a['anomaly_prob']:.2f} | run {a['run_id']}, "
            f"stage={a['stage']}, status={a['status']}, "
            f"ts={a['timestamp']}, msg=\"{a['message']}\""
        )
    
    if args.describe:
        print("\nГенерируем обзор аномалий через OpenAI…")
        try:
            summary = openai_utils.describe_anomalies(anomalies)
            print("\n=== ОБЗОР АНОМАЛИЙ ===")
            print(summary)
        except Exception as e:
            print(f"[OpenAI error] {e}", file=sys.stderr)

    sys.exit(1)

if __name__ == "__main__":
    main()
