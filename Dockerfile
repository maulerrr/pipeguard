FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY openai_utils.py train_model.py detect_cli.py ./

RUN chmod +x detect_cli.py

ENTRYPOINT ["python", "detect_cli.py"]
