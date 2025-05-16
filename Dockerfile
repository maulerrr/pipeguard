FROM python:3.13-slim

WORKDIR /app

# 1) copy & install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 2) copy all scripts + model
COPY detect_cli.py openai_utils.py convert_workflow_logs.py model.pkl ./

RUN chmod +x detect_cli.py convert_workflow_logs.py

# 3) entrypoint: wrapper that converts if needed, then runs detection
ENTRYPOINT ["./detect_cli.py"]
