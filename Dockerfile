FROM python:3.11-slim

WORKDIR /app

# install dos2unix (to convert CRLF→LF)
RUN apt-get update && apt-get install -y dos2unix && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy sources
COPY detect_cli.py openai_utils.py convert_workflow_logs.py model.pkl ./

# convert line endings
RUN dos2unix *.py

# make sure they’re executable
RUN chmod +x *.py

ENTRYPOINT ["./detect_cli.py"]
