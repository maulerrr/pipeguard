FROM python:3.13-slim

WORKDIR /app

# Скопировать зависимости
COPY requirements.txt .

# Установить их
RUN pip install --no-cache-dir -r requirements.txt

# Скопировать CLI-утилиту, модель и утилиты
COPY detect_cli.py openai_utils.py model.pkl ./

# Сделать скрипт исполняемым
RUN chmod +x detect_cli.py

# По умолчанию запускаем detect_cli.py
ENTRYPOINT ["python", "detect_cli.py"]
