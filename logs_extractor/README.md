# Генератор синтетических логов CI/CD

Утилита `generate_logs.py` служит для быстрого создания набора синтетических логов CI/CD-пайплайнов с размеченными аномалиями.


## 📦 Требования

- Python 3.7+
- Библиотеки:
  - `argparse` (входит в стандартную библиотеку)
  - `json` (стандартная)
  - `random` (стандартная)
  - `datetime` (стандартная)


## Установка

1. Склонируйте репозиторий или скачайте файл `generate_logs.py`.
2. Сделайте скрипт исполняемым (необязательно): 
```
chmod +x generate_logs.py
```

(Опционально) Создайте и активируйте виртуальное окружение:
```
python3 -m venv venv
source venv/bin/activate
```

## Использование

```
usage: generate_logs.py [-h] [--output-dir OUTPUT_DIR] [--runs RUNS]
                        [--anomaly-prob ANOMALY_PROB]

Генератор синтетических CI/CD логов

optional arguments:
  -h, --help            показать это сообщение и выйти
  -o, --output-dir      директория для сохранения JSON-файлов (default: "logs")
  -r, --runs            количество прогонов/сессий пайплайнов (default: 100)
  -p, --anomaly-prob    вероятность аномалий в каждом шаге (0.0–1.0, default: 0.05)
  ```

## Примеры

### Сгенерировать 200 прогонов с 3 % аномалий, сохранить в папке `logs/`
```
python generate_logs.py --runs 200 --anomaly-prob 0.03
```

После выполнения в директории `logs/` появится 200 файлов:

```
logs/
├── run_001.json
├── run_002.json
├── ...
└── run_200.json
```

### Генерация в пользовательскую папку

```
python generate_logs.py -r 50 -p 0.10 -o synthetic_logs
```

Результат будет в `synthetic_logs/`.

---

### Формат выходных файлов

Каждый файл `run_XXX.json` содержит массив записей вида:
```
[
  {
    "run_id": 1,
    "timestamp": "2025-05-14T12:00:05.123456",
    "stage": "checkout",
    "status": "INFO",
    "message": "checkout succeeded",
    "label": 0              // 0 = норма, 1 = аномалия
  },
  {
    "run_id": 1,
    "timestamp": "2025-05-14T12:00:25.654321",
    "stage": "install",
    "status": "ERROR",
    "message": "install failed",
    "label": 1
  },
  ...
]
```


`run_id `— номер сессии пайплайна.

`timestamp` — метка времени шага.

`stage` — этап выполнения (из списка ["checkout", "install", "build", "test", "deploy"]).

`status` — "INFO" для нормального шага, "ERROR" для аномалии.

`message` — текст сообщения.

`label` — числовая метка (0 — норма, 1 — аномалия).

---

# Дальнейшие шаги

* Использовать полученные JSON-логи для тренировки модели.

* Перенести генерацию в CI/CD, запускать скрипт прямо в workflow.

* Настроить параметры runs и anomaly-prob для более реалистичных сценариев.