import re
import json
import sys
from datetime import datetime
from pathlib import Path

if len(sys.argv) != 3:
    print("Usage: convert_workflow_logs.py <raw_log.txt> <out.json>", file=sys.stderr)
    sys.exit(1)

raw_log, out_json = sys.argv[1], sys.argv[2]
stage = "init"
records = []
with open(raw_log, encoding="utf-8") as f:
    for line in f:
        
        m = re.match(r"^::group::(.+)", line)
        if m:
            stage = m.group(1).strip()
            continue
        if line.startswith("::endgroup::"):
            stage = "init"
            continue

        ts_match = re.match(r"^(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\s+(.*)", line)
        if ts_match:
            ts, msg = ts_match.groups()
        else:
            ts = datetime.utcnow().isoformat()
            msg = line.strip()

        status = "ERROR" if re.search(r"\berror\b", msg, re.IGNORECASE) else "INFO"
        records.append({
            "run_id": 1,
            "stage": stage,
            "status": status,
            "message": msg,
            "timestamp": ts
        })

with open(out_json, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
