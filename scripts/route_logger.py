import csv
from pathlib import Path
from datetime import datetime

LOG_PATH = Path("reports/route_log.csv")

def log_call(model: str, tokens: int, cost: float, latency: float):
    """
    Basit logger: her satır bir çağrı.
    model: kullanılan model adı
    tokens: işlenen token sayısı
    cost: maliyet (USD)
    latency: saniye
    """
    LOG_PATH.parent.mkdir(exist_ok=True, parents=True)
    file_exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "model", "tokens", "cost", "latency"])
        writer.writerow([datetime.utcnow().isoformat(), model, tokens, cost, latency])
