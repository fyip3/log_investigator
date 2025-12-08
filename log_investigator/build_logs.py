import json
import sqlite3
from pathlib import Path

SCENARIO_DIR = Path("log_scenarios/train")
DB_PATH = Path("logs.db")

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Schema expected by log_search_tools.py
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS logs (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            service   TEXT NOT NULL,
            level     TEXT NOT NULL,
            trace_id  TEXT,
            message   TEXT NOT NULL,
            context   TEXT
        )
        """
    )

    cur.execute("DELETE FROM logs")

    for scenario_file in sorted(SCENARIO_DIR.glob("scenario-*.json")):
        with scenario_file.open() as f:
            scenario = json.load(f)

        scenario_id = scenario["id"]
        logs = scenario["logs"]

        for log in logs:
            context = json.dumps({"scenario_id": scenario_id})

            cur.execute(
                """
                INSERT INTO logs (timestamp, service, level, trace_id, message, context)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    log["timestamp"],
                    log["service"],
                    log["level"],
                    log["trace_id"],
                    log["message"],
                    context,
                ),
            )

    conn.commit()
    conn.close()
    print(f"Created/updated {DB_PATH} from {SCENARIO_DIR}")

if __name__ == "__main__":
    main()
