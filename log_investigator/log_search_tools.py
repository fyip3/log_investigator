
import sqlite3
import logging
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)

# Override per-call with the `db_path` argument if needed.
DEFAULT_LOG_DB_PATH = "logs.db"

_conn: Optional[sqlite3.Connection] = None


def get_connection(db_path: str = DEFAULT_LOG_DB_PATH) -> sqlite3.Connection:
    """Return a cached SQLite connection with row access by column name."""
    global _conn
    if _conn is None:
        logger.info("Opening log database at %s", db_path)
        _conn = sqlite3.connect(db_path)
        _conn.row_factory = sqlite3.Row
    return _conn


@dataclass
class LogEntry:
    """Structured representation of a single log line.

    This assumes the following SQLite schema:

        CREATE TABLE logs (
            id        INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,      -- ISO8601 string
            service   TEXT NOT NULL,      -- e.g. "payments-api"
            level     TEXT NOT NULL,      -- DEBUG / INFO / WARN / ERROR / FATAL
            trace_id  TEXT,               -- nullable trace ID
            message   TEXT NOT NULL,
            context   TEXT                -- optional JSON string with extra fields
        );
    """

    id: int
    timestamp: str
    service: str
    level: str
    trace_id: Optional[str]
    message: str
    context: Optional[str] = None


def _row_to_log_entry(row: sqlite3.Row) -> LogEntry:
    return LogEntry(
        id=row["id"],
        timestamp=row["timestamp"],
        service=row["service"],
        level=row["level"],
        trace_id=row["trace_id"],
        message=row["message"],
        context=row["context"],
    )


def search_logs(
    *,
    scenario_id: str,
    keywords: Optional[List[str]] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    service: Optional[str] = None,
    min_level: Optional[str] = None,
    trace_id: Optional[str] = None,
    max_results: int = 25,
    db_path: str = DEFAULT_LOG_DB_PATH,
) -> List[LogEntry]:
    """Search structured application logs.

    Args:
        scenario_id: REQUIRED scenario ID to filter logs (ensures isolation).
        keywords: Optional list of substrings to match in log messages (OR logic).
        start_time: ISO8601 lower bound on timestamp (inclusive).
        end_time: ISO8601 upper bound on timestamp (inclusive).
        service: Optional service / component name filter.
        min_level: Optional minimum log level
            (DEBUG < INFO < WARN < ERROR < FATAL).
        trace_id: Restrict logs to a single trace.
        max_results: Cap on results; must be <= 500.
        db_path: Optional override of the log DB location.

    Returns:
        List of LogEntry instances sorted by timestamp ascending.
    """
    # If no filters provided, just return max_results logs (useful for exploration)

    if max_results > 500:
        raise ValueError("max_results must be less than or equal to 500")

    if not scenario_id:
        raise ValueError("scenario_id is required for log isolation")

    conn = get_connection(db_path)

    clauses = []
    params: list[object] = []

    # Filter by scenario_id to ensure isolation
    clauses.append("json_extract(context, '$.scenario_id') = ?")
    params.append(scenario_id)

    # Keyword filters - OR logic
    if keywords:
        kw_clauses = []
        for kw in keywords:
            kw_clauses.append("message LIKE ?")
            params.append(f"%{kw}%")
        if kw_clauses:
            clauses.append(f"({' OR '.join(kw_clauses)})")

    # Time filters
    if start_time:
        clauses.append("timestamp >= ?")
        params.append(start_time)
    if end_time:
        clauses.append("timestamp <= ?")
        params.append(end_time)

    # Service filter
    if service:
        clauses.append("service = ?")
        params.append(service)

    # Trace ID filter
    if trace_id:
        clauses.append("trace_id = ?")
        params.append(trace_id)

    # Log level filter with a simple ordinal mapping
    if min_level:
        min_level = min_level.upper()
        # Accept both WARN and WARNING
        if min_level == "WARNING":
            min_level = "WARN"
        level_rank = {"DEBUG": 0, "INFO": 1, "WARN": 2, "ERROR": 3, "FATAL": 4}
        if min_level not in level_rank:
            raise ValueError(
                "min_level must be one of DEBUG, INFO, WARN, WARNING, ERROR, FATAL"
            )
        clauses.append(
            "CASE level "
            "WHEN 'DEBUG' THEN 0 "
            "WHEN 'INFO' THEN 1 "
            "WHEN 'WARN' THEN 2 "
            "WHEN 'ERROR' THEN 3 "
            "WHEN 'FATAL' THEN 4 "
            "ELSE 0 END >= ?"
        )
        params.append(level_rank[min_level])

    where_sql = " AND ".join(clauses) if clauses else "1=1"

    sql = (
        "SELECT id, timestamp, service, level, trace_id, message, context "
        "FROM logs "
        f"WHERE {where_sql} "
        "ORDER BY timestamp ASC "
        "LIMIT ?"
    )
    params.append(max_results)

    logger.debug("Executing log search: %s with params %s", sql, params)
    cur = conn.execute(sql, params)
    rows = cur.fetchall()
    return [_row_to_log_entry(r) for r in rows]


def read_trace(
    *,
    scenario_id: str,
    trace_id: str,
    db_path: str = DEFAULT_LOG_DB_PATH,
    max_events: int = 30,
) -> List[LogEntry]:
    """Fetch all log events for a particular trace ID, ordered by timestamp.

    This is especially helpful once the agent has discovered a suspicious
    trace ID and wants to follow it across multiple services.
    """
    if not scenario_id:
        raise ValueError("scenario_id is required for log isolation")

    if not trace_id:
        raise ValueError("trace_id is required")

    if max_events <= 0 or max_events > 1000:
        raise ValueError("max_events must be between 1 and 1000")

    conn = get_connection(db_path)
    sql = (
        "SELECT id, timestamp, service, level, trace_id, message, context "
        "FROM logs "
        "WHERE trace_id = ? AND json_extract(context, '$.scenario_id') = ? "
        "ORDER BY timestamp ASC "
        "LIMIT ?"
    )
    logger.debug("Reading trace %s for scenario %s (max_events=%s)", trace_id, scenario_id, max_events)
    cur = conn.execute(sql, (trace_id, scenario_id, max_events))
    rows = cur.fetchall()
    return [_row_to_log_entry(r) for r in rows]
