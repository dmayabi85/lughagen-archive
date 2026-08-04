#!/usr/bin/env python3
"""
/workspace/scripts/registry_sqlite.py

LughaGen Training Framework v1.5.1 §7.2 (priority 2: SQLite)
SQLite mirror of registry_csv.py. The CSV is the source of truth (§7.2
priority 1); the SQLite is a derived index for query convenience (faster
filtering, joins, ad-hoc analysis).

Operations:
    rebuild       — drop tables, re-create from runs.csv
    import-row    — import a single run record (called by train_orchestrator)
    sync          — append rows that exist in CSV but not yet in SQLite
    query         — convenience CLI for SELECT statements
    stats         — quick summary by phase

Schema mirrors §7.1 with nested fields stored as JSON columns where the CSV
also serialises them as JSON strings (language_split, per-lang metrics, etc.).
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/workspace/scripts")
from registry_csv import (
    SCHEMA_FIELDS, REGISTRY_DIR, RUNS_CSV, read_all_runs,
)

DB_PATH = REGISTRY_DIR / "runs.sqlite"
TABLE_NAME = "runs"


# Which SCHEMA_FIELDS columns should be stored as REAL, INTEGER, BOOL or JSON in SQLite
TYPE_HINTS = {
    "seed": "INTEGER",
    "is_confirmation_run": "INTEGER",       # boolean as 0/1
    "factors.batch_size": "INTEGER",
    "factors.sequence_length": "INTEGER",
    "dataset.total_words": "INTEGER",
    "dataset.total_tokens_llama": "INTEGER",
    "dataset.total_documents": "INTEGER",
    "hyperparameters.peak_lr": "REAL",
    "hyperparameters.warmup_steps": "INTEGER",
    "hyperparameters.weight_decay": "REAL",
    "hyperparameters.max_grad_norm": "REAL",
    "hyperparameters.lora_r": "INTEGER",
    "hyperparameters.lora_alpha": "INTEGER",
    "hyperparameters.lora_dropout": "REAL",
    "hyperparameters.num_train_epochs": "INTEGER",
    "gpu_config.gpu_count": "INTEGER",
    "gpu_config.gradient_checkpointing": "INTEGER",
    "gpu_config.flash_attention_engaged": "INTEGER",
    "gpu_hours_consumed": "REAL",
    "wall_clock_seconds": "INTEGER",
    "metrics.baseline_perplexity": "REAL",
    "metrics.validation_perplexity": "REAL",
    "metrics.pph_score": "REAL",
    "metrics.pph_log_ratio": "REAL",
    "metrics.pph_flag_negative": "INTEGER",
    "metrics.tokens_per_sec": "REAL",
    "metrics.peak_vram_gb": "REAL",
    "metrics.gpu_utilization_pct": "REAL",
    "metrics.convergence_steps_to_5pct": "INTEGER",
    "metrics.token_fertility_ratio": "REAL",
    "metrics.per_language_grad_norm_ratio_mean": "REAL",
    "confirmation_run_metadata.predicted_pph_mean": "REAL",
    "confirmation_run_metadata.predicted_pph_bootstrap_low": "REAL",
    "confirmation_run_metadata.predicted_pph_bootstrap_high": "REAL",
    "confirmation_run_metadata.selection_rank": "INTEGER",
    "confirmation_run_metadata.flagged_for_manual_review": "INTEGER",
    "training_events.early_stopped": "INTEGER",
}

# All others default to TEXT (covers identifiers, dates, version strings,
# JSON-serialised dicts/lists)
def sql_type(field: str) -> str:
    return TYPE_HINTS.get(field, "TEXT")


def sql_column_name(field: str) -> str:
    """Convert dotted field name to SQL column name (replace . with _)."""
    return field.replace(".", "_")


def column_definitions() -> str:
    """Build CREATE TABLE column list."""
    cols = []
    for field in SCHEMA_FIELDS:
        col = sql_column_name(field)
        typ = sql_type(field)
        if field == "experiment_id":
            cols.append(f"{col} {typ} PRIMARY KEY")
        else:
            cols.append(f"{col} {typ}")
    return ",\n  ".join(cols)


def create_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"DROP TABLE IF EXISTS {TABLE_NAME}")
    conn.execute(f"CREATE TABLE {TABLE_NAME} (\n  {column_definitions()}\n)")
    # Useful indices for the most common queries
    conn.execute(f"CREATE INDEX idx_phase ON {TABLE_NAME}(phase)")
    conn.execute(f"CREATE INDEX idx_factorial_run_id ON {TABLE_NAME}(factorial_run_id)")
    conn.execute(f"CREATE INDEX idx_is_confirmation ON {TABLE_NAME}(is_confirmation_run)")
    conn.execute(f"CREATE INDEX idx_spec_version ON {TABLE_NAME}(spec_version)")
    conn.commit()


def coerce_value(field: str, raw: str):
    """Convert a CSV string value to the SQL column's expected type."""
    if raw == "" or raw is None:
        return None
    typ = sql_type(field)
    if typ == "INTEGER":
        if raw.lower() in ("true", "yes"):
            return 1
        if raw.lower() in ("false", "no"):
            return 0
        try:
            return int(float(raw))
        except (ValueError, TypeError):
            return None
    elif typ == "REAL":
        try:
            return float(raw)
        except (ValueError, TypeError):
            return None
    else:
        # TEXT: preserve as-is (JSON serialisation already in the string)
        return raw


def insert_row(conn: sqlite3.Connection, row: dict) -> None:
    """Insert a single row (dict keyed by SCHEMA_FIELDS dotted names)."""
    cols = [sql_column_name(f) for f in SCHEMA_FIELDS]
    placeholders = ", ".join("?" * len(SCHEMA_FIELDS))
    values = [coerce_value(f, row.get(f, "")) for f in SCHEMA_FIELDS]
    sql = (
        f"INSERT OR REPLACE INTO {TABLE_NAME} ({', '.join(cols)}) "
        f"VALUES ({placeholders})"
    )
    conn.execute(sql, values)


def rebuild_from_csv(verbose: bool = True) -> int:
    """Drop and recreate the SQLite from the canonical CSV."""
    if not RUNS_CSV.exists():
        if verbose:
            print(f"runs.csv not found at {RUNS_CSV}; nothing to rebuild")
        # Still create empty schema
        REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        create_table(conn)
        conn.close()
        return 0

    conn = sqlite3.connect(DB_PATH)
    create_table(conn)

    n = 0
    for row in read_all_runs():
        insert_row(conn, row)
        n += 1
    conn.commit()
    conn.close()

    if verbose:
        print(f"Rebuilt {DB_PATH} from {RUNS_CSV}")
        print(f"  Rows imported: {n}")
    return n


def sync(verbose: bool = True) -> int:
    """Add CSV rows not yet present in SQLite. Idempotent."""
    if not RUNS_CSV.exists():
        if verbose:
            print(f"runs.csv not found; nothing to sync")
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(f"SELECT experiment_id FROM {TABLE_NAME}")
    existing_ids = {row[0] for row in cursor.fetchall()}

    n_added = 0
    for row in read_all_runs():
        if row.get("experiment_id") and row["experiment_id"] not in existing_ids:
            insert_row(conn, row)
            n_added += 1
    conn.commit()
    conn.close()

    if verbose:
        print(f"Sync complete. Added {n_added} new rows.")
    return n_added


def query(sql: str, verbose: bool = True) -> list[tuple]:
    """Execute a read-only SELECT."""
    if not sql.strip().lower().startswith("select"):
        raise ValueError("Only SELECT statements allowed via query()")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(sql)
    rows = cursor.fetchall()
    cols = [c[0] for c in cursor.description]
    conn.close()
    if verbose:
        print(" | ".join(cols))
        print("-" * 80)
        for row in rows:
            print(" | ".join(str(v) for v in row))
    return rows


def stats() -> None:
    """Print quick summary."""
    if not DB_PATH.exists():
        print(f"DB not yet built. Run: registry_sqlite.py rebuild")
        return
    conn = sqlite3.connect(DB_PATH)

    # Rows by phase
    cursor = conn.execute(
        f"SELECT phase, COUNT(*), SUM(gpu_hours_consumed) "
        f"FROM {TABLE_NAME} GROUP BY phase"
    )
    print("=== Runs by phase ===")
    print(f"{'phase':<25} {'runs':>6} {'gpu_hours':>12}")
    for phase, n, hours in cursor.fetchall():
        hours_str = f"{hours:.1f}" if hours is not None else "0"
        print(f"{phase or 'unknown':<25} {n:>6} {hours_str:>12}")

    # Confirmation runs
    cursor = conn.execute(
        f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE is_confirmation_run = 1"
    )
    n_conf = cursor.fetchone()[0]
    print(f"\nConfirmation runs: {n_conf}")

    # Mean PPH across Phase 2
    cursor = conn.execute(
        f"SELECT AVG(metrics_pph_score) FROM {TABLE_NAME} "
        f"WHERE phase = 'Phase2_Factorial' AND metrics_pph_score IS NOT NULL"
    )
    mean_pph = cursor.fetchone()[0]
    if mean_pph is not None:
        print(f"Phase 2 mean PPH: {mean_pph:.6f}")

    conn.close()


def _smoke_test() -> int:
    """In-memory smoke test."""
    print("Running registry_sqlite.py smoke test...")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        global DB_PATH
        original_db = DB_PATH
        DB_PATH = Path(tmp) / "test_runs.sqlite"
        try:
            conn = sqlite3.connect(DB_PATH)
            create_table(conn)

            test_row = {f: "" for f in SCHEMA_FIELDS}
            test_row.update({
                "experiment_id": "EXP-TEST-001",
                "factorial_run_id": "RUN-P2-C01-S1",
                "phase": "Phase2_Factorial",
                "is_confirmation_run": "false",
                "spec_version": "1.5.1",
                "factors.lr_schedule": "cosine",
                "factors.batch_size": "128",
                "factors.optimiser": "AdamW",
                "factors.sequence_length": "1024",
                "factors.data_ordering": "curriculum",
                "seed": "42",
                "metrics.pph_score": "0.123",
                "gpu_hours_consumed": "75.5",
            })
            insert_row(conn, test_row)
            conn.commit()

            cursor = conn.execute(
                f"SELECT experiment_id, factors_batch_size, metrics_pph_score, gpu_hours_consumed "
                f"FROM {TABLE_NAME}"
            )
            rows = cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "EXP-TEST-001"
            assert rows[0][1] == 128
            assert abs(rows[0][2] - 0.123) < 1e-6
            assert abs(rows[0][3] - 75.5) < 1e-6

            conn.close()
            print("  PASS: insert + query round-trip works with type coercion")
            return 0
        finally:
            DB_PATH = original_db


def main() -> int:
    parser = argparse.ArgumentParser(description="SQLite registry backend (§7.2)")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rebuild")
    sub.add_parser("sync")
    sub.add_parser("stats")
    sub.add_parser("smoke")
    p_query = sub.add_parser("query")
    p_query.add_argument("sql", help="SELECT statement")
    args = parser.parse_args()

    if args.cmd == "rebuild":
        rebuild_from_csv()
        return 0
    elif args.cmd == "sync":
        sync()
        return 0
    elif args.cmd == "stats":
        stats()
        return 0
    elif args.cmd == "query":
        query(args.sql)
        return 0
    elif args.cmd == "smoke":
        return _smoke_test()


if __name__ == "__main__":
    sys.exit(main())
