# ============================================================
# PATCH for app/database.py
# Add these three things in the locations described below.
# ============================================================

# -----------------------------------------------------------
# 1. Add this block to the SCHEMA string (paste it just before
#    the closing triple-quote of SCHEMA, after the last
#    CREATE INDEX statement):
# -----------------------------------------------------------

SCHEMA_ADDITION = """
-- Watcher: persistent key/value store for the drive_watcher background
-- thread (e.g. "last_checked" timestamp). A generic k/v table avoids
-- needing a new migration every time watcher state grows.
CREATE TABLE IF NOT EXISTS watcher_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# -----------------------------------------------------------
# 2. No migration needed — watcher_state uses CREATE TABLE IF
#    NOT EXISTS so it is created on first startup automatically.
#    Nothing to add to _run_migrations().
# -----------------------------------------------------------


# -----------------------------------------------------------
# 3. Add these two functions anywhere after get_connection()
#    in database.py (e.g. after reset_stuck_processing_folders):
# -----------------------------------------------------------

def get_watcher_state(key: str) -> str | None:
    """Read one value from the watcher key/value store.
    Returns None when the key has never been written."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT value FROM watcher_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None


def set_watcher_state(key: str, value: str) -> None:
    """Upsert one value in the watcher key/value store."""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO watcher_state (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        conn.commit()
