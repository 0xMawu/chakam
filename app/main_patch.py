# ============================================================
# PATCH for app/main.py
# Two small additions only — nothing else changes.
# ============================================================

# -----------------------------------------------------------
# 1. Add this import near the top of main.py, alongside the
#    other `from app import ...` line (line 56):
# -----------------------------------------------------------
#
#   from app import auth, clustering, database, drive_watcher, ingest_queue, ...
#
# i.e. just insert `drive_watcher,` into the existing import.
# -----------------------------------------------------------


# -----------------------------------------------------------
# 2. Add one line at the END of the on_startup() function
#    (after the reconcile_stuck_folders try/except block):
# -----------------------------------------------------------
#
#   @app.on_event("startup")
#   def on_startup():
#       logging.basicConfig(level=logging.INFO, force=True)
#       logging.getLogger("app").setLevel(logging.INFO)
#       database.init_db()
#       try:
#           ingest_queue.reconcile_stuck_folders()
#       except Exception:
#           logger.exception("Skipping stuck-folder reconciliation -- couldn't reach Redis.")
#
#       # ↓ ADD THIS LINE
#       drive_watcher.start()
#
# -----------------------------------------------------------
# That's it. The watcher thread is daemon=True so it stops
# automatically when the main process exits — no shutdown
# hook needed.
# -----------------------------------------------------------
