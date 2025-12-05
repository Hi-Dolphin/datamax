"""
Generate QA pairs from text with domain tree labeling.
Designed for Daemon (nohup/systemd) or Kubernetes execution.

Features:
- Single Instance Locking (prevents overlapping runs).
- Graceful Shutdown (finish current task before exit).
- OSS Integration & Postgres State Tracking.
"""

import os
import time
import sys
import signal
import fcntl  # UNIX/Linux only. Required for file locking.
import urllib.parse
from pathlib import Path
from typing import List, Optional
import threading
import hashlib

from loguru import logger

from datamax import DataMax
from datamax.generator.types import PersistenceConfig
from datamax.loader.core import DataLoader
from datamax.persistence.postgres import PostgresPersistence

# -----------------------------------------
#                Env Configuration
# -----------------------------------------
api_key = os.getenv("DASHSCOPE_API_KEY", "YOUR_API_KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL", "YOUR_BASE_URL")
model = os.getenv("QA_MODEL", "qwen-max")
qa_input_source = os.getenv("QA_INPUT_SOURCE", "local").lower()

# OSS (Aliyun) Configuration
oss_endpoint = os.getenv("OSS_ENDPOINT")
oss_access_key = os.getenv("OSS_ACCESS_KEY_ID")
oss_secret_key = os.getenv("OSS_ACCESS_KEY_SECRET")
oss_bucket_name = os.getenv("OSS_BUCKET_NAME")
oss_prefix = os.getenv("OSS_PREFIX", "")
oss_download_dir_env = os.getenv("OSS_DOWNLOAD_DIR")

# Global Paths
root_dir = Path(os.getenv("DATAMAX_ROOT")) if os.getenv("DATAMAX_ROOT") else Path.cwd()
train_dir_name = "train"
default_oss_download_dir = root_dir / "oss_downloads"
save_parent_path = root_dir / train_dir_name

# Generation Parameters
question_number = int(os.getenv("QA_QUESTION_NUMBER", 50))
chunk_size = int(os.getenv("QA_CHUNK_SIZE", 3000))
chunk_overlap = int(os.getenv("QA_CHUNK_OVERLAP", 1500))
max_qps = int(os.getenv("QA_MAX_QPS", 100))
schedule_interval = int(os.getenv("QA_SCHEDULE_INTERVAL", 60))


# -------------------------------------------------
#  System Utilities: Locking & Signals
# -------------------------------------------------


class DaemonContext:
    """
    Context manager to handle PID locking and Signal interception.
    Ensures safe, single-instance execution on Linux/Unix.
    """

    def __init__(self, lock_file_name="qa_daemon.lock"):
        self.lock_file = root_dir / lock_file_name
        self.fp = None
        self.stop_requested = False

    def __enter__(self):
        # 1. Acquire File Lock (Singleton Check)
        try:
            self.fp = open(self.lock_file, "w")
            # LOCK_EX: Exclusive lock, LOCK_NB: Non-blocking
            fcntl.flock(self.fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.fp.write(str(os.getpid()))
            self.fp.flush()
        except IOError:
            logger.error(f"FATAL: Process is already running. Lock file exists at: {self.lock_file}")
            logger.error("Please stop the existing process or delete/wait for the lock file.")
            sys.exit(1)

        # 2. Register Signal Handlers
        signal.signal(signal.SIGINT, self._signal_handler)  # Ctrl+C
        signal.signal(signal.SIGTERM, self._signal_handler)  # kill command

        return self

    def __exit__(self, _type, _value, _traceback):
        # Cleanup lock on exit
        if self.fp:
            try:
                fcntl.flock(self.fp, fcntl.LOCK_UN)
                self.fp.close()
                if self.lock_file.exists():
                    self.lock_file.unlink()
            except Exception as e:
                logger.warning(f"Error cleaning up lock file: {e}")

    def _signal_handler(self, signum, frame):
        """
        Sets a flag when a kill signal is received.
        Does NOT kill the process immediately (Graceful Shutdown).
        """
        sig_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        logger.warning(f"Received signal {sig_name} ({signum}).")
        logger.warning("Shutdown requested. The process will exit AFTER the current task completes.")
        self.stop_requested = True

    def should_stop(self) -> bool:
        """Check if shutdown has been requested."""
        return self.stop_requested


# -------------------------------------------------
#  Database & Persistence Configuration
# -------------------------------------------------


def get_file_hash(filepath: Path) -> str:
    """Calculates MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def build_persistence_config() -> dict | None:
    """Constructs the Postgres DSN and configuration map from env vars."""
    prefix = os.getenv("QA_DB_PREFIX", "POSTGRES")
    dsn = os.getenv("QA_DB_DSN") or os.getenv("DATABASE_URL")

    # Construct DSN if not provided explicitly
    if not dsn:
        host = os.getenv(f"{prefix}_HOST")
        user = os.getenv(f"{prefix}_USERNAME")
        password = os.getenv(f"{prefix}_PASSWORD")
        database = os.getenv(f"{prefix}_DB")
        port = os.getenv(f"{prefix}_PORT")
        schema = os.getenv(f"{prefix}_SCHEMA")
        connect_type = os.getenv(f"{prefix}_CONNECT", "postgresql")

        if not (host and user and database):
            return None

        port_part = f":{port}" if port else ""
        schema_part = f"?options=-c%20search_path%3D{schema}" if schema else ""
        username = urllib.parse.quote(user, safe="")
        password_part = f":{urllib.parse.quote(password, safe="")}" if password else ""

        dsn = f"{connect_type}://{username}{password_part}@{host}{port_part}/{database}{schema_part}"

    source_key = os.getenv("QA_SOURCE_KEY", "local_corpus")
    if not source_key:
        return None

    config = {
        "backend": "postgres",
        "dsn": dsn,
        "source_key": source_key,
        "source_name": os.getenv("QA_SOURCE_NAME"),
        "owner_team": os.getenv("QA_OWNER_TEAM"),
        "created_by": os.getenv("QA_CREATED_BY"),
        "run_name": os.getenv("QA_RUN_NAME"),
        "trigger_type": os.getenv("QA_TRIGGER_TYPE", "manual"),
        "model_key": model,
        "model_provider": os.getenv("QA_MODEL_PROVIDER"),
        "model_version": os.getenv("QA_MODEL_VERSION"),
    }

    # Optional parameters
    for k, v in {
        "question_number": question_number,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "max_qps": max_qps,
    }.items():
        if v:
            config[k] = v

    return config


persistence_config = build_persistence_config()


# -------------------------------------------------
#  OSS & File Operations
# -------------------------------------------------


def get_oss_loader() -> DataLoader:
    """Validates and returns the OSS DataLoader."""
    missing = [k for k, v in {"OSS_ENDPOINT": oss_endpoint, "OSS_BUCKET_NAME": oss_bucket_name}.items() if not v]
    if missing:
        logger.error(f"Missing environment variables: {missing}")
        raise ValueError(f"Missing OSS config: {missing}")

    loader = DataLoader(
        endpoint=oss_endpoint,
        secret_key=oss_secret_key,
        access_key=oss_access_key,
        bucket_name=oss_bucket_name,
        source="oss",
    )
    return loader


def fetch_and_filter_oss_files(loader: DataLoader) -> List[str]:
    """
    Lists files in OSS and filters out those marked as COMPLETED in DB.
    """
    if not loader.oss:
        logger.error("OSS Client not initialized.")
        return []

    # 1. List objects from OSS
    logger.info(f"Listing objects in bucket '{oss_bucket_name}' (Prefix: '{oss_prefix}')...")
    all_keys = loader.oss.get_objects_in_folders(prefix=oss_prefix)

    if not all_keys:
        logger.info("No files found in OSS bucket.")
        return []

    # 2. If no DB, perform no filtering
    if not persistence_config:
        logger.warning("No persistence config. Skipping deduplication (All files will be processed).")
        return all_keys

    # 3. Check DB status
    try:
        p_config = PersistenceConfig.from_mapping(persistence_config)
        with PostgresPersistence(p_config) as p:
            status_map = p.check_files_status(p_config.source_key, all_keys)
    except Exception as e:
        logger.error(f"Failed to check file status from DB: {e}")
        return []  # Return empty to avoid reprocessing everything on DB error

    # 4. Filter logic
    new_keys = []
    skipped_count = 0
    for key in all_keys:
        status = status_map.get(key)
        if status in ("COMPLETED", "PROCESSING"):
            # Note: "PROCESSING" is skipped here.
            # If the process crashed previously, these files might need manual reset or timeout logic
            # (handled by separate reset logic if implemented).
            skipped_count += 1
            continue
        new_keys.append(key)

    logger.info(f"Found {len(all_keys)} objects. Skipped {skipped_count}. New pending: {len(new_keys)}.")
    return new_keys


def update_file_status(oss_key: str, status: str, error_message: Optional[str] = None) -> None:
    """Updates the processing status of a file in Postgres."""
    if not oss_key or not persistence_config:
        return

    try:
        p_config = PersistenceConfig.from_mapping(persistence_config)
        with PostgresPersistence(p_config) as p:
            p.upsert_file_status(p_config.source_key, oss_key, status, error_message=error_message)
    except Exception as e:
        logger.error(f"Failed to update status in DB for {oss_key}: {e}")


# -------------------------------------------------
#  Liveness Probe
# -------------------------------------------------


def liveness_probe(ctx: DaemonContext) -> None:
    """
    Periodically checks OSS connectivity and logs status.
    Runs in a separate thread.
    """
    # Lazy import to ensure it exists or fail gracefully inside the thread
    try:
        import oss2
    except ImportError:
        logger.error("Liveness Probe: 'oss2' library not found. Probe disabled.")
        return

    logger.info("Starting Liveness Probe (10s interval)...")

    while not ctx.should_stop():
        try:
            if not (oss_access_key and oss_secret_key and oss_endpoint and oss_bucket_name):
                logger.warning("Liveness Probe: Missing OSS configuration.")
            else:
                auth = oss2.Auth(oss_access_key, oss_secret_key)
                bucket = oss2.Bucket(auth, oss_endpoint, oss_bucket_name)
                # Lightweight call to verify connectivity
                bucket.get_bucket_info()
                logger.info(f"Liveness Probe: [OK] Connected to OSS. Endpoint: {oss_endpoint}, Bucket: {oss_bucket_name}")
        except Exception as e:
            logger.error(f"Liveness Probe: [FAILED] Connection Error: {e}")

        # Sleep 60s, checking for stop signal
        for _ in range(60):
            if ctx.should_stop():
                break
            time.sleep(1)


# -------------------------------------------------
#  Database Pre-flight Check
# -------------------------------------------------


def preflight_database_check() -> None:
    """
    Checks if the database schema is up-to-date by comparing the SQL file hash.
    Automatically applies the SQL script if the hash has changed.
    """
    if not persistence_config:
        logger.info("Persistence not configured. Skipping DB checks.")
        return

    dsn = persistence_config.get("dsn")
    if not dsn:
        return

    # Locate the SQL file
    sql_file_path = Path(__file__).parent / "pgsql-database.sql"
    if not sql_file_path.exists():
        logger.warning(f"SQL file not found at {sql_file_path}. Skipping schema auto-migration.")
        return

    # Calculate current file hash
    current_hash = get_file_hash(sql_file_path)

    logger.info("Starting Database Schema Check...")

    try:
        import psycopg2
    except ImportError:
        logger.warning("psycopg2 module not found. Skipping DB checks.")
        return

    conn = None
    try:
        conn = psycopg2.connect(dsn)
        conn.autocommit = False  # Use transaction

        with conn.cursor() as cur:
            # 1. Bootstrap: Ensure Schema and Version Table exist
            # We do this in Python to ensure we can track the version even if the main SQL hasn't run yet.
            cur.execute("CREATE SCHEMA IF NOT EXISTS sdc_ai")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS sdc_ai.schema_version (
                    filename TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    applied_at TIMESTAMPTZ DEFAULT NOW()
                )
            """
            )

            # 2. Check stored hash
            cur.execute("SELECT file_hash FROM sdc_ai.schema_version WHERE filename = %s", ("pgsql-database.sql",))
            row = cur.fetchone()
            stored_hash = row[0] if row else None

            # 3. Compare and Migrate if needed
            if stored_hash != current_hash:
                if stored_hash is None:
                    logger.info("Initializing database schema for the first time...")
                else:
                    logger.info(f"Schema change detected (Hash mismatch). Upgrading schema...")

                # Read and Execute SQL Script
                with open(sql_file_path, "r", encoding="utf-8") as f:
                    sql_script = f.read()

                cur.execute(sql_script)

                # Update Hash Record
                cur.execute(
                    """
                    INSERT INTO sdc_ai.schema_version (filename, file_hash, applied_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (filename) DO UPDATE 
                    SET file_hash = EXCLUDED.file_hash, applied_at = NOW()
                """,
                    ("pgsql-database.sql", current_hash),
                )

                conn.commit()
                logger.info("Database Schema applied successfully.")
            else:
                logger.info("Database Schema is up-to-date (Hash matched).")

    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database Pre-flight Check Failed: {e}")
        sys.exit(1)
    finally:
        if conn:
            conn.close()


# -------------------------------------------------
#  QA Generation Logic
# -------------------------------------------------


def _generate_and_save_qa(input_file: Path, save_path: Path, relative_path: Path) -> Optional[str]:
    """
    Parses content and interacts with LLM to generate QA pairs.
    Returns: None if success, string error message otherwise.
    """
    try:
        dm = DataMax(file_path=str(input_file), to_markdown=True)
        data = dm.get_data()

        content = data.get("content")
        if isinstance(content, list):
            content = "\n\n".join(text for text in content if text)

        if not content or not content.strip():
            logger.warning(f"Parsed content is empty for {relative_path}. Skipping.")
            return "Empty parsed content"

        # This call might take a long time (hours/days for massive files)
        qa = dm.get_pre_label(
            content=content,
            api_key=api_key,
            base_url=base_url,
            model_name=model,
            question_number=question_number,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_qps=max_qps,
            debug=False,
            structured_data=False,
            auto_self_review_mode=True,
            review_max_qps=max_qps,
            persistence=persistence_config,
        )

        if not qa:
            logger.warning(f"No QA pairs generated for {relative_path}.")
            return "No QA generated"

        dm.save_label_data(qa, str(save_path))
        return None

    except Exception as e:
        logger.exception(f"Exception during QA generation for {relative_path}")
        return str(e)


def process_file(input_file: Path, oss_key: Optional[str] = None) -> bool:
    """
    Orchestrates the processing of a single file.
    """
    relative_path = Path(input_file.name)

    # Update Status: PROCESSING
    if oss_key:
        update_file_status(oss_key, "PROCESSING")

    # Determine save path
    save_dir = save_parent_path / "oss_output"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{input_file.stem}_train"

    logger.info(f"[START] Processing file: {oss_key or relative_path}")
    start_time = time.time()

    # Blocking call - core workload
    error_msg = _generate_and_save_qa(input_file, save_path, relative_path)

    duration = time.time() - start_time
    logger.info(f"[DONE] Processing finished in {duration:.2f}s.")

    # Update Status: COMPLETED or FAILED
    if error_msg:
        logger.error(f"Task Failed: {error_msg}")
        if oss_key:
            update_file_status(oss_key, "FAILED", error_message=error_msg)
        return False

    if oss_key:
        update_file_status(oss_key, "COMPLETED")
    return True


# -------------------------------------------------
#  Main Loop
# -------------------------------------------------


def run_batch(ctx: DaemonContext) -> None:
    """
    Fetches pending files and processes them one by one.
    Checks for shutdown signals inside the loop.
    """
    download_dir = Path(oss_download_dir_env) if oss_download_dir_env else default_oss_download_dir
    download_dir.mkdir(parents=True, exist_ok=True)

    try:
        loader = get_oss_loader()
    except Exception as e:
        logger.error(f"Failed to initialize OSS loader: {e}")
        return

    # 1. Fetch Work
    target_keys = fetch_and_filter_oss_files(loader)

    if not target_keys:
        # No work found
        return

    # 2. Process Work
    logger.info(f"Starting batch of {len(target_keys)} files.")

    for i, key in enumerate(target_keys):
        # ----------------------------------------------------------
        # Check Signal: Graceful Exit
        # If a signal (SIGTERM/SIGINT) was received during the LAST
        # process_file() call, we stop here before starting a new one.
        # ----------------------------------------------------------
        if ctx.should_stop():
            logger.warning("Stop signal detected. Aborting the remaining batch.")
            break

        try:
            filename = key.split("/")[-1]
            local_path = download_dir / filename

            logger.info(f"Downloading [{i + 1}/{len(target_keys)}]: {key}")
            if loader.oss:
                loader.oss.get_object_to_file(key, str(local_path))

            # PROCESS (This may take hours)
            process_file(local_path, oss_key=key)

            # Cleanup
            if local_path.exists():
                os.remove(local_path)

        except Exception as e:
            logger.error(f"Structured error in batch loop for {key}: {e}")
            update_file_status(key, "FAILED", error_message=f"System Error: {e}")
            # Continue to next file despite error
            continue


def main() -> None:
    # Ensure root directories exist
    save_parent_path.mkdir(parents=True, exist_ok=True)

    # Perform Database Pre-flight Check
    preflight_database_check()

    # Enter Daemon Context (Locking & Signal Handling)
    with DaemonContext() as ctx:
        logger.info(f"QA Generator Daemon Started. PID: {os.getpid()}")
        logger.info(f"Schedule Interval: {schedule_interval}s")

        # Start Liveness Probe
        probe_thread = threading.Thread(target=liveness_probe, args=(ctx,), daemon=True)
        probe_thread.start()

        # Main Event Loop
        while not ctx.should_stop():
            try:
                run_batch(ctx)
                # time.sleep(10)
                # logger.debug("mock run batch")
            except Exception as e:
                logger.critical(f"Critical error in main loop: {e}")
                time.sleep(10)  # Prevent tight crash loop

            # Polling Wait
            # Break sleep if stop requested (optimized for long sleep intervals)
            if not ctx.should_stop():
                logger.debug(f"Waiting for {schedule_interval}s before next check...")

                # Simple sleep implementation
                # For more responsiveness during long sleep, split this into smaller chunks
                time.sleep(schedule_interval)

        logger.info("Daemon exited gracefully.")


if __name__ == "__main__":
    main()
