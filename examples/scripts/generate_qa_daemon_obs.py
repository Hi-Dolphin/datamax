"""
Generate QA pairs from text with domain tree labeling.
Requires DASHSCOPE_API_KEY/DASHSCOPE_BASE_URL or provide explicitly.
Set QA_INPUT_SOURCE=oss with OSS_* credentials to pull inputs from Alibaba Cloud OSS.
Support scheduled execution via QA_SCHEDULE_INTERVAL (seconds).
"""

import os
import time
import urllib.parse
from pathlib import Path
from typing import List, Optional

from loguru import logger

from datamax import DataMax
from datamax.generator.types import PersistenceConfig
from datamax.loader.core import DataLoader
from datamax.persistence.postgres import PostgresPersistence
from obs import ObsClient

# -----------------------------------------
#                Env
# -----------------------------------------
api_key = os.getenv("DASHSCOPE_API_KEY", "YOUR OWN KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL", "YOUR BASE URL")
model = os.getenv("QA_MODEL", "YOUR QA MODEL")
qa_input_source = os.getenv("QA_INPUT_SOURCE", "local").lower()

# OBS (Huawei Cloud)
obs_endpoint = os.getenv("OBS_ENDPOINT")
obs_access_key = os.getenv("OBS_ACCESS_KEY_ID")
obs_secret_key = os.getenv("OBS_ACCESS_KEY_SECRET")
obs_bucket_name = os.getenv("OBS_BUCKET_NAME")
obs_prefix = os.getenv("OBS_PREFIX", "")
obs_download_dir_env = os.getenv("OBS_DOWNLOAD_DIR")

# Global Paths
root_dir = Path(os.getenv("DATAMAX_ROOT")) if os.getenv("DATAMAX_ROOT") else Path.cwd()
train_dir_name = "train"
local_dataset_dir = root_dir / "data" / "数据集"
default_oss_download_dir = root_dir / "oss_downloads"
save_parent_path = root_dir / train_dir_name

question_number = int(os.getenv("QA_QUESTION_NUMBER", 50))
chunk_size = int(os.getenv("QA_CHUNK_SIZE", 3000))
chunk_overlap = int(os.getenv("QA_CHUNK_OVERLAP", 1500))
max_qps = int(os.getenv("QA_MAX_QPS", 100))
schedule_interval = int(os.getenv("QA_SCHEDULE_INTERVAL", 0))


def build_persistence_config() -> dict | None:
    prefix = os.getenv("QA_DB_PREFIX", "POSTGRES")
    dsn = os.getenv("QA_DB_DSN") or os.getenv("DATABASE_URL")
    if not dsn:
        host = os.getenv(f"{prefix}_HOST")
        port = os.getenv(f"{prefix}_PORT")
        user = os.getenv(f"{prefix}_USERNAME")
        password = os.getenv(f"{prefix}_PASSWORD")
        database = os.getenv(f"{prefix}_DB")
        schema = os.getenv(f"{prefix}_SCHEMA")
        connect = os.getenv(f"{prefix}_CONNECT", "postgresql")
        if not (host and user and database):
            return None
        port_part = f":{port}" if port else ""
        schema_part = f"?options=-c%20search_path%3D{schema}" if schema else ""
        username = urllib.parse.quote(user, safe="")
        password_part = ""
        if password is not None:
            encoded_password = urllib.parse.quote(password, safe="")
            password_part = f":{encoded_password}"
        dsn = f"{connect}://{username}{password_part}@{host}{port_part}/{database}{schema_part}"

    source_key = os.getenv("QA_SOURCE_KEY", "local_corpus")
    if not source_key:
        return None

    config: dict[str, object] = {
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
    # Attach tags
    if question_number:
        config["question_number"] = question_number
    if chunk_size:
        config["chunk_size"] = chunk_size
    if chunk_overlap:
        config["chunk_overlap"] = chunk_overlap
    if max_qps:
        config["max_qps"] = max_qps

    return config


persistence_config = build_persistence_config()


def discover_local_files() -> list[Path]:
    if not local_dataset_dir.exists():
        return []
    return sorted(path for path in local_dataset_dir.rglob("*") if path.is_file())


def get_obs_loader() -> ObsClient:
    missing = [
        name
        for name, value in {
            "OBS_ENDPOINT": obs_endpoint,
            "OBS_ACCESS_KEY_ID": obs_access_key,
            "OBS_ACCESS_KEY_SECRET": obs_secret_key,
            "OBS_BUCKET_NAME": obs_bucket_name,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing OBS configuration for generate_qa: {', '.join(missing)}")

    obs_client = ObsClient(
        access_key_id=obs_access_key,
        secret_access_key=obs_secret_key,
        server=obs_endpoint,
    )
    return obs_client


def get_download_dir() -> Path:
    if obs_download_dir_env:
        download_dir = Path(obs_download_dir_env)
        if not download_dir.is_absolute():
            download_dir = root_dir / download_dir
    else:
        download_dir = default_oss_download_dir
    download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir


def fetch_and_filter_obs_files(loader: ObsClient) -> List[str]:
    """
    List all files in OBS and filter out those already processed.
    Returns a list of OBS keys (file paths) to be processed.
    """
    # 1. List all objects
    logger.info(f"Listing objects in bucket '{obs_bucket_name}' with prefix '{obs_prefix}'...")
    resp = loader.listObjects(bucketName=obs_bucket_name, prefix=obs_prefix)
    all_keys = [obj['key'] for obj in resp.get('contents', [])] if resp.get('status') < 300 else []
    if not all_keys:
        logger.info("No files found in OBS.")
        return []

    # 2. If no DB configured, return all (re-process everything or user must handle manually)
    if not persistence_config:
        logger.warning("No persistence config found. Skipping deduplication (will process ALL files).")
        return all_keys

    # 3. Check DB for status
    try:
        p_config = PersistenceConfig.from_mapping(persistence_config)
        with PostgresPersistence(p_config) as p:
            # status_map: {file_path: status}
            status_map = p.check_files_status(p_config.source_key, all_keys)
    except Exception as e:
        logger.error(f"Failed to check file status from DB: {e}")
        # On DB failure, return empty list to avoid flooding
        return []

    # 4. Filter logic: Skip 'COMPLETED' and 'PROCESSING'
    new_keys = []
    for key in all_keys:
        status = status_map.get(key)
        if status in ("COMPLETED", "PROCESSING"):
            # Skip
            continue
        # Pending, Failed, or None (New) -> Process
        new_keys.append(key)

    logger.info(
        f"Found {len(all_keys)} files. {len(all_keys) - len(new_keys)} processed/processing. {len(new_keys)} new/failed to process."
    )
    return new_keys


def update_file_status(obs_key: str, status: str, error_message: Optional[str] = None) -> None:
    """Helper to update file status in DB safely."""
    if not obs_key or not persistence_config:
        return

    try:
        p_config = PersistenceConfig.from_mapping(persistence_config)
        with PostgresPersistence(p_config) as p:
            p.upsert_file_status(p_config.source_key, obs_key, status, error_message=error_message)
    except Exception as e:
        logger.error(f"Failed to update DB status for {obs_key}: {e}")


def _generate_and_save_qa(input_file: Path, save_path: Path, relative_path: Path) -> Optional[str]:
    """
    Core logic to generate and save QA.
    Returns error message if skipped/failed, None if success.
    """
    dm = DataMax(file_path=str(input_file), to_markdown=True)
    data = dm.get_data()

    content = data.get("content")
    if isinstance(content, list):
        content = "\n\n".join(text for text in content if text)

    if not content or not content.strip():
        logger.warning(f"[skip] Parsed content empty for {relative_path}, skipping QA generation.")
        return "Empty content"

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
        logger.warning(f"[skip] No QA pairs generated for {relative_path}, skipping save.")
        return "No QA generated"

    dm.save_label_data(qa, str(save_path))
    return None


def process_file(input_file: Path, obs_key: Optional[str] = None) -> bool:
    """
    Process a single file.
    :param input_file: Local path to the file.
    :param oss_key: Original OSS key if applicable (for status tracking).
    :return: True if successful, False otherwise.
    """
    relative_path = "unknown"
    try:
        input_file = input_file.resolve()
        try:
            relative_path = input_file.relative_to(root_dir)
        except ValueError:
            relative_path = Path(input_file.name)

        # Mark as PROCESSING
        if obs_key:
            update_file_status(obs_key, "PROCESSING")

        relative_stem = relative_path.with_suffix("")
        save_dir = save_parent_path / relative_stem.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{relative_stem.name}_train"

        logger.info(f"Processing file: {relative_path}")

        error_msg = _generate_and_save_qa(input_file, save_path, relative_path)

        if error_msg:
            if obs_key:
                update_file_status(obs_key, "COMPLETED", error_message=error_msg)
            return True

        # Mark as COMPLETED
        if obs_key:
            update_file_status(obs_key, "COMPLETED")

        return True

    except Exception as e:
        logger.error(f"Error processing {relative_path}: {e}")
        # Mark as FAILED
        if obs_key:
            update_file_status(obs_key, "FAILED", error_message=str(e))
        return False


def run_once() -> None:
    """
    Execute a single run of discovery -> download -> processing.
    """
    save_parent_path.mkdir(parents=True, exist_ok=True)

    if qa_input_source == "obs":
        loader = get_obs_loader()
        download_dir = get_download_dir()

        # 1. Filter
        target_keys = fetch_and_filter_obs_files(loader)

        # 2. Download & Process loop
        for key in target_keys:
            try:
                logger.info(f"Downloading {key}...")
                # We construct local path
                # Note: obs_key might contain slashes
                local_file_name = key.split("/")[-1]
                
                # Just stick to flat or original simple logic for now to allow resolution
                local_file_path = download_dir / local_file_name

                resp = loader.getObject(bucketName=obs_bucket_name, objectKey=key, downloadPath=str(local_file_path))
                if resp.status < 300:
                    process_file(local_file_path, obs_key=key)
                else:
                    logger.error(f"Failed to download {key}: {resp.errorMessage}")

                # Optional: cleanup file after processing to save space?
                # os.remove(local_file_path)
            except Exception as e:
                logger.error(f"Failed loop for {key}: {e}")

    elif qa_input_source in {"", "local"}:
        # Local mode doesn't use status table deduplication same way usually,
        # but could extend if we used file hash or path as key.
        # For now, keep original behavior: process all local files found.
        files = discover_local_files()
        if not files:
            logger.warning("No local files found.")
            return

        for f in files:
            process_file(f, oss_key=None)

    else:
        logger.error(f"Unsupported QA_INPUT_SOURCE value: {qa_input_source}")


def main() -> None:
    if schedule_interval > 0:
        logger.info(f"Starting scheduler loop with interval {schedule_interval}s")
        while True:
            try:
                logger.info("Starting scheduled run...")
                run_once()
                logger.info("Scheduled run finished.")
            except Exception as e:
                logger.error(f"Scheduler run crashed: {e}")

            logger.info(f"Sleeping for {schedule_interval}s...")
            time.sleep(schedule_interval)
    else:
        run_once()


if __name__ == "__main__":
    main()
