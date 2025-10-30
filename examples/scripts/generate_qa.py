"""
Generate QA pairs from text with domain tree labeling.
Requires DASHSCOPE_API_KEY/DASHSCOPE_BASE_URL or provide explicitly.
Set QA_INPUT_SOURCE=obs with OBS_* credentials to pull inputs from Huawei OBS.
"""

import os
from pathlib import Path

from datamax import DataMax
from datamax.loader.core import DataLoader

api_key = os.getenv("DASHSCOPE_API_KEY", "YOUR OWN KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL", "YOUR BASE URL")
model = os.getenv("QA_MODEL", "YOUR QA MODEL")
qa_input_source = os.getenv("QA_INPUT_SOURCE", "local").lower()
obs_endpoint = os.getenv("OBS_ENDPOINT")
obs_access_key = os.getenv("OBS_ACCESS_KEY_ID")
obs_secret_key = os.getenv("OBS_ACCESS_KEY_SECRET")
obs_bucket_name = os.getenv("OBS_BUCKET_NAME")
obs_download_dir_env = os.getenv("OBS_DOWNLOAD_DIR")
obs_prefix = os.getenv("OBS_PREFIX", "")

root_dir = Path(os.getenv("DATAMAX_ROOT", "/mnt/f/datamax"))
if not root_dir.is_absolute():
    root_dir = Path(__file__).resolve().parents[2] / root_dir

train_dir_name = "train"
local_dataset_dir = root_dir / "data" / "Step1"
default_obs_download_dir = root_dir / "obs_downloads"

save_parent_path = root_dir / train_dir_name


def discover_local_files() -> list[Path]:
    if not local_dataset_dir.exists():
        return []
    return sorted(path for path in local_dataset_dir.rglob("*") if path.is_file())


def download_files_from_obs() -> list[Path]:
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
        raise SystemExit(
            f"Missing OBS configuration for generate_qa: {', '.join(missing)}"
        )

    if obs_download_dir_env:
        download_dir = Path(obs_download_dir_env)
        if not download_dir.is_absolute():
            download_dir = root_dir / download_dir
    else:
        download_dir = default_obs_download_dir

    loader = DataLoader(
        endpoint=obs_endpoint,
        secret_key=obs_secret_key,
        access_key=obs_access_key,
        bucket_name=obs_bucket_name,
        source="obs",
    )
    loader.download_path = str(download_dir)
    files = loader.load_from_obs_source(obs_prefix)

    resolved_files = []
    for file_path in files:
        resolved_files.append(Path(file_path).resolve())
    return sorted(resolved_files)


def resolve_input_files() -> list[Path]:
    if qa_input_source == "obs":
        return download_files_from_obs()
    if qa_input_source in {"", "local"}:
        return discover_local_files()
    raise SystemExit(f"Unsupported QA_INPUT_SOURCE value: {qa_input_source}")


def main() -> None:
    save_parent_path.mkdir(parents=True, exist_ok=True)

    input_files = resolve_input_files()
    if not input_files:
        raise SystemExit("No input files found for QA generation.")

    for input_file in input_files:
        input_file = input_file.resolve()
        try:
            relative_path = input_file.relative_to(root_dir)
        except ValueError:
            relative_path = Path(input_file.name)

        relative_stem = relative_path.with_suffix("")
        save_dir = save_parent_path / relative_stem.parent
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{relative_stem.name}_train"

        dm = DataMax(file_path=str(input_file), to_markdown=True)
        data = dm.get_data()

        content = data.get("content")

        qa = dm.get_pre_label(
            content=content,
            api_key=api_key,
            base_url=base_url,
            model_name=model,
            question_number=50,  # question_number_per_chunk
            max_qps=100.0,
            debug=False,
            structured_data=True,  # enable structured output
            auto_self_review_mode=True,
            review_max_qps=100.0,
        )

        dm.save_label_data(qa, str(save_path))
        break


if __name__ == "__main__":
    main()

# nohup python examples/scripts/generate_qa.py > generate_qa.out 2>&1 & echo $! > generate_qa.pid
