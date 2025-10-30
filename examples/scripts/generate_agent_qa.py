"""
Generate agent-style QA data from API specifications under ``data/api``.
Requires DASHSCOPE_API_KEY/DASHSCOPE_BASE_URL or explicit values.
"""

import json
import os
from pathlib import Path

from datamax import DataMax
from datamax.generator.auth import load_auth_configuration_from_env
from datamax.generator.agent_qa_generator import (
    AgentScriptSettings,
    build_agent_mode_config_for_spec,
    convert_episodes_to_sharegpt,
    discover_spec_files,
    load_agent_script_settings_from_env,
    make_agent_checkpoint_path,
    make_agent_output_stem,
    validate_auth_configuration_for_spec,
)

api_key = os.getenv("DASHSCOPE_API_KEY", "YOUR OWN KEY")
base_url = os.getenv("DASHSCOPE_BASE_URL", "YOUR BASE URL")
model = os.getenv("QA_MODEL", "YOUR QA MODEL")

root_dir = Path("/mnt/f/datamax")
spec_root = root_dir / "data" / "api"
output_root = root_dir / "train" / "agent"
checkpoint_root = output_root / "checkpoints"

settings: AgentScriptSettings = load_agent_script_settings_from_env(model)
auth_config = load_auth_configuration_from_env()

if auth_config is None:
    print(
        "[warn] 未检测到 AGENT_AUTH_CONFIG，受保护的 API 可能无法调用。",
        flush=True,
    )


def main() -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    spec_files = discover_spec_files(spec_root)
    if not spec_files:
        raise SystemExit(f"No API specifications found in {spec_root}")

    for spec_path in spec_files:
        try:
            checkpoint_path = make_agent_checkpoint_path(spec_path, spec_root, checkpoint_root)
            agent_mode = build_agent_mode_config_for_spec(
                spec_path,
                settings,
                checkpoint_path=checkpoint_path,
                auth_config=auth_config,
            )

            validate_auth_configuration_for_spec(spec_path, auth_config)

            dm = DataMax(file_path=str(spec_path), domain="API")
            spec_content = spec_path.read_text(encoding="utf-8")

            qa_payload = dm.get_pre_label(
                content=spec_content,
                api_key=api_key,
                base_url=base_url,
                model_name=model,
                question_number=settings.question_count,
                max_qps=settings.max_qps,
                structured_data=True,
                auto_self_review_mode=False,
                debug=settings.debug,
                agent_mode=agent_mode,
            )

            output_stem = make_agent_output_stem(spec_path, spec_root, output_root)

            if isinstance(qa_payload, dict) and "episodes" in qa_payload:
                episodes = qa_payload.get("episodes") or []
                tool_catalog = qa_payload.get("tool_catalog")
                sharegpt_payload = convert_episodes_to_sharegpt(episodes, tool_catalog)

                dm.save_label_data(sharegpt_payload, str(output_stem))

                metadata = {key: value for key, value in qa_payload.items() if key != "episodes"}
                metadata["format"] = "sharegpt"
                metadata["sharegpt_message_roles"] = ["human", "function_call", "observation", "gpt"]

                metadata_path = output_stem.with_name(f"{output_stem.name}_metadata.json")
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                dm.save_label_data(qa_payload, str(output_stem))

            print(f"[agent] {spec_path.name} -> {output_stem}.jsonl")
        except Exception as exc:  # pragma: no cover - sample script safeguard
            print(f"[error] Failed to generate agent QA for {spec_path}: {exc}")


if __name__ == "__main__":
    main()
