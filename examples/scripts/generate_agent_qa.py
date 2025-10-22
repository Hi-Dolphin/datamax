# 说明：生成符合 DataMax 自主代理模式的问答语料，读取 data/api 下的 OpenAPI 规范并产出 JSONL。
"""Generate agent-style QA data from API specifications in ``data/api``.

The script expects DataMax-compatible API credentials to be available in
environment variables (see constants below) and will emit JSONL files under
``train/agent`` mirroring the spec directory structure.
"""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from datamax import DataMax
from datamax.generator.auth import AuthManager

# 布尔型环境变量兼容的真值 / 假值集合，方便与 shell 约定对齐
BOOL_TRUE = {"1", "true", "yes", "on"}
BOOL_FALSE = {"0", "false", "no", "off"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    # 从环境变量读取整数，缺失或格式不合法时返回默认值
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    # 支持浮点配置，例如 QPS、超时时间等
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in BOOL_TRUE:
        return True
    if normalized in BOOL_FALSE:
        return False
    return default


def load_auth_configuration() -> Optional[dict]:
    """Load authentication configuration from environment variables.

    配置方式：
    1. 直接设置环境变量 `AGENT_AUTH_CONFIG` 为 JSON 字符串，例如：
       ```json
       {
         "default": "primary",
         "providers": {
           "primary": {
             "type": "oauth_client_credentials",
             "client_id": "your-client-id",
             "client_secret": "your-client-secret",
             "token_url": "https://auth.example.com/oauth/token",
             "scopes": ["read:data", "write:data"]
           },
           "internal-basic": {
             "type": "basic_auth",
             "username": "api-user",
             "password": "secure-password"
           },
           "public-key": {
             "type": "url_auth_key",
             "value": "your-public-key",
             "param_name": "auth_key",
             "location": "query"
           }
         }
       }
       ```

    2. 或者通过 `AGENT_AUTH_CONFIG_PATH` 指向含有以上 JSON 的文件。

    `AuthManager` 支持的 provider 类型包括：
      - oauth_client_credentials：OAuth2 客户端模式，支持 tenant_id、audience、extra_params 等扩展字段。
      - url_auth_key：将 `value` 注入指定 header/query 参数，可配置 param_name/location/header_name。
      - basic_auth：标准 Basic 认证，需要 username / password。
    """
    # 优先读取内联 JSON 配置，其次读取外部文件
    raw_json = os.getenv("AGENT_AUTH_CONFIG")
    config_path = os.getenv("AGENT_AUTH_CONFIG_PATH")

    if raw_json:
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON provided via AGENT_AUTH_CONFIG: {exc}") from exc

    if config_path:
        path = Path(config_path).expanduser()
        if not path.exists():
            raise SystemExit(f"Auth configuration file {path} not found.")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON inside auth configuration file {path}: {exc}") from exc

    return None


def validate_auth_for_spec(spec_path: Path, auth_config: Optional[dict]) -> None:
    """Ensure provided auth config覆盖规范定义的安全方案."""
    if not auth_config:
        return
    # 延迟导入，避免脚本在无需校验时引入重量级依赖
    from datamax.generator.agent_qa_generator import ApiSpecLoader

    loader = ApiSpecLoader()
    specs = loader.load([str(spec_path)])
    if not specs:
        return

    manager = AuthManager(auth_config)
    for spec in specs:
        for endpoint in spec.endpoints:
            tool_spec = endpoint.to_tool_spec()
            # 若鉴权缺失，AuthManager 会抛出 RuntimeError，提前暴露问题
            manager.get_context(tool_spec)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIRECTORY = Path(os.getenv("AGENT_SPEC_DIR", str(PROJECT_ROOT / "data" / "api"))).expanduser()
OUTPUT_ROOT = Path(os.getenv("AGENT_OUTPUT_DIR", str(PROJECT_ROOT / "train" / "agent"))).expanduser()
CHECKPOINT_ROOT = Path(os.getenv("AGENT_CHECKPOINT_DIR", str(OUTPUT_ROOT / "checkpoints"))).expanduser()
# agent 工具调用鉴权配置：支持 OAuth、URL key、Basic 等多种方案
AUTH_CONFIGURATION = load_auth_configuration()
DEFAULT_TOOL_SERVER = os.getenv("AGENT_DEFAULT_TOOL_SERVER")
TOOL_REQUEST_TIMEOUT = env_float("AGENT_TOOL_TIMEOUT", 30.0)
REQUIRE_AUTH = env_bool("AGENT_REQUIRE_AUTH", True)

API_KEY = os.getenv("DASHSCOPE_API_KEY", "YOUR OWN KEY")
BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "YOUR BASE URL")
MODEL_NAME = os.getenv("QA_MODEL", "YOUR QA MODEL")

if AUTH_CONFIGURATION is None:
    # 未提供鉴权配置时提前警告，同时提示示例结构
    print(
        "[warn] 未检测到 AGENT_AUTH_CONFIG，受保护的 API 可能无法调用。\n"
        "       请参考脚本顶部 docstring 提供的 JSON 模板配置鉴权信息。",
        flush=True,
    )

# 代理相关模型与后端配置，可按需切换到 OpenAI、LangGraph 等实现
AGENT_BACKEND = (os.getenv("AGENT_BACKEND", "openai") or "openai").strip().lower() or "langgraph"
AGENT_QUESTION_MODEL = os.getenv("AGENT_QUESTION_MODEL") or "qwen3-max"
AGENT_CLASSIFY_MODEL = os.getenv("AGENT_CLASSIFY_MODEL") or "qwen3-max"
AGENT_ANSWER_MODEL = os.getenv("AGENT_ANSWER_MODEL") or "qwen3-max"
AGENT_REVIEW_MODEL = os.getenv("AGENT_REVIEW_MODEL") or "qwen3-max"

# 核心参数：问题数量、节流策略及并发控制，默认值可通过环境变量覆盖
QUESTION_COUNT = max(1, env_int("AGENT_QUESTION_COUNT", 20))
MAX_QPS = max(0.0, env_float("AGENT_MAX_QPS", 10.0))
TOP_K_TOOLS = max(1, env_int("AGENT_TOP_K_TOOLS", 5))
MAX_TURNS = max(1, env_int("AGENT_MAX_TURNS", 8))
MAX_WORKERS = max(1, env_int("AGENT_MAX_WORKERS", 4))
MAX_RETRIES = max(1, env_int("AGENT_MAX_RETRIES", 3))
MAX_QUESTIONS_PER_CONTEXT = max(1, env_int("AGENT_MAX_QUESTIONS_PER_CONTEXT", 4))
LANGGRAPH_RETRY = max(0, env_int("AGENT_LANGGRAPH_RETRY", 1))
RESUME_FROM_CHECKPOINT = env_bool("AGENT_RESUME_FROM_CHECKPOINT", True)
DEBUG_MODE = env_bool("AGENT_DEBUG", True)

_min_interval_override = os.getenv("AGENT_MIN_REQUEST_INTERVAL")
if _min_interval_override and _min_interval_override.strip():
    try:
        MIN_REQUEST_INTERVAL: float | None = float(_min_interval_override)
    except ValueError:
        MIN_REQUEST_INTERVAL = None
else:
    MIN_REQUEST_INTERVAL = None

# 若未显式配置最小请求间隔，则根据 QPS 自动推算节流时间
if MIN_REQUEST_INTERVAL is None and MAX_QPS > 0:
    MIN_REQUEST_INTERVAL = 1.0 / MAX_QPS

# OpenAPI 规范支持的文件扩展名，搜索目录时仅匹配这些类型
SPEC_EXTENSIONS = (".json", ".yaml", ".yml")


def ensure_text(value: Any) -> str:
    # 统一将任意类型的字段转换为字符串，避免 ShareGPT 导出时出现类型不匹配
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)

def format_function_call_payload(tool_name: str, arguments: Any) -> str:
    # ShareGPT 结构约定：function_call 节点必须存储 JSON 字符串
    payload = {"name": tool_name, "arguments": arguments}
    try:
        return json.dumps(payload, ensure_ascii=False)
    except TypeError:
        return json.dumps({"name": tool_name, "arguments": None}, ensure_ascii=False)

def episode_to_sharegpt(
    episode: Dict[str, Any],
    tool_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    # conversations 按照 ShareGPT 角色顺序组织：提问、工具调用、观测、回答
    conversations: List[Dict[str, str]] = []
    question = episode.get("question") or {}

    prompt = ensure_text(
        # 优先取结构化 prompt，其次回退到原始用户轮次
        question.get("prompt")
        or question.get("question")
        or next(
            (
                turn.get("content")
                for turn in (episode.get("turns") or [])
                if isinstance(turn, dict) and turn.get("role") == "user" and turn.get("content")
            ),
            "",
        )
    )
    conversations.append({"from": "human", "value": prompt})

    used_tools: List[str] = []
    for call in episode.get("tool_calls") or []:
        tool_name = ensure_text(call.get("tool_name") or "tool")
        conversations.append(
            {
                "from": "function_call",
                "value": format_function_call_payload(tool_name, call.get("input")),
            }
        )
        conversations.append(
            {
                "from": "observation",
                "value": ensure_text(call.get("observation")),
            }
        )
        if tool_name not in used_tools:
            used_tools.append(tool_name)

    final_answer = ensure_text(episode.get("final_answer"))
    if not final_answer:
        fallback = next(
            (
                turn.get("content")
                for turn in reversed(episode.get("turns") or [])
                if isinstance(turn, dict)
                and turn.get("role") == "assistant"
                and turn.get("content")
                and not turn.get("tool_name")
            ),
            "",
        )
        final_answer = ensure_text(fallback)

    conversations.append({"from": "gpt", "value": final_answer})

    sharegpt_entry: Dict[str, Any] = {"conversations": conversations}
    if used_tools:
        # 按照工具名称回填 OpenAPI 元数据，方便后续评估或回放
        tool_specs = [tool_lookup[name] for name in used_tools if name in tool_lookup]
        if tool_specs:
            sharegpt_entry["tools"] = tool_specs

    metadata = episode.get("metadata")
    backend = metadata.get("agent_backend") if isinstance(metadata, dict) else None
    if backend:
        # system 字段记录代理执行后端，输出数据时保留上下文信息
        sharegpt_entry["system"] = f"Agent backend: {backend}"

    return sharegpt_entry


def convert_episodes_to_sharegpt(
    episodes: Sequence[Dict[str, Any]],
    tool_catalog: Iterable[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    # 将全部 episode 映射至 ShareGPT 列表，同时构建工具名 -> 元数据索引
    tool_lookup: Dict[str, Dict[str, Any]] = {}
    if tool_catalog:
        for tool in tool_catalog:
            if isinstance(tool, dict) and tool.get("name"):
                tool_lookup[str(tool["name"])] = tool
    return [episode_to_sharegpt(ep, tool_lookup) for ep in episodes if isinstance(ep, dict)]


def discover_spec_files(directory: Path) -> list[Path]:
    # 递归搜索 OpenAPI 规范文件，返回按相对路径排序的列表
    if not directory.exists():
        return []

    files: dict[Path, Path] = {}
    for ext in SPEC_EXTENSIONS:
        for path in directory.rglob(f"*{ext}"):
            if path.is_file():
                # 使用相对路径去重，避免同名文件被覆盖
                relative = try_relative(path, directory)
                files.setdefault(relative, path)

    return [files[key] for key in sorted(files)]


def try_relative(path: Path, base: Path) -> Path:
    # 尝试转换为 base 目录下的相对路径，失败则返回文件名
    try:
        return path.relative_to(base)
    except ValueError:
        return Path(path.name)


def remove_all_suffixes(path: Path) -> Path:
    # 逐层剥离所有扩展名，便于在输出目录中复用统一的 stem
    result = path
    while result.suffix:
        try:
            result = result.with_suffix("")
        except ValueError:
            break
    return result


def split_relative_path(spec_path: Path) -> tuple[Path, str]:
    # 将规格文件拆分成相对目录与文件干名，便于组织输出路径
    relative = try_relative(spec_path, SPEC_DIRECTORY)
    base = remove_all_suffixes(relative)
    return base.parent, base.name


def make_output_stem(spec_path: Path) -> Path:
    # 生成输出文件的基础路径（不含扩展名），保持与规范目录结构一致
    relative_dir, stem = split_relative_path(spec_path)
    output_dir = OUTPUT_ROOT / relative_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / stem


def make_checkpoint_path(spec_path: Path) -> Path:
    # 构建用于断点续跑的检查点文件路径
    relative_dir, stem = split_relative_path(spec_path)
    checkpoint_dir = CHECKPOINT_ROOT / relative_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{stem}.jsonl"


def build_agent_config(spec_path: Path) -> dict[str, object]:
    # 按单个规范文件生成 agent 模式配置，交由 DataMax SDK 调用
    config: dict[str, object] = {
        "enabled": True,
        "agent_backend": AGENT_BACKEND,
        "spec_sources": [str(spec_path)],
        "question_count": QUESTION_COUNT,
        "max_questions_per_context": MAX_QUESTIONS_PER_CONTEXT,
        "top_k_tools": TOP_K_TOOLS,
        "max_turns": MAX_TURNS,
        "max_workers": MAX_WORKERS,
        "max_retries": MAX_RETRIES,
        "langgraph_retry": LANGGRAPH_RETRY,
        "resume_from_checkpoint": RESUME_FROM_CHECKPOINT,
        "checkpoint_path": str(make_checkpoint_path(spec_path)),
    }

    if MIN_REQUEST_INTERVAL and MIN_REQUEST_INTERVAL > 0:
        config["min_request_interval_seconds"] = MIN_REQUEST_INTERVAL

    if AGENT_QUESTION_MODEL:
        config["agent_question_generate_model"] = AGENT_QUESTION_MODEL
    if AGENT_CLASSIFY_MODEL:
        config["classify_model"] = AGENT_CLASSIFY_MODEL
    if AGENT_ANSWER_MODEL:
        config["core_agent_answer_generate_model"] = AGENT_ANSWER_MODEL
    if AGENT_REVIEW_MODEL:
        config["review_model"] = AGENT_REVIEW_MODEL

    if AUTH_CONFIGURATION:
        # 深拷贝一份鉴权配置，避免运行时被下游修改
        config["auth"] = json.loads(json.dumps(AUTH_CONFIGURATION))
    if DEFAULT_TOOL_SERVER:
        config["default_tool_server"] = DEFAULT_TOOL_SERVER
    if TOOL_REQUEST_TIMEOUT is not None:
        config["tool_request_timeout"] = TOOL_REQUEST_TIMEOUT
    config["require_auth_for_protected_tools"] = REQUIRE_AUTH

    return config


def generate_for_spec(spec_path: Path) -> None:
    # 针对单个规范文件执行完整的 agent QA 生成流程
    agent_config = build_agent_config(spec_path)
    output_stem = make_output_stem(spec_path)
    spec_content = spec_path.read_text(encoding="utf-8")

    # 预检查鉴权配置是否覆盖规范中的安全定义，减少运行中断
    validate_auth_for_spec(spec_path, AUTH_CONFIGURATION)

    dm = DataMax(file_path=str(spec_path), domain="API")
    # 调用 DataMax SDK，触发模型生成 + 工具调用 + 自我评审
    qa_payload = dm.get_pre_label(
        content=spec_content,
        api_key=API_KEY,
        base_url=BASE_URL,
        model_name=MODEL_NAME,
        question_number=QUESTION_COUNT,
        max_qps=MAX_QPS,
        structured_data=True,
        auto_self_review_mode=False,
        debug=DEBUG_MODE,
        agent_mode=agent_config,
    )

    if isinstance(qa_payload, dict) and "episodes" in qa_payload:
        # agent 模式产出结构化 episode 数据，需要转换成 ShareGPT
        episodes = qa_payload.get("episodes") or []
        tool_catalog = qa_payload.get("tool_catalog")
        sharegpt_payload = convert_episodes_to_sharegpt(episodes, tool_catalog)

        dm.save_label_data(sharegpt_payload, str(output_stem))

        metadata: dict[str, object] = {
            key: value for key, value in qa_payload.items() if key != "episodes"
        }
        metadata["format"] = "sharegpt"
        metadata["sharegpt_message_roles"] = ["human", "function_call", "observation", "gpt"]

        metadata_path = output_stem.with_name(output_stem.name + "_metadata.json")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        # 非结构化返回场景（如错误信息）直接落盘
        dm.save_label_data(qa_payload, str(output_stem))

    print(f"[agent] {spec_path.name} -> {output_stem}.jsonl")


def main() -> None:
    # 准备输出与检查点目录，确保后续写入不会失败
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    spec_files = discover_spec_files(SPEC_DIRECTORY)
    if not spec_files:
        raise SystemExit(f"No API specifications found in {SPEC_DIRECTORY}")

    for spec_path in spec_files:
        try:
            # 逐个处理规范文件，异常时打印告警继续下一个
            generate_for_spec(spec_path)
        except Exception as exc:  # pragma: no cover - sample script safeguard
            print(f"[error] Failed to generate agent QA for {spec_path}: {exc}")


if __name__ == "__main__":
    main()
