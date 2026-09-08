"""
Multi-provider LLM factory.
Creates browser_use chat models from user settings.
Supports: OpenAI, Anthropic (direct), AWS Bedrock, Google Gemini.
"""

import json
import logging
import os
import re
from functools import wraps


logger = logging.getLogger(__name__)


def _debug_token_estimate(value: object) -> int:
    """Small, dependency-free diagnostic estimate; provider usage is logged separately."""
    return max(1, (len(str(value)) + 3) // 4) if value else 0


def _debug_message_text(message: object) -> tuple[str, int, int]:
    content = getattr(message, "content", "")
    parts = content if isinstance(content, list) else [content]
    text_parts = []
    screenshots = 0
    screenshot_chars = 0
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        else:
            image_url = getattr(part, "image_url", None)
            if image_url is not None:
                screenshots += 1
                screenshot_chars += len(getattr(image_url, "url", "") or "")
            text = getattr(part, "text", None)
            if text:
                text_parts.append(text)
    return "\n".join(text_parts), screenshots, screenshot_chars


def _log_llm_token_debug(messages, kwargs: dict) -> None:
    buckets = {name: 0 for name in ("system_task", "profile_qa", "resume", "browser_dom_state", "history_memory", "action_results")}
    message_sections = {}
    screenshot_count = 0
    screenshot_chars = 0
    section_tags = {
        "user_request": "system_task",
        "agent_history": "history_memory",
        "agent_state": "profile_qa",
        "browser_state": "browser_dom_state",
        "read_state": "action_results",
        "page_specific_actions": "system_task",
    }
    for message_index, message in enumerate(messages):
        text, images, image_chars = _debug_message_text(message)
        screenshot_count += images
        screenshot_chars += image_chars
        role = getattr(message, "role", "")
        parsed = []
        if role == "user":
            tag_pattern = re.compile(r"<(user_request|agent_history|agent_state|browser_state|read_state|page_specific_actions)>.*?</\1>", re.S)
            cursor = 0
            for match in tag_pattern.finditer(text):
                if match.start() > cursor:
                    parsed.append(("system_task", text[cursor:match.start()]))
                parsed.append((match.group(1), match.group(0)))
                cursor = match.end()
            if cursor < len(text):
                parsed.append(("system_task", text[cursor:]))
        if not parsed and role == "system":
            # These are the real sections emitted by build_memory_context().
            heading_pattern = re.compile(
                r"(?m)^(FULL CANDIDATE RESUME:|CANDIDATE PROFILE:|ANSWERING POLICY.*:|"
                r"SCREENING-QUESTION ANSWERING INSTRUCTIONS:|COUNTRY-SPECIFIC INSTRUCTIONS:|"
                r"Already applied.*:|SAVED Q&A .*?:|TRACKING INSTRUCTIONS:|SELF-LEARNING.*:|"
                r"[A-Z][A-Z -]+: )"
            )
            matches = list(heading_pattern.finditer(text))
            if matches and matches[0].start():
                parsed.append(("system_task", text[:matches[0].start()]))
            for index, match in enumerate(matches):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
                heading = match.group(1)
                section = "resume" if heading.startswith("FULL CANDIDATE RESUME") else "profile_qa" if heading.startswith(("CANDIDATE PROFILE", "SAVED Q&A")) else "system_task"
                parsed.append((section, text[match.start():end]))
        if not parsed:
            parsed = [("system_task", text)]
        for section_index, (section_name, section_text) in enumerate(parsed):
            bucket = section_tags.get(section_name, section_name)
            if bucket not in buckets:
                bucket = "system_task"
            # Browser Use stores action results inside each serialized <step>.
            if section_name == "agent_history" and "\nResult\n" in section_text:
                history_text, result_text = section_text.split("\nResult\n", 1)
                buckets["history_memory"] += len(history_text)
                buckets["action_results"] += len(result_text)
                message_sections[f"message_{message_index}.section_{section_index}.agent_history"] = {"history_memory_chars": len(history_text), "action_results_chars": len(result_text)}
                continue
            buckets[bucket] += len(section_text)
            message_sections[f"message_{message_index}.section_{section_index}.{section_name}"] = {"bucket": bucket, "text_chars": len(section_text)}
        if images:
            message_sections[f"message_{message_index}.screenshots"] = {"count": images, "payload_chars": image_chars}

    schema = kwargs.get("output_format")
    schema_text = ""
    if schema is not None:
        schema_builder = getattr(schema, "model_json_schema", None)
        schema_text = json.dumps(schema_builder(), sort_keys=True) if schema_builder else str(schema)
    tool_schema_chars = len(schema_text)
    logger.info(
        "LLM_TOKEN_DEBUG input_text_chars_total=%d input_text_chars_by_bucket=%s message_sections=%s tool_schema_chars=%d screenshot_count=%d screenshot_payload_chars=%d",
        sum(buckets.values()),
        buckets,
        message_sections,
        tool_schema_chars,
        screenshot_count,
        screenshot_chars,
    )


def debug_llm_calls(llm):
    """Opt-in logging around the model call, without changing the model or payload."""
    if not os.getenv("LLM_TOKEN_DEBUG") or getattr(llm, "_langhire_token_debug", False):
        return llm
    original_ainvoke = llm.ainvoke

    @wraps(original_ainvoke)
    async def logged_ainvoke(messages, *args, **kwargs):
        _log_llm_token_debug(messages, kwargs)
        response = await original_ainvoke(messages, *args, **kwargs)
        usage = getattr(response, "usage", None)
        logger.info(
            "LLM_TOKEN_DEBUG provider_input_tokens=%s provider_cached_input_tokens=%s provider_image_input_tokens=%s provider_total_tokens=%s",
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "prompt_cached_tokens", None),
            getattr(usage, "prompt_image_tokens", None),
            getattr(usage, "total_tokens", None),
        )
        return response

    llm.ainvoke = logged_ainvoke
    llm._langhire_token_debug = True
    return llm


class _PatchedSession:
    """Wraps a boto3.Session to inject a BotoConfig into every client() call."""

    def __init__(self, session, config):
        self._session = session
        self._config = config

    def client(self, service_name, **kwargs):
        kwargs.setdefault("config", self._config)
        return self._session.client(service_name, **kwargs)

    def __getattr__(self, name):
        return getattr(self._session, name)


def _debug_factory(factory):
    @wraps(factory)
    def wrapped(*args, **kwargs):
        return debug_llm_calls(factory(*args, **kwargs))
    return wrapped


@_debug_factory
def create_llm(settings: dict, session_id: str | None = None):
    """Create a browser_use BaseChatModel from settings dict."""
    provider = settings.get("provider", "openai")

    if provider == "openai":
        from browser_use.llm import ChatOpenAI
        cfg = settings.get("openai", {})
        return ChatOpenAI(
            model=cfg.get("model", "gpt-4o"),
            api_key=cfg.get("api_key", "").strip(),
        )

    elif provider == "anthropic":
        from browser_use.llm import ChatAnthropic
        cfg = settings.get("anthropic", {})
        return ChatAnthropic(
            model=cfg.get("model", "claude-sonnet-4-5"),
            api_key=cfg.get("api_key", "").strip(),
        )

    elif provider == "bedrock":
        from browser_use.llm import ChatAWSBedrock
        import boto3
        from botocore.config import Config as BotoConfig
        cfg = settings.get("bedrock", {})
        region = cfg.get("region", "us-west-2")
        model = cfg.get("model", "us.anthropic.claude-sonnet-4-6")
        auth_mode = cfg.get("auth_mode", "profile")

        if auth_mode == "keys" and cfg.get("access_key") and cfg.get("secret_key"):
            session = boto3.Session(
                aws_access_key_id=cfg["access_key"],
                aws_secret_access_key=cfg["secret_key"],
                region_name=region,
            )
        else:
            session = boto3.Session(
                profile_name=cfg.get("profile_name", "default"),
                region_name=region,
            )

        # Disable request compression to avoid "Error -3 decompressing"
        # with Bedrock's converse API
        boto_config = BotoConfig(disable_request_compression=True)
        patched_session = _PatchedSession(session, boto_config)
        return ChatAWSBedrock(model=model, session=patched_session)

    elif provider == "ollama":
        # Use browser-use's native ChatOllama (not ChatOpenAI against the /v1 shim).
        # ChatOllama drives Ollama's native `format=<schema>` grammar-constrained
        # decoding, which small local models (e.g. gemma3:4b) honor reliably —
        # the OpenAI-compatible `response_format` path produced truncated JSON and
        # crashed with "Invalid JSON: EOF while parsing" (see issue #60).
        from browser_use.llm import ChatOllama
        cfg = settings.get("ollama", {})
        base_url = cfg.get("base_url", "http://localhost:11434").rstrip("/")
        return ChatOllama(
            model=cfg.get("model") or "llama3.1",
            host=base_url,  # native Ollama host (NOT the /v1 OpenAI shim)
            timeout=300,
            ollama_options={
                "num_ctx": 32768,    # large context: agent prompt + DOM/screenshot is big
                "num_predict": 4096,  # room for full JSON output (avoids truncation)
                "temperature": 0.0,
            },
        )

    elif provider == "gemini":
        from browser_use.llm import ChatGoogle
        cfg = settings.get("gemini", {})
        return ChatGoogle(
            model=cfg.get("model", "gemini-2.5-pro"),
            api_key=cfg.get("api_key", "").strip(),
        )

    elif provider == "openrouter":
        from browser_use.llm import ChatOpenAI
        cfg = settings.get("openrouter", {})
        return ChatOpenAI(
            model=cfg.get("model", "openai/gpt-4o"),
            api_key=cfg.get("api_key", "").strip(),
            base_url="https://openrouter.ai/api/v1",
            default_headers={"x-session-id": session_id} if session_id else None,
        )

    elif provider == "openai_compatible":
        from browser_use.llm import ChatOpenAI
        cfg = settings.get("openai_compatible", {})
        return ChatOpenAI(
            model=cfg.get("model", "default"),
            api_key=cfg.get("api_key") or "not-needed",
            base_url=cfg.get("base_url"),
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def test_connection(llm) -> str:
    """Send a test message and return the response."""
    import asyncio
    import zlib
    from browser_use.llm.messages import UserMessage
    from browser_use.llm.exceptions import ModelProviderError
    try:
        # Increase timeout to 60s for local models
        response = await asyncio.wait_for(
            llm.ainvoke([UserMessage(content="Say 'hello' in one word.")]),
            timeout=60,
        )
        return f"Model responded: {response.completion}"
    except (asyncio.TimeoutError, TimeoutError):
        raise RuntimeError("LLM test timed out after 60 seconds. Your model might be slow to load or the server is busy.")
    except zlib.error:
        raise RuntimeError(
            "Decompression error communicating with AWS Bedrock. "
            "This is often caused by a corporate proxy or VPN. "
            "Try disabling your VPN or proxy, or use a direct connection."
        )
    except ModelProviderError as e:
        raise RuntimeError(_friendly_llm_error(getattr(e, "message", str(e)), getattr(e, "status_code", None)))
    except Exception as e:
        raise RuntimeError(_friendly_llm_error(str(e), getattr(e, "status_code", None)))


def _friendly_llm_error(message: str, status_code=None) -> str:
    """Map an LLM provider failure to a friendly, actionable message."""
    msg = (message or "").lower()
    code = status_code

    # Authentication / invalid API key
    if code in (401, 403) or any(
        s in msg for s in ("api_key", "api key", "authentication", "unauthorized", "invalid api", "invalid key")
    ):
        return "Invalid API key. Please check your key and try again."

    # Rate limit / quota
    if code == 429 or any(s in msg for s in ("rate limit", "ratelimit", "quota", "too many requests")):
        return "Rate limit or quota exceeded. Wait a moment and try again."

    # Connection / network
    if any(
        s in msg
        for s in ("connection", "connect", "network", "timed out", "timeout", "name resolution",
                  "could not resolve", "dns", "unreachable", "refused")
    ):
        return "Cannot reach the API. Check your internet connection and base URL."

    # Fall back to the raw provider message so the user still sees something useful.
    return message or "LLM request failed."
