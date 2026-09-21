"""Environment switches for the reversible page-oriented apply path."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


def _enabled(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off", ""}:
        return False
    return default


@dataclass(frozen=True)
class ApplyFeatureFlags:
    page_orchestrator: bool = False
    local_browser_operator: bool = False
    full_agent_fallback: bool = True

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ApplyFeatureFlags":
        env = environ if environ is not None else os.environ
        return cls(
            page_orchestrator=_enabled(env.get("ENABLE_PAGE_ORCHESTRATOR"), False),
            local_browser_operator=_enabled(env.get("ENABLE_LOCAL_BROWSER_OPERATOR"), False),
            full_agent_fallback=_enabled(env.get("ENABLE_FULL_AGENT_FALLBACK"), True),
        )
