"""Non-sensitive counters for one page-oriented application run."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class OrchestrationMetrics:
    page_level_big_llm_calls: int = 0
    local_operator_calls: int = 0
    deterministic_actions: int = 0
    local_recoveries: int = 0
    fallback_calls: int = 0
    stale_element_events: int = 0
    validation_failures: int = 0
    retries: int = 0
    completed_fields: int = 0
    total_fields: int = 0
    pages_observed: int = 0

    @property
    def estimated_expensive_llm_calls_avoided(self) -> int:
        # Conservative estimate: one old full-agent step per successfully
        # completed field, less any full-agent fallback invocation.
        avoided = max(0, self.completed_fields - self.fallback_calls)
        return avoided

    @property
    def completion_rate(self) -> float | None:
        if not self.total_fields:
            return None
        return round(self.completed_fields / self.total_fields, 4)

    def as_dict(self) -> dict[str, int | float | None]:
        values = asdict(self)
        values["estimated_expensive_llm_calls_avoided"] = self.estimated_expensive_llm_calls_avoided
        values["completion_rate"] = self.completion_rate
        return values
