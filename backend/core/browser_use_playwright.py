"""Verified Playwright hands for Browser Use's existing planner.

Browser Use owns observation, planning, prompts, and native CDP recovery.
This module owns primary interaction, verification, batching, and telemetry.
"""

from __future__ import annotations

import json
import logging
import re
from urllib.parse import urlsplit
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
MAX_ATTEMPTS = 2
ACTION_WAIT_MS = 350

_SUBMISSION_TEXT_RE = re.compile(
    r"\b(?:application (?:has been |was )?submitted|application received|"
    r"thank(?:s| you) for applying|we(?:'ve| have) received your application|"
    r"application (?:was )?sent|successfully applied|already applied)\b",
    re.IGNORECASE,
)
_SUBMISSION_URL_RE = re.compile(
    r"(?:^|[/_-])(?:application[-_]?submitted|submitted|thank[-_]?you|confirmation|application[-_]?success)(?:[/_?&#-]|$)",
    re.IGNORECASE,
)


class FormOperation(BaseModel):
    index: int | None = Field(default=None, ge=1)
    label: str | None = None
    placeholder: str | None = None
    name: str | None = None
    value: str | None = None
    kind: Literal["text", "select", "checkbox", "radio", "upload"] = "text"
    checked: bool | None = None
    path: str | None = None


class FillFormAction(BaseModel):
    operations: list[FormOperation] = Field(min_length=1, max_length=50)


class _ExecutorFailure(RuntimeError):
    def __init__(self, operation: str, attempted: list[str], retries: int, reason: str):
        self.operation, self.attempted, self.retries, self.reason = operation, attempted, retries, reason
        super().__init__(reason)


class _PlaywrightSession:
    def __init__(self) -> None:
        self._playwright = self._browser = self._cdp_url = None

    async def page(self, browser_session: Any):
        cdp_url = getattr(browser_session, "cdp_url", None)
        if not cdp_url:
            raise RuntimeError("Browser Use CDP URL is unavailable")
        if self._browser is None or self._cdp_url != cdp_url or not self._browser.is_connected():
            await self.close()
            from playwright.async_api import async_playwright
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.connect_over_cdp(cdp_url)
            self._cdp_url = cdp_url
        current_url = ""
        try:
            current_url = await browser_session.get_current_page_url()
        except Exception:
            pass
        pages = [p for context in reversed(self._browser.contexts) for p in reversed(context.pages) if not p.is_closed()]
        if not pages:
            raise RuntimeError("No active Playwright page")
        if current_url:
            for page in pages:
                if page.url == current_url or page.url.rstrip("#") == current_url.rstrip("#"):
                    return page
        return pages[0]

    async def close(self) -> None:
        if self._playwright is not None:
            await self._playwright.stop()
        self._playwright = self._browser = self._cdp_url = None


def submission_evidence(url: str, visible_text: str) -> str | None:
    """Return deterministic evidence from the live page after submission."""
    text_match = _SUBMISSION_TEXT_RE.search(re.sub(r"\s+", " ", visible_text or ""))
    if text_match:
        return f"page text: {text_match.group(0)}"
    if _SUBMISSION_URL_RE.search(url or ""):
        return f"confirmation URL: {url}"
    return None


def _linkedin_id_from_url(url: str) -> str | None:
    parsed = urlsplit(url or "")
    host = (parsed.hostname or "").lower()
    if host != "linkedin.com" and not host.endswith(".linkedin.com"):
        return None
    match = re.search(r"/jobs/view/(\d{7,})(?:/|$)|[?&]currentJobId=(\d{7,})", url or "")
    return next((value for value in match.groups() if value), None) if match else None


async def _first_visible_text(page: Any, selectors: tuple[str, ...]) -> str:
    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if await locator.count() and await locator.is_visible():
                value = (await locator.inner_text()).strip()
                if value:
                    return value
        except Exception:
            continue
    return ""


async def read_linkedin_job_listing(browser_session: Any) -> tuple[dict[str, Any] | None, str]:
    """Read one LinkedIn listing's identity and metadata from the current live page."""
    session = _PlaywrightSession()
    try:
        page = await session.page(browser_session)
        await page.wait_for_timeout(1000)
        job_id = _linkedin_id_from_url(page.url)
        if not job_id:
            return None, f"Current page has no LinkedIn job ID: {page.url}"
        card = page.locator(
            f'[data-occludable-job-id="{job_id}"], [data-job-id="{job_id}"]'
        ).first
        has_card = await card.count() and await card.is_visible()
        if has_card:
            title = await _first_visible_text(card, (
                ".job-card-list__title",
                ".artdeco-entity-lockup__title",
                ".job-card-container__link",
            ))
            company = await _first_visible_text(card, (
                ".artdeco-entity-lockup__subtitle",
                ".job-card-container__primary-description",
            ))
            location = await _first_visible_text(card, (
                ".artdeco-entity-lockup__caption",
                ".job-card-container__metadata-item",
            ))
        else:
            title = await _first_visible_text(page, (
                ".job-details-jobs-unified-top-card__job-title h1",
                ".job-details-jobs-unified-top-card__job-title",
                ".jobs-unified-top-card__job-title",
                "h1",
            ))
            company = await _first_visible_text(page, (
                ".job-details-jobs-unified-top-card__company-name",
                ".jobs-unified-top-card__company-name",
                ".topcard__org-name-link",
                "[class*='top-card__company-name']",
            ))
            location = await _first_visible_text(page, (
                ".job-details-jobs-unified-top-card__primary-description-container .tvm__text--low-emphasis",
                ".jobs-unified-top-card__bullet",
                ".topcard__flavor--bullet",
                "[class*='top-card__location']",
            ))
        location = re.split(r"\s*[·|]\s*|\n", location, maxsplit=1)[0].strip()
        if not title or not company:
            return None, "LinkedIn title or company was not readable from the selected listing"
        button_text = " ".join(await page.locator("button").all_inner_texts())
        easy_apply = True if re.search(r"\beasy apply\b", button_text, re.IGNORECASE) else False
        return {
            "url": f"https://www.linkedin.com/jobs/view/{job_id}/",
            "title": title.strip(),
            "company": company.strip(),
            "location": location,
            "easy_apply": easy_apply,
        }, ""
    except Exception as exc:
        return None, f"LinkedIn listing read failed: {exc}"
    finally:
        await session.close()


async def verify_submission(browser_session: Any) -> tuple[bool, str]:
    """Verify submission from the current live browser page, not agent self-report."""
    session = _PlaywrightSession()
    try:
        page = await session.page(browser_session)
        visible_text = await page.locator("body").inner_text(timeout=3000)
        evidence = submission_evidence(page.url, visible_text)
        return bool(evidence), evidence or "No submission confirmation was visible"
    except Exception as exc:
        return False, f"Could not inspect submission confirmation: {exc}"
    finally:
        await session.close()


def _xpath_for(node: Any) -> str:
    path = getattr(node, "xpath", "")
    if not path:
        raise LookupError("Browser Use element has no XPath")
    return "xpath=//" + path.lstrip("/")


def _attr(node: Any, name: str) -> str:
    return str((getattr(node, "attributes", {}) or {}).get(name, "") or "").strip()


def _text(node: Any) -> str:
    try:
        return str(node.get_meaningful_text_for_llm() or "").strip()
    except Exception:
        return str(getattr(node, "node_value", "") or "").strip()


def _css(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _role(node: Any) -> str | None:
    role = _attr(node, "role").lower()
    if role:
        return role
    return {"button": "button", "a": "link", "select": "combobox"}.get(str(getattr(node, "node_name", "")).lower())


def _descriptors(node: Any, hints: FormOperation | None = None) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    def add(kind: str, value: str | None) -> None:
        value = (value or "").strip()
        if value and (kind, value) not in result:
            result.append((kind, value))
    if hints:
        add("label", hints.label); add("placeholder", hints.placeholder); add("name", hints.name)
    add("label", _attr(node, "aria-label")); add("label", _attr(node, "title"))
    add("placeholder", _attr(node, "placeholder")); add("name", _attr(node, "name")); add("id", _attr(node, "id"))
    add("testid", _attr(node, "data-testid")); add("data-test", _attr(node, "data-test"))
    add("aria-labelledby", _attr(node, "aria-labelledby"))
    meaningful = hints.label if hints and hints.label else _text(node)
    node_role = _role(node)
    if node_role and meaningful:
        add("role", f"{node_role}\x00{meaningful}")
    add("text", meaningful); add("xpath", _xpath_for(node))
    return result


def _candidate(page: Any, kind: str, value: str):
    if kind == "label": return page.get_by_label(value, exact=True)
    if kind == "placeholder": return page.get_by_placeholder(value, exact=True)
    if kind == "role":
        role, name = value.split("\x00", 1)
        return page.get_by_role(role, name=name, exact=True)
    if kind == "name": return page.locator(f'[name="{_css(value)}"]')
    if kind == "id": return page.locator(f'[id="{_css(value)}"]')
    if kind == "testid": return page.locator(f'[data-testid="{_css(value)}"]')
    if kind == "data-test": return page.locator(f'[data-test="{_css(value)}"]')
    if kind == "aria-labelledby": return page.locator(f'[aria-labelledby="{_css(value)}"]')
    if kind == "text": return page.get_by_text(value, exact=True)
    return page.locator(value)


def _frame_hint(node: Any) -> str | None:
    current = getattr(node, "parent_node", None)
    while current is not None:
        if str(getattr(current, "node_name", "")).lower() in {"iframe", "frame"}:
            return _attr(current, "src") or _attr(current, "name") or None
        current = getattr(current, "parent_node", None)
    return None


def _same_frame_url(hint: str, frame_url: str) -> bool:
    """Match an observed iframe despite ATS tracking-query differences."""
    if not hint or not frame_url:
        return False
    if hint == frame_url:
        return True
    expected, actual = urlsplit(hint), urlsplit(frame_url)
    return (
        expected.scheme == actual.scheme
        and expected.netloc == actual.netloc
        and expected.path.rstrip("/") == actual.path.rstrip("/")
    )


def _scope_for_node(page: Any, node: Any) -> Any:
    """Return the Playwright Page/Frame containing a Browser Use node."""
    hint = _frame_hint(node)
    if not hint:
        return page
    for frame in page.frames:
        if _same_frame_url(hint, frame.url) or frame.name == hint:
            return frame
    raise LookupError(f"Playwright frame not found for Browser Use node: {hint}")


async def _visible(locator: Any) -> bool:
    try:
        return bool(await locator.count()) and bool(await locator.first.is_visible())
    except Exception:
        return False


async def _resolve(browser_session: Any, session: _PlaywrightSession, index: int | None, hints: FormOperation | None = None):
    page = await session.page(browser_session)
    attempted: list[str] = []
    node = await browser_session.get_element_by_index(index) if index is not None else None
    if node is not None:
        descriptors = _descriptors(node, hints)
    elif hints is not None:
        # Browser Use indexes are observation-scoped and can disappear after a
        # React/Vue re-render.  A batch action often still has stable semantic
        # hints, so do not discard those hints merely because the index became
        # stale between observation and execution.
        descriptors = []
        for kind, value in (
            ("label", hints.label),
            ("placeholder", hints.placeholder),
            ("name", hints.name),
        ):
            if value:
                descriptors.append((kind, value))
        if hints.label:
            role = {"checkbox": "checkbox", "radio": "radio", "select": "combobox"}.get(hints.kind)
            if role:
                descriptors.append(("role", f"{role}\x00{hints.label}"))
    else:
        raise LookupError("A Browser Use index or semantic field hint is required")
    scope = _scope_for_node(page, node) if node is not None else page
    for kind, value in descriptors:
        attempted.append(f"{kind}:{value}")
        try:
            locator = _candidate(scope, kind, value).first
            if await _visible(locator):
                return page, locator, attempted, node
        except Exception as exc:
            logger.debug("EXECUTOR=playwright locator_failed locator=%s error=%s", attempted[-1], exc)
            continue
    suffix = " (stale Browser Use index)" if index is not None and node is None else ""
    raise LookupError(f"No visible Playwright locator matched index {index or 'semantic field'}{suffix}")


async def _tag_name(locator: Any) -> str:
    try:
        value = await locator.evaluate("el => el.tagName.toLowerCase()")
        return value if isinstance(value, str) else ""
    except Exception:
        return ""


async def _set_checked(locator: Any, checked: bool) -> None:
    """Set native and ARIA checkbox/radio controls without blind toggles."""
    try:
        current = bool(await locator.is_checked())
        if current == checked:
            return
        if checked:
            await locator.check()
        else:
            await locator.uncheck()
        return
    except Exception:
        current = await locator.get_attribute("aria-checked")
        if current is not None and (current.lower() == "true") == checked:
            return
        await locator.click()
        current = await locator.get_attribute("aria-checked")
        if current is not None and (current.lower() == "true") != checked:
            raise AssertionError(f"aria-checked did not persist: {current!r}")


async def _checked_state(locator: Any) -> bool:
    """Read checked state from native inputs or ARIA checkbox/radio widgets."""
    try:
        return bool(await locator.is_checked())
    except Exception:
        current = await locator.get_attribute("aria-checked")
        if current is None:
            raise AssertionError("control exposes neither native checked state nor aria-checked")
        return current.lower() == "true"


async def _select_value(page: Any, locator: Any, value: str) -> None:
    """Select native options and common ARIA combobox/listbox controls."""
    if await _tag_name(locator) == "select":
        option_value = None
        try:
            options = await locator.evaluate(
                "el => Array.from(el.options).map(o => ({value: o.value, text: o.textContent || ''}))"
            )
            option_value = next(
                (item["value"] for item in options if _equivalent(item["text"], value)),
                None,
            )
        except Exception:
            pass
        try:
            if option_value is not None:
                await locator.select_option(value=option_value)
            else:
                await locator.select_option(label=value)
        except Exception:
            await locator.select_option(value=value)
        selected_text = await locator.locator("option:checked").text_content()
        selected_value = await locator.input_value()
        if not _equivalent(selected_text or selected_value, value):
            raise AssertionError("selected native option did not persist")
        return

    await locator.click()
    option = page.get_by_role("option", name=value, exact=True)
    if not await _visible(option):
        option = page.get_by_text(value, exact=True)
    if not await _visible(option):
        raise LookupError(f"combobox option is not visible: {value}")
    await option.click()
    expanded = await locator.get_attribute("aria-expanded")
    if expanded == "true":
        raise AssertionError("combobox remained expanded after selection")


async def _scroll_page(page: Any, amount: int) -> None:
    """Scroll the document or its actual scrollable form container.

    ATS forms frequently put the application inside an independently scrolling
    shell.  A mouse wheel over the outer page can be a successful browser call
    while changing no scroll position at all, which previously caused the LLM
    to repeat up/down actions indefinitely.
    """
    before = await page.evaluate("window.scrollY")
    await page.mouse.wheel(0, amount)
    await page.wait_for_timeout(ACTION_WAIT_MS)
    after = await page.evaluate("window.scrollY")
    if after != before:
        return

    changed = await page.evaluate(
        """amount => {
            const candidates = [document.scrollingElement, ...document.querySelectorAll('*')]
                .filter(el => el && el.scrollHeight > el.clientHeight && el.clientHeight > 0);
            for (const el of candidates.sort((a, b) => b.clientHeight - a.clientHeight)) {
                const previous = el.scrollTop;
                el.scrollTop = Math.max(0, Math.min(el.scrollHeight - el.clientHeight, previous + amount));
                if (el.scrollTop !== previous) return true;
            }
            return false;
        }""",
        amount,
    )
    if not changed:
        raise AssertionError("no scrollable document or form container moved")


async def _retry(
    operation: str,
    fn: Callable[[], Awaitable[Any]],
    attempted: list[str],
    refresh: Callable[[], Awaitable[Any]] | None = None,
):
    last: Exception | None = None
    for n in range(MAX_ATTEMPTS):
        if n and refresh is not None:
            try:
                await refresh()
            except Exception as exc:
                last = exc
                logger.info("EXECUTOR=playwright re-observe_failed operation=%s attempt=%d reason=%s", operation, n + 1, exc)
                continue
        try:
            return await fn()
        except Exception as exc:
            last = exc
            if n + 1 < MAX_ATTEMPTS:
                logger.info("EXECUTOR=playwright retry operation=%s attempt=%d reason=%s", operation, n + 1, exc)
    raise _ExecutorFailure(operation, attempted, MAX_ATTEMPTS - 1, str(last or "operation failed"))


async def _tabs(browser_session: Any) -> set[str]:
    try:
        return {str(t.target_id) for t in await browser_session.get_tabs()}
    except Exception:
        return set()


async def _sync_tab(browser_session: Any, before: set[str]) -> bool:
    try:
        tabs = await browser_session.get_tabs()
        new = [t for t in tabs if str(t.target_id) not in before]
        if not new:
            return False
        from browser_use.browser.events import SwitchTabEvent
        event = browser_session.event_bus.dispatch(SwitchTabEvent(target_id=new[0].target_id))
        await event
        await event.event_result(raise_if_any=False, raise_if_none=False)
        logger.info("EXECUTOR=playwright synchronized new tab target=%s", new[0].target_id)
        return True
    except Exception as exc:
        logger.warning("EXECUTOR=playwright new-tab synchronization failed: %s", exc)
        return False


async def _signature(page: Any) -> str:
    try:
        return page.url + "\n" + (await page.locator("body").inner_text(timeout=500))[:12000]
    except Exception:
        return page.url


def _failure(ActionResult: Any, failure: _ExecutorFailure) -> Any:
    details = {"operation": failure.operation, "attempted_locators": failure.attempted, "retry_count": failure.retries, "verification_failure": failure.reason}
    return ActionResult(error=json.dumps(details, separators=(",", ":")), metadata={"executor": "playwright", **details})


def _mark(result: Any, metadata: dict[str, Any]) -> Any:
    values = dict(getattr(result, "metadata", None) or {})
    values.update(metadata)
    result.metadata = values
    return result


def _normalized(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "", (value or "").lower())
    aliases = {
        "us": "unitedstates",
        "usa": "unitedstates",
        "unitedstatesofamerica": "unitedstates",
        "yesauthorizedtowork": "yes",
        "authorized": "yes",
        "nope": "no",
    }
    return aliases.get(value, value)


def _equivalent(left: str, right: str) -> bool:
    """Allow harmless formatting and common dropdown wording differences."""
    a, b = _normalized(str(left)), _normalized(str(right))
    return bool(a and b and (a == b or (a in {"yes", "no"} and b.startswith(a)) or (b in {"yes", "no"} and a.startswith(b))))


def _protected_key(hint: str) -> str | None:
    """Map a small set of high-risk factual fields to structured profile keys."""
    hint = re.sub(r"[^a-z0-9]+", " ", (hint or "").lower()).strip()
    if re.search(r"\b(?:login|sign in|username|account email)\b", hint):
        return None
    if re.search(r"\b(?:desired|preferred|job|office|work) location\b", hint):
        return None
    if (
        re.search(r"\b(?:address|street|city|town|state|province|region|zip|postal|country|nation)\b", hint)
        and re.search(r"\b(?:education|school|university|college|institution|employment|employer|company|work|office)\b", hint)
    ):
        return None
    rules = (
        (r"\b(?:full|legal|applicant|candidate|your) name\b|\bname as it appears\b", "full_name"),
        (r"^(?:name\s*){1,2}$", "full_name"),
        (r"\b(?:first|given) name\b|\blegal first\b", "first_name"),
        (r"\bmiddle name\b", "middle_name"),
        (r"\b(?:last|family|sur)name\b|\blegal last\b", "last_name"),
        (r"\b(?:country code|dialing code)\b", "phone_country_code"),
        (r"\b(?:email|e mail)\b", "email"),
        (r"\b(?:phone|mobile|telephone|cell)\b", "phone"),
        (r"\b(?:street|address line|address street)\b|\b(?:home|mailing|residential) address\b", "address_street"),
        (r"^(?:address\s*){1,2}$", "address_street"),
        (r"\b(?:current|home|residential) location\b", "current_location"),
        (r"\b(?:city|town)\b", "address_city"),
        (r"\b(?:zip|postal code)\b", "address_zip"),
        (r"\b(?:state|province|region)\b", "address_state"),
        (r"\b(?:country|nation)\b", "address_country"),
        (r"\b(?:visa )?sponsor(?:ship)?\b", "sponsorship"),
        (r"\brelocat(?:e|ion)\b", "relocation"),
        (r"\b(?:legally )?authorized to work\b", "work_authorized"),
        (r"\bwork authorization(?: status)?\b", "work_authorization"),
        (r"\bcurrent (?:job )?(?:title|role)\b", "current_role"),
        (r"\b(?:school|university|college)\b", "education_school"),
        (r"\bdegree\b", "education_degree"),
        (r"\bgraduat(?:ion|ed)\b", "education_graduation"),
        (r"\b(?:minimum|min) (?:salary|compensation)\b", "salary_min"),
        (r"\b(?:maximum|max) (?:salary|compensation)\b", "salary_max"),
        (r"\byears?\b.*\b(?:machine learning|ml)\b", "years_ml"),
        (r"\byears?\b.*\b(?:generative ai|genai|gen ai)\b", "years_genai"),
        (r"\byears?\b.*\bpython\b", "years_python"),
        (r"\byears?\b.*\baws\b", "years_aws"),
        (r"\byears?\b.*\bazure\b", "years_azure"),
        (r"\b(?:total|professional) years?\b|\byears? of (?:professional )?experience\b", "years_experience"),
    )
    return next((key for pattern, key in rules if re.search(pattern, hint)), None)


def _canonical_value(operation: FormOperation, authoritative: dict[str, str]) -> str | None:
    """Resolve secret tokens and profile-controlled fields before touching the page.

    The planner may describe a field correctly while still inventing its value.
    Values for identity/contact fields therefore come from the application-owned
    profile, never from the model's prose.  Unknown questions remain planner-
    supplied so the normal Q&A flow is preserved.
    """
    value = operation.value
    if value is None:
        return None
    token = re.fullmatch(r"<secret>([^<]+)</secret>", value.strip())
    if token and token.group(1) in authoritative:
        return authoritative[token.group(1)]

    hint = " ".join(filter(None, (operation.label, operation.name, operation.placeholder)))
    key = _protected_key(hint)
    if key and authoritative.get(key):
        return authoritative[key]
    return value


def _resolve_token(value: str | None, authoritative: dict[str, str]) -> str | None:
    if value is None:
        return None
    match = re.fullmatch(r"<secret>([^<]+)</secret>", value.strip())
    return authoritative.get(match.group(1), value) if match else value


def _operation_for_node(node: Any, value: str, kind: Literal["text", "select"]) -> FormOperation:
    return FormOperation(
        label=_text(node),
        placeholder=_attr(node, "placeholder"),
        name=_attr(node, "name"),
        value=value,
        kind=kind,
    )


async def _validate_authoritative_fields(page: Any, authoritative: dict[str, str]) -> int:
    """Correct conflicting protected fields immediately before submission.

    This is deliberately best-effort: unfamiliar ATS controls are left to the
    agent so the safety net cannot turn a recoverable application into a failure.
    """
    if not authoritative:
        return 0
    controls = page.locator("input:not([type=hidden]), select, textarea")
    corrected = 0
    for index in range(await controls.count()):
        locator = controls.nth(index)
        try:
            if not await locator.is_visible():
                continue
            info = await locator.evaluate(
                """el => {
                    const labels = Array.from(el.labels || []).map(x => x.textContent || '');
                    const section = el.closest('fieldset, section, [role="group"]');
                    const legend = section?.querySelector('legend')?.textContent || '';
                    const headings = section
                        ? Array.from(section.querySelectorAll('h1,h2,h3,h4,h5,h6,[role="heading"]'))
                            .map(x => x.textContent || '').join(' ')
                        : '';
                    const optionLabel = el.type === 'radio' || el.type === 'checkbox'
                        ? labels.join(' ') || el.value || '' : '';
                    return {
                        hint: [labels.join(' '), legend, headings, el.getAttribute('aria-label') || '',
                               el.getAttribute('placeholder') || '', el.name || '', el.id || ''].join(' '),
                        tag: el.tagName.toLowerCase(), type: (el.type || '').toLowerCase(),
                        value: el.value || '', checked: Boolean(el.checked), optionLabel,
                        selectedText: el.tagName.toLowerCase() === 'select'
                            ? el.options[el.selectedIndex]?.textContent || '' : '',
                        options: el.tagName.toLowerCase() === 'select'
                            ? Array.from(el.options).map(o => ({value: o.value, text: o.textContent || ''})) : []
                    };
                }"""
            )
            key = _protected_key(info.get("hint", ""))
            expected = authoritative.get(key or "")
            if not expected:
                continue

            if info["type"] in {"radio", "checkbox"}:
                should_check = _equivalent(info.get("optionLabel", ""), expected)
                if info["type"] == "checkbox" and _normalized(expected) in {"yes", "no"}:
                    should_check = _normalized(expected) == "yes"
                if info["type"] == "checkbox" and info["checked"] != should_check:
                    if should_check:
                        await locator.check()
                    else:
                        await locator.uncheck()
                    corrected += 1
                elif info["type"] == "radio" and should_check and not info["checked"]:
                    await locator.check()
                    corrected += 1
            elif info["tag"] == "select":
                if _equivalent(info["value"], expected) or _equivalent(info["selectedText"], expected):
                    continue
                option = next(
                    (item for item in info["options"] if _equivalent(item["text"], expected)),
                    None,
                )
                if option:
                    await locator.select_option(value=option["value"])
                    corrected += 1
            elif info["value"] and not _equivalent(info["value"], expected):
                await locator.fill(expected)
                corrected += 1
        except Exception as exc:
            logger.debug("PROFILE_VALIDATION skipped control=%d reason=%s", index, exc)
    return corrected


def install_playwright_actions(tools: Any, authoritative_values: dict[str, str] | None = None) -> None:
    """Replace native interaction handlers with verified Playwright handlers."""
    if getattr(tools, "_langhire_playwright_actions_installed", False):
        return
    from browser_use.agent.views import ActionResult
    from browser_use.tools.registry.views import RegisteredAction
    registry = tools.registry.registry.actions
    names = ("click", "input", "upload_file", "select_dropdown", "scroll", "send_keys")
    originals = {name: registry[name] for name in names if name in registry}
    session = _PlaywrightSession()
    authoritative = {str(k): str(v) for k, v in (authoritative_values or {}).items() if v is not None and str(v).strip()}

    async def use_native(
        name: str,
        failure: _ExecutorFailure,
        params: Any,
        browser_session: Any,
        kwargs: dict[str, Any],
        verify: Callable[[], Awaitable[bool]] | None = None,
    ):
        logger.warning("EXECUTOR=browser-use-fallback operation=%s reason=%s", name, failure.reason)
        result = await originals[name].function(params=params, browser_session=browser_session, **kwargs)
        native_error = getattr(result, "error", None)
        native_content = getattr(result, "long_term_memory", None) or getattr(result, "extracted_content", None) or "No Browser Use result content"
        feedback = (
            f"Playwright fallback reason: {failure.reason}\n"
            f"Browser Use fallback result: {native_content}"
        )
        if native_error:
            feedback += f"\nBrowser Use fallback error: {native_error}"
        if verify is not None and not native_error and not await verify():
            result.error = f"{feedback}\nFallback did not verify the intended target/state change."
            feedback += "\nFallback did not verify the intended target/state change."
        result.extracted_content = feedback
        result.long_term_memory = feedback
        return _mark(result, {"executor": "browser-use-fallback", "fallback_operation": failure.operation, "playwright_retries": failure.retries, "playwright_verification_failure": failure.reason})

    def replace(name: str, handler: Callable) -> None:
        old = originals[name]
        registry[name] = RegisteredAction(name=old.name, description=old.description, function=handler, param_model=old.param_model, terminates_sequence=old.terminates_sequence, domains=old.domains)

    def authoritative_override(operation: FormOperation, effective_value: str | None) -> bool:
        return bool(
            operation.value is not None
            and effective_value is not None
            and not _equivalent(operation.value, effective_value)
        )

    def mark_authoritative_success(result: Any, field: str) -> Any:
        if not getattr(result, "error", None):
            result.extracted_content = (
                f"Authoritative Profile value was applied and verified for {field}. "
                "This field is successfully completed; do not retry the proposed value."
            )
            _mark(result, {"authoritative_override": True})
        return result

    async def click(*, params, browser_session=None, **kwargs):
        attempted: list[str] = []
        fallback_verify: Callable[[], Awaitable[bool]] | None = None
        resolve_attempted = False
        resolve_succeeded = False
        try:
            before_tabs = await _tabs(browser_session)
            if params.index is None and params.coordinate_x is not None and params.coordinate_y is not None:
                page = await session.page(browser_session)
                await _validate_authoritative_fields(page, authoritative)
                x, y = params.coordinate_x, params.coordinate_y
                sizes = (getattr(browser_session, "llm_screenshot_size", None), getattr(browser_session, "_original_viewport_size", None))
                if sizes[0] and sizes[1]:
                    x, y = int(x / sizes[0][0] * sizes[1][0]), int(y / sizes[0][1] * sizes[1][1])
                old_url, old_state = page.url, await _signature(page)
                async def verify_coordinate_fallback():
                    await page.wait_for_timeout(ACTION_WAIT_MS)
                    current = await session.page(browser_session)
                    return current.url != old_url or await _signature(current) != old_state
                fallback_verify = verify_coordinate_fallback
                attempted = [f"coordinates:{x},{y}"]
                await _retry("click:coordinates", lambda: page.mouse.click(x, y), attempted)
                await page.wait_for_timeout(ACTION_WAIT_MS)
                opened = await _sync_tab(browser_session, before_tabs)
                if not opened and page.url == old_url and await _signature(page) == old_state:
                    raise _ExecutorFailure("click:coordinates", attempted, MAX_ATTEMPTS - 1, "click produced no observable state change")
                return ActionResult(extracted_content="Playwright clicked coordinates", metadata={"executor": "playwright"})
            page = await session.page(browser_session)
            old_url, old_state = page.url, await _signature(page)
            resolve_attempted = True
            page, locator, attempted, node = await _resolve(browser_session, session, params.index)
            resolve_succeeded = True
            target_text = _text(node).lower()
            if re.search(r"\b(?:submit|send|finish|complete)(?: application)?\b|\bapply(?: now)?\b", target_text):
                try:
                    corrected = await _validate_authoritative_fields(
                        _scope_for_node(page, node), authoritative
                    )
                    if corrected:
                        logger.info("PROFILE_VALIDATION corrected=%d before=%s", corrected, target_text[:80])
                except Exception as exc:
                    logger.warning("PROFILE_VALIDATION unavailable before submit: %s", exc)
            if any(word in target_text for word in ("apply", "next", "continue", "submit", "sign in", "login")):
                async def verify_indexed_fallback():
                    await page.wait_for_timeout(ACTION_WAIT_MS)
                    current = await session.page(browser_session)
                    return current.url != old_url or await _signature(current) != old_state
                fallback_verify = verify_indexed_fallback
            input_type = await locator.evaluate("el => el instanceof HTMLInputElement ? el.type.toLowerCase() : ''")
            aria_role = (await locator.get_attribute("role") or "").lower()
            if input_type in {"checkbox", "radio"} or aria_role in {"checkbox", "radio"}:
                await _retry("click:check", lambda: _set_checked(locator, True), attempted)
                if not await _checked_state(locator):
                    raise _ExecutorFailure("click:check", attempted, MAX_ATTEMPTS - 1, "checked state did not persist")
            else:
                await _retry("click", locator.click, attempted)
            await page.wait_for_timeout(ACTION_WAIT_MS)
            opened = await _sync_tab(browser_session, before_tabs)
            needs_change = any(word in target_text for word in ("apply", "next", "continue", "submit", "sign in", "login"))
            if not opened and needs_change and page.url == old_url and await _signature(page) == old_state:
                raise _ExecutorFailure("click", attempted, MAX_ATTEMPTS - 1, "expected navigation or visible state transition did not occur")
            return ActionResult(extracted_content=f"Playwright clicked element {params.index}", metadata={"executor": "playwright", "locators": attempted})
        except _ExecutorFailure as failure:
            if resolve_attempted and not resolve_succeeded and params.index is not None:
                async def unresolved_index_fallback_is_unverified():
                    return False
                fallback_verify = unresolved_index_fallback_is_unverified
            return await use_native("click", failure, params, browser_session, kwargs, fallback_verify)
        except Exception as exc:
            if resolve_attempted and not resolve_succeeded and params.index is not None:
                async def unresolved_index_fallback_is_unverified():
                    return False
                fallback_verify = unresolved_index_fallback_is_unverified
            return await use_native("click", _ExecutorFailure("click", attempted, 0, str(exc)), params, browser_session, kwargs, fallback_verify)

    async def input_text(*, params, browser_session=None, **kwargs):
        attempted: list[str] = []
        safe_params = params
        overridden = False
        try:
            _, locator, attempted, node = await _resolve(browser_session, session, params.index)
            operation = _operation_for_node(node, params.text, "text")
            text = _canonical_value(operation, authoritative)
            overridden = authoritative_override(operation, text)
            safe_params = params.model_copy(update={"text": text})
            async def fill():
                if params.clear: await locator.fill(text)
                else: await locator.press_sequentially(text)
                value = await locator.input_value()
                if (params.clear and value != text) or (not params.clear and not value.endswith(text)):
                    raise AssertionError(f"value mismatch: {value!r}")
            await _retry("input", fill, attempted)
            logger.info("EXECUTOR=playwright operation=input verified index=%s", params.index)
            result = ActionResult(extracted_content=f"Playwright filled element {params.index}", metadata={"executor": "playwright", "locators": attempted})
            return mark_authoritative_success(result, f"element {params.index}") if overridden else result
        except _ExecutorFailure as failure:
            result = await use_native("input", failure, safe_params, browser_session, kwargs)
            return mark_authoritative_success(result, f"element {params.index}") if overridden else result
        except Exception as exc:
            result = await use_native("input", _ExecutorFailure("input", attempted, 0, str(exc)), safe_params, browser_session, kwargs)
            return mark_authoritative_success(result, f"element {params.index}") if overridden else result

    async def upload_file(*, params, browser_session=None, **kwargs):
        attempted: list[str] = []
        try:
            _, locator, attempted, _ = await _resolve(browser_session, session, params.index)
            await _retry("upload_file", lambda: locator.set_input_files(params.path), attempted)
            if not await locator.evaluate("el => el.files ? el.files.length : 0"): raise AssertionError("file input is empty after upload")
            return ActionResult(extracted_content=f"Playwright uploaded file at element {params.index}", metadata={"executor": "playwright", "locators": attempted})
        except _ExecutorFailure as failure: return await use_native("upload_file", failure, params, browser_session, kwargs)
        except Exception as exc: return await use_native("upload_file", _ExecutorFailure("upload_file", attempted, 0, str(exc)), params, browser_session, kwargs)

    async def select_dropdown(*, params, browser_session=None, **kwargs):
        attempted: list[str] = []
        safe_params = params
        overridden = False
        try:
            page, locator, attempted, node = await _resolve(browser_session, session, params.index)
            operation = _operation_for_node(node, params.text, "select")
            text = _canonical_value(operation, authoritative)
            overridden = authoritative_override(operation, text)
            safe_params = params.model_copy(update={"text": text})
            await _retry("select_dropdown", lambda: _select_value(page, locator, text), attempted)
            logger.info("EXECUTOR=playwright operation=select verified index=%s", params.index)
            result = ActionResult(extracted_content=f"Playwright selected {params.text!r}", metadata={"executor": "playwright", "locators": attempted})
            return mark_authoritative_success(result, f"element {params.index}") if overridden else result
        except _ExecutorFailure as failure:
            result = await use_native("select_dropdown", failure, safe_params, browser_session, kwargs)
            return mark_authoritative_success(result, f"element {params.index}") if overridden else result
        except Exception as exc:
            result = await use_native("select_dropdown", _ExecutorFailure("select_dropdown", attempted, 0, str(exc)), safe_params, browser_session, kwargs)
            return mark_authoritative_success(result, f"element {params.index}") if overridden else result

    async def scroll(*, params, browser_session=None, **kwargs):
        attempted: list[str] = []
        try:
            page = await session.page(browser_session); amount = int((getattr(params, "pages", 1) or 1) * 800) * (1 if getattr(params, "down", True) else -1)
            if getattr(params, "index", None):
                _, locator, attempted, _ = await _resolve(browser_session, session, params.index)
                async def scroll_element():
                    before = await locator.evaluate("element => element.scrollTop")
                    await locator.evaluate("(element, y) => element.scrollBy(0, y)", amount)
                    await page.wait_for_timeout(ACTION_WAIT_MS)
                    after = await locator.evaluate("element => element.scrollTop")
                    if after == before:
                        raise AssertionError("target scroll container did not move")
                await _retry("scroll", scroll_element, attempted)
            else: await _retry("scroll", lambda: _scroll_page(page, amount), [f"page:{amount}"])
            return ActionResult(extracted_content="Playwright scrolled", metadata={"executor": "playwright"})
        except _ExecutorFailure as failure: return await use_native("scroll", failure, params, browser_session, kwargs)
        except Exception as exc: return await use_native("scroll", _ExecutorFailure("scroll", attempted, 0, str(exc)), params, browser_session, kwargs)

    async def send_keys(*, params, browser_session=None, **kwargs):
        try:
            page = await session.page(browser_session)
            if str(params.keys).lower() in {"enter", "return"}:
                await _validate_authoritative_fields(page, authoritative)
            await _retry("send_keys", lambda: page.keyboard.press(params.keys), [f"key:{params.keys}"])
            return ActionResult(extracted_content=f"Playwright sent keys: {params.keys}", metadata={"executor": "playwright"})
        except _ExecutorFailure as failure: return await use_native("send_keys", failure, params, browser_session, kwargs)
        except Exception as exc: return await use_native("send_keys", _ExecutorFailure("send_keys", [], 0, str(exc)), params, browser_session, kwargs)

    for name, handler in (("click", click), ("input", input_text), ("upload_file", upload_file), ("select_dropdown", select_dropdown), ("scroll", scroll), ("send_keys", send_keys)):
        replace(name, handler)

    # This action is installed directly into Browser Use's registry (so the
    # registry cannot normalize its signature for us).  Browser Use injects
    # these special arguments when it executes an action.  Keep them explicit
    # and optional: accepting **kwargs here would hide contract drift, while a
    # narrow ``params, browser_session`` signature causes the real dispatcher
    # to fail before Playwright is reached.
    async def fill_form(
        *,
        params: FillFormAction,
        browser_session=None,
        page_url=None,
        cdp_client=None,
        page_extraction_llm=None,
        file_system=None,
        available_file_paths=None,
        has_sensitive_data=False,
        extraction_schema=None,
        context=None,
    ):
        completed: list[str] = []
        fallback_operations: list[str] = []
        authoritative_overrides: list[str] = []
        operation_results: list[dict[str, Any]] = []
        for operation in params.operations:
            attempted: list[str] = []
            operation_name = f"{operation.kind}:{operation.label or operation.name or operation.index}"
            effective_value = _canonical_value(operation, authoritative)
            overridden = authoritative_override(operation, effective_value)
            try:
                page, locator, attempted, _ = await _resolve(browser_session, session, operation.index, operation)

                async def refresh_locator():
                    nonlocal page, locator, attempted
                    page, locator, fresh_attempted, _ = await _resolve(
                        browser_session, session, operation.index, operation
                    )
                    attempted[:] = fresh_attempted

                if operation.kind == "upload":
                    if not operation.path: raise ValueError("upload requires path")
                    await _retry("fill_form:upload", lambda: locator.set_input_files(operation.path), attempted, refresh_locator)
                    if not await locator.evaluate("el => el.files ? el.files.length : 0"): raise AssertionError("upload did not persist")
                elif operation.kind == "select":
                    if effective_value is None: raise ValueError("select requires value")
                    await _retry("fill_form:select", lambda: _select_value(page, locator, effective_value), attempted, refresh_locator)
                elif operation.kind in {"checkbox", "radio"}:
                    if operation.checked is not True: raise ValueError("only selecting checkbox/radio is supported")
                    await _retry("fill_form:check", lambda: _set_checked(locator, True), attempted, refresh_locator)
                    if not await _checked_state(locator): raise AssertionError("checked state did not persist")
                else:
                    if effective_value is None: raise ValueError("text requires value")
                    await _retry("fill_form:text", lambda: locator.fill(effective_value), attempted, refresh_locator)
                    if await locator.input_value() != effective_value: raise AssertionError("text did not persist")
                completed.append(operation_name)
                if overridden:
                    authoritative_overrides.append(operation_name)
                logger.info("EXECUTOR=playwright operation=%s verified", operation_name)
                operation_results.append({"operation": operation_name, "executor": "playwright", "locators": attempted, "authoritative_override": overridden})
            except Exception as exc:
                failure = exc if isinstance(exc, _ExecutorFailure) else _ExecutorFailure("fill_form", attempted, 0, str(exc))
                try:
                    native_kwargs = {
                        "browser_session": browser_session,
                        "page_url": page_url,
                        "cdp_client": cdp_client,
                        "page_extraction_llm": page_extraction_llm,
                        "file_system": file_system,
                        "available_file_paths": available_file_paths,
                        "has_sensitive_data": has_sensitive_data,
                        "extraction_schema": extraction_schema,
                        "context": context,
                    }
                    if operation.kind == "upload": native = originals["upload_file"].param_model(index=operation.index, path=operation.path); native_result = await originals["upload_file"].function(params=native, **native_kwargs)
                    elif operation.kind == "select": native = originals["select_dropdown"].param_model(index=operation.index, text=effective_value); native_result = await originals["select_dropdown"].function(params=native, **native_kwargs)
                    elif operation.kind in {"checkbox", "radio"}: native = originals["click"].param_model(index=operation.index); native_result = await originals["click"].function(params=native, **native_kwargs)
                    else: native = originals["input"].param_model(index=operation.index, text=effective_value, clear=True); native_result = await originals["input"].function(params=native, **native_kwargs)
                    if getattr(native_result, "error", None):
                        native_error = getattr(native_result, "error", None)
                        return ActionResult(
                            error=(
                                f"Playwright fallback reason: {failure.reason}; "
                                f"Browser Use fallback error: {native_error}"
                            ),
                            metadata={
                                "executor": "browser-use-fallback",
                                "fallback_operation": failure.operation,
                                "playwright_verification_failure": failure.reason,
                            },
                        )
                    fallback_operations.append(operation_name)
                    completed.append(f"{operation_name}:browser-use-fallback")
                    if overridden:
                        authoritative_overrides.append(operation_name)
                    operation_results.append({"operation": operation_name, "executor": "browser-use-fallback", "playwright_error": failure.reason, "locators": attempted, "authoritative_override": overridden})
                except Exception as native_exc:
                    failure.reason = f"Playwright: {failure.reason}; Browser Use: {native_exc}"
                    return _failure(ActionResult, failure)
        executor = "playwright+browser-use-fallback" if fallback_operations else "playwright"
        override_message = ""
        if authoritative_overrides:
            override_message = (
                ". Authoritative Profile values were applied and verified for "
                + ", ".join(authoritative_overrides)
                + ". These fields are successfully completed; do not retry the proposed values"
            )
        return ActionResult(
            extracted_content="Playwright batch-filled " + ", ".join(completed) + override_message,
            metadata={
                "executor": executor,
                "batch_size": len(params.operations),
                "fallback_operations": fallback_operations,
                "authoritative_overrides": authoritative_overrides,
                "operation_results": operation_results,
            },
        )

    registry["playwright_fill_form"] = RegisteredAction(
        name="playwright_fill_form",
        description="Fill all currently visible application fields in one verified Playwright batch. Provide the Browser Use index and semantic label/name/placeholder when available.",
        function=fill_form,
        param_model=FillFormAction,
    )
    # Mark only after every replacement and the custom action registration has
    # succeeded.  A partial setup must remain retryable.
    try:
        setattr(tools, "_langhire_playwright_actions_installed", True)
    except Exception as exc:
        raise RuntimeError("Browser Use Tools object cannot be marked for idempotent Playwright installation") from exc
