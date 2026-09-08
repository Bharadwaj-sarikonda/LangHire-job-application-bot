import json
import sys
import types

from cli import collect_jobs


class FakePage:
    def __init__(self, scans, scrolls, next_pages=None):
        self.scans = list(scans)
        self.scrolls = list(scrolls)
        self.next_pages = list(next_pages or [False])
        self.last_scan = []
        self.url = "https://www.linkedin.com/jobs/search/?start=0"

    async def evaluate(self, script):
        if script == "() => location.href":
            return self.url
        if script == collect_jobs._READ_LINKEDIN_CARDS_JS:
            if self.scans:
                self.last_scan = self.scans.pop(0)
            return json.dumps(self.last_scan)
        if script == collect_jobs._SCROLL_LINKEDIN_RESULTS_JS:
            value = self.scrolls.pop(0) if self.scrolls else {"advanced": False}
            return json.dumps(value)
        if script == collect_jobs._NEXT_LINKEDIN_PAGE_JS:
            clicked = self.next_pages.pop(0) if self.next_pages else False
            if clicked:
                self.url = "https://www.linkedin.com/jobs/search/?start=25"
            return str(clicked)
        raise AssertionError("unexpected script")


def card(job_id, *, easy_apply=None):
    return {
        "job_id": str(job_id),
        "title": f"Title {job_id}",
        "company": f"Company {job_id}",
        "location": f"Location {job_id}",
        "easy_apply": easy_apply,
    }


def test_persisted_linkedin_ids_use_stable_ids_not_full_urls():
    jobs = {
        "https://www.linkedin.com/jobs/view/123/?trackingId=old": {"url": ""},
        "https://example.com/jobs/123": {"job_id": "456"},
    }
    assert collect_jobs._persisted_linkedin_ids(jobs) == {"123", "456"}


async def test_card_collection_scans_scrolls_dedupes_and_saves_immediately(monkeypatch):
    existing_url = "https://www.linkedin.com/jobs/view/100/"
    stored = {existing_url: {"url": existing_url, "title": "Already saved"}}
    writes = []

    monkeypatch.setattr(collect_jobs, "read_jobs", lambda: dict(stored))

    def write_jobs(jobs):
        stored.clear()
        stored.update(jobs)
        writes.append(dict(jobs))

    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(collect_jobs, "write_jobs", write_jobs)
    monkeypatch.setattr(collect_jobs.asyncio, "sleep", no_sleep)
    page = FakePage(
        scans=[
            [card(100), card(200)],
            [card(200), card(300, easy_apply=True)],
            [card(300, easy_apply=True)],
        ],
        scrolls=[{"advanced": True}, {"advanced": False}, {"advanced": False}],
    )

    found = await collect_jobs._collect_linkedin_result_cards(
        page, "Engineer", dict(stored), max_jobs=0
    )

    assert [job["url"] for job in found] == [
        "https://www.linkedin.com/jobs/view/200/",
        "https://www.linkedin.com/jobs/view/300/",
    ]
    assert stored["https://www.linkedin.com/jobs/view/200/"]["easy_apply"] is None
    assert stored["https://www.linkedin.com/jobs/view/300/"]["easy_apply"] is True
    assert len(writes) == 2


async def test_card_collection_stops_at_new_persisted_job_limit(monkeypatch):
    stored = {}
    monkeypatch.setattr(collect_jobs, "read_jobs", lambda: dict(stored))

    def write_jobs(jobs):
        stored.clear()
        stored.update(jobs)

    monkeypatch.setattr(collect_jobs, "write_jobs", write_jobs)
    page = FakePage(scans=[[card(1), card(2)]], scrolls=[])

    found = await collect_jobs._collect_linkedin_result_cards(
        page, "Engineer", {}, max_jobs=1
    )

    assert [job["url"] for job in found] == ["https://www.linkedin.com/jobs/view/1/"]
    assert len(stored) == 1
    assert page.scrolls == []


async def test_next_page_waits_for_results_page_to_change(monkeypatch):
    async def no_sleep(_delay):
        return None

    monkeypatch.setattr(collect_jobs.asyncio, "sleep", no_sleep)
    page = FakePage(scans=[[card(2)]], scrolls=[], next_pages=[True])

    assert await collect_jobs._go_to_next_linkedin_page(page, ("1",)) is True
    assert page.url.endswith("start=25")


async def test_collection_handoff_keeps_and_reuses_prefiltered_page(monkeypatch):
    calls = {}

    class Page:
        async def goto(self, url):
            calls["search_url"] = url

    class BrowserSession:
        def __init__(self, **kwargs):
            calls["browser_kwargs"] = kwargs
            calls["browser"] = self
            self.page = Page()
            self.current_page_calls = 0

        async def start(self):
            pass

        async def must_get_current_page(self):
            self.current_page_calls += 1
            return self.page

        async def kill(self):
            calls["killed"] = True

    class Tools:
        def set_coordinate_clicking(self, enabled):
            calls["coordinate_clicking"] = enabled

    class Agent:
        def __init__(self, **kwargs):
            calls["agent_kwargs"] = kwargs
            self.tools = Tools()

        async def run(self):
            calls["agent_ran"] = True

    async def collect_cards(page, title, existing_jobs, max_jobs):
        calls["collection_page"] = page
        return []

    fake_browser_use = types.ModuleType("browser_use")
    fake_browser_use.Agent = Agent
    fake_browser_use.BrowserSession = BrowserSession
    monkeypatch.setitem(sys.modules, "browser_use", fake_browser_use)
    monkeypatch.setattr(collect_jobs, "refresh_credentials", lambda: None)
    monkeypatch.setattr(collect_jobs, "_agent_log_start", lambda *_args: None)
    monkeypatch.setattr(collect_jobs.config, "get_llm", lambda session_id=None: object())
    monkeypatch.setattr(collect_jobs, "_collect_linkedin_result_cards", collect_cards)

    await collect_jobs.collect_for_title(
        "AI Engineer",
        {},
        {"target_locations": ["United States"]},
        max_jobs=10,
    )

    browser = calls["browser"]
    assert calls["browser_kwargs"]["keep_alive"] is True
    assert calls["agent_kwargs"]["directly_open_url"] is False
    assert calls["collection_page"] is browser.page
    assert browser.current_page_calls == 1
    assert calls["killed"] is True
