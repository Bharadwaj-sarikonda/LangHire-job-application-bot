
"""
Script 2: Apply to collected jobs with multiple concurrent workers.
One agent per job. Tracks status in jobs.json.

Usage:
  uv run python apply_jobs.py                          # 1 worker, easy apply
  uv run python apply_jobs.py --workers 3              # 3 concurrent workers
  uv run python apply_jobs.py --workers 2 --no-easy-apply  # non-easy apply
  uv run python apply_jobs.py --limit 10               # apply to max 10 jobs
"""
import argparse
import asyncio
import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

if not getattr(sys, 'frozen', False):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser_use import Agent, BrowserSession

try:
    import core.shared_config as config
    from core.shared_config import (
        BASE_DIR, BROWSER_PROFILE_DIR,
        JOBS_FILE, QA_FILE, CANDIDATE_PROFILE, LOGS_DIR, RESUME_PATH, SENSITIVE_DATA, BLOCKED_DOMAINS,
        AWS_PROFILE, AWS_REGION, MODEL_ID,
        load_json, save_json, refresh_credentials, credential_refresh_loop,
        build_memory_context, extract_from_history, normalize_question,
        is_profile_controlled_question,
        read_jobs, claim_job, update_job, get_memory_store,
    )
    from memory import extract_learnings_from_markers, extract_learnings_via_llm, store_learnings
    from memory.metrics import MetricsStore
    from core.agent_logger import on_step as _agent_on_step, on_done as _agent_on_done, log_run_start as _agent_log_start
    from core.browser_profiles import create_worker_profiles as _create_worker_profiles, remove_worker_profiles, remove_stale_worker_profiles as _remove_stale_worker_profiles
except ImportError:
    import backend.core.shared_config as config
    from backend.core.shared_config import (
        BASE_DIR, BROWSER_PROFILE_DIR,
        JOBS_FILE, QA_FILE, CANDIDATE_PROFILE, LOGS_DIR, RESUME_PATH, SENSITIVE_DATA, BLOCKED_DOMAINS,
        AWS_PROFILE, AWS_REGION, MODEL_ID,
        load_json, save_json, refresh_credentials, credential_refresh_loop,
        build_memory_context, extract_from_history, normalize_question,
        is_profile_controlled_question,
        read_jobs, claim_job, update_job, get_memory_store,
    )
    from backend.memory import extract_learnings_from_markers, extract_learnings_via_llm, store_learnings
    from backend.memory.metrics import MetricsStore
    from backend.core.agent_logger import on_step as _agent_on_step, on_done as _agent_on_done, log_run_start as _agent_log_start
    from backend.core.browser_profiles import create_worker_profiles as _create_worker_profiles, remove_worker_profiles, remove_stale_worker_profiles as _remove_stale_worker_profiles

# Lock for thread-safe QA file writes
_qa_lock = asyncio.Lock()


class _ProgressWatchdog:
    """Allow the extended step budget only while the page keeps changing."""

    BASE_STEPS = 120
    MAX_STEPS = 150
    REQUIRED_RECENT_PROGRESS = 12

    def __init__(self) -> None:
        self.last_step = 0
        self.last_progress_step = 0
        self._last_fingerprint = ""

    def observe(self, browser_state, step_num: int) -> None:
        """Record meaningful browser-state changes without retaining page data."""
        values = [
            getattr(browser_state, name, "")
            for name in ("url", "title", "tabs", "dom_state")
        ]
        fingerprint = hashlib.sha256(repr(values).encode("utf-8", "replace")).hexdigest()
        if fingerprint != self._last_fingerprint:
            self._last_fingerprint = fingerprint
            self.last_progress_step = step_num
        self.last_step = step_num

    async def should_stop(self) -> bool:
        """Stop at 120 unless the preceding form state was still progressing."""
        if self.last_step < self.BASE_STEPS:
            return False
        return self.last_step - self.last_progress_step >= self.REQUIRED_RECENT_PROGRESS


async def verify_batch_logins() -> bool:
    """Verify the shared LinkedIn/Gmail session once before worker profiles fork."""
    browser = BrowserSession(
        user_data_dir=str(BROWSER_PROFILE_DIR),
        chromium_sandbox=(sys.platform != "linux"),
        cross_origin_iframes=True,
    )
    try:
        agent = Agent(
            task=(
                "Verify the shared application-browser logins before a batch starts. "
                "Open LinkedIn feed and confirm the feed/home page is visible. Then open Gmail "
                "and confirm the inbox is visible. If either service shows a sign-in page, wait "
                "for the user to sign in manually, checking every 15 seconds for up to five minutes. "
                "Do not apply to jobs, change account settings, or read/send email. Close Gmail when done. "
                "Call done(success=true) only after both services are confirmed logged in."
            ),
            llm=config.get_llm(session_id=str(uuid4())),
            max_actions_per_step=1,
            use_vision="auto",
            llm_call_timeout=300,
            max_failures=5,
            max_history_items=10,
            message_compaction=True,
            loop_detection_enabled=True,
            loop_detection_window=5,
            browser_session=browser,
        )
        result = await agent.run(max_steps=40)
        if result.is_successful():
            print("✅ Shared LinkedIn and Gmail login check passed")
            return True
        errors = [error for error in result.errors() if error]
        print(f"❌ Shared login check failed: {(errors[-1] if errors else result.final_result() or 'not logged in')[:200]}")
        return False
    finally:
        try:
            await browser.close()
        except Exception as close_err:
            print(f"⚠️ Shared login browser cleanup error: {close_err}")


def create_worker_profiles(worker_count: int) -> tuple[Path | None, list[Path]]:
    """Create isolated profiles from the application's persistent login profile."""
    return _create_worker_profiles(BROWSER_PROFILE_DIR, worker_count)


def remove_stale_worker_profiles() -> None:
    """Remove temporary profiles left by a stopped or crashed desktop run."""
    _remove_stale_worker_profiles(BROWSER_PROFILE_DIR)


_ERROR_MAP = [
    ("'NoneType' object is not subscriptable", "Browser agent encountered an unexpected page state. The page may have changed or timed out."),
    ("'NoneType' object has no attribute", "Browser agent lost track of a page element. The site may have redirected or loaded slowly."),
    ("net::ERR_", "Network error — the page failed to load. Check your internet connection."),
    ("Timeout", "Operation timed out. The page took too long to respond."),
    ("ERR_CONNECTION_REFUSED", "Could not connect to the website. It may be temporarily down."),
    ("security token", "AWS credentials expired. They will be refreshed automatically on retry."),
    ("rate limit", "API rate limit reached. Wait a moment and try again."),
    ("context was destroyed", "Browser page closed unexpectedly during the application."),
    ("Target page, context or browser has been closed", "Browser closed unexpectedly during the application."),
]


def _friendly_error(raw: str) -> str:
    """Translate raw Python/browser errors into user-readable messages."""
    for pattern, friendly in _ERROR_MAP:
        if pattern.lower() in raw.lower():
            return friendly
    if len(raw) > 200 and ("Traceback" in raw or "Error:" in raw):
        return "Application failed due to an unexpected error. Check the Logs page for details."
    return raw


async def save_job_status(url: str, status: str, error: str | None = None):
    fields: dict = {"status": status}
    if error:
        fields["error"] = _friendly_error(error)
    else:
        fields["error"] = None
    if status == "applied":
        fields["applied_at"] = datetime.now(timezone.utc).isoformat()
    update_job(url, **fields)


async def save_new_qa(new_questions: dict, source_domain: str = ""):
    if not new_questions:
        return
    async with _qa_lock:
        store = get_memory_store()
        if store:
            for q, a in new_questions.items():
                if not is_profile_controlled_question(q):
                    store.qa_add(question=q, answer=a or "", source_domain=source_domain)
        else:
            qa = load_json(QA_FILE, {})
            existing_norms = {normalize_question(k) for k in qa}
            for q, a in new_questions.items():
                if not is_profile_controlled_question(q) and normalize_question(q) not in existing_norms:
                    qa[q] = a
                    existing_norms.add(normalize_question(q))
            save_json(QA_FILE, qa)


async def apply_to_job(job: dict, profile: dict, qa: dict, applied_labels: list[str], easy_apply: bool, worker_id: int, resume_path_override: str | None = None, browser_profile_dir: Path | None = None) -> str:
    """Apply to a single job. Returns final status."""
    url = job["url"]
    title = job.get("title", "Unknown")
    company = job.get("company", "Unknown")
    print(f"  🚀 [W{worker_id}] Starting: {title} at {company}")

    if not config.validate_job_url(url):
        await save_job_status(url, "blocked", "Invalid or internal URL")
        print(f"  🚫 [W{worker_id}] Blocked (invalid URL): {title} at {company}")
        return "blocked"

    if any(domain in url for domain in BLOCKED_DOMAINS):
        await save_job_status(url, "blocked", "Blocked domain")
        print(f"  🚫 [W{worker_id}] Blocked: {title} at {company}")
        return "blocked"

    if not claim_job(url):
        print(f"  ⏭️  [W{worker_id}] Skipped (already claimed): {title} at {company}")
        return "skipped"

    # Always refresh credentials before each job to avoid mid-run expiry
    refresh_credentials()

    session_id = str(uuid4())
    llm = config.get_llm(session_id=session_id)
    # Parallel workers receive isolated copies of the authenticated profile.
    browser = BrowserSession(
        user_data_dir=str(browser_profile_dir or BROWSER_PROFILE_DIR),
        chromium_sandbox=(sys.platform != "linux"),
        cross_origin_iframes=True,
    )
    mem_store = get_memory_store()
    # Count memories injected for metrics tracking
    domain = mem_store.extract_domain(url)
    memories_before = mem_store.get_domain_memories(url, limit=50)
    memories_injected_count = len(memories_before) if memories_before else 0
    memory = build_memory_context(profile, qa, applied_labels, job_url=url)
    run_started_at = datetime.now(timezone.utc)

    resume_path = resume_path_override or RESUME_PATH

    # Use tailored resume if available for this job
    try:
        from resume.tailor import get_tailored_resume_path
        tailored_path = get_tailored_resume_path(url)
        if tailored_path:
            resume_path = tailored_path
            print(f"  📄 [W{worker_id}] Using tailored resume: {tailored_path}")
    except ImportError:
        pass

    resume_file = Path(resume_path).expanduser()
    if not resume_file.is_file():
        await save_job_status(url, "failed", f"Resume file not found: {resume_path}")
        print(f"  ❌ [W{worker_id}] Resume file not found: {resume_path}")
        return "failed"
    resume_path = str(resume_file.resolve())

    # Profile email is for application forms; credentials email/password are for ATS login
    agent_sensitive_data = {
        "email": profile.get("email", "").strip(),
        "account_email": SENSITIVE_DATA.get("email", "").strip(),
        "password": SENSITIVE_DATA.get("password", ""),
    }
    if not agent_sensitive_data["email"]:
        agent_sensitive_data["email"] = agent_sensitive_data["account_email"]

    otp_instructions = (
        "\n\nOTP/VERIFICATION CODES: If ANY site asks for a verification code, OTP, or 2FA token:\n"
        "1. Choose the EMAIL option if given a choice\n"
        "2. Open a new tab and go to https://mail.google.com\n"
        "3. Use the newest verification email that arrived after this application displayed its verification request. Do not rely on the sender name because ATS senders can be shared. Gmail may group several codes into one conversation: inspect the message timestamps and expand/read the BOTTOM-most newest message in that thread. Never use the first code shown in a thread, a quoted earlier message, or a code from an older timestamp.\n"
        "4. Copy the code from that newest message, switch back to the application tab, and enter it. If rejected, refresh Gmail and use the newest subsequent code once.\n"
        "5. This is NOT a blocker — always attempt to retrieve the latest code from Gmail before giving up"
    )

    if easy_apply:
        apply_instructions = (
            f"Go to {url} on LinkedIn. Click Easy Apply and complete the application. "
            f"Use resume at {resume_path}. Auto-fill all fields from candidate profile."
            f"{otp_instructions}"
        )
    else:
        has_password = bool(agent_sensitive_data.get("password", "").strip())
        password_note = ""
        if not has_password:
            password_note = (
                "\n\nIMPORTANT: No password is configured in Settings. If the external site requires "
                "account creation or login:\n"
                "1. First check if you can apply as a guest (without creating an account)\n"
                "2. Try 'Sign in with LinkedIn' or 'Sign in with Google' buttons\n"
                "3. If no SSO option, CREATE A NEW ACCOUNT using <secret>account_email</secret> and a generated password "
                "(create a strong password like 'JobApp2026!')\n"
                "4. If account creation also fails, try 'Forgot password' → reset via email\n"
                "5. If nothing works after 3 attempts, report failure and move to the next job\n"
            )

        apply_instructions = (
            f"Go to {url} on LinkedIn. Click Apply and follow through to the external application page. "
            f"Use resume at {resume_path}. Auto-fill all fields from candidate profile.\n"
            f"- For resume/CV uploads, use only the browser's built-in upload_file action to attach the file at {resume_path}.\n\n"
            f"NAVIGATING EXTERNAL SITES:\n"
            f"- The LinkedIn 'Apply' button often opens a company careers page, NOT the application form directly.\n"
            f"- You MUST explore the landing page: look for 'Apply Now', 'Submit Application', or similar buttons.\n"
            f"- Scroll down — the apply button is often below the job description.\n"
            f"- If you see a job listing page, click on the specific job title first, then look for the apply button.\n"
            f"- Some sites require you to click through 2-3 pages before reaching the actual form.\n"
            f"- If the page looks blank or is loading, wait 3-5 seconds and try scrolling.\n"
            f"- NEVER give up just because the form isn't immediately visible — always explore the page first.\n\n"
            f"EMAIL USAGE:\n"
            f"- For APPLICATION FORM fields (contact email, email address, etc.): use <secret>email</secret>\n"
            f"- For LOGGING IN or CREATING ACCOUNTS on external ATS sites: use <secret>account_email</secret> and <secret>password</secret>\n"
            f"{password_note}"
            f"If it's a video funnel or recruitment pitch, report failure and stop. "
            f"If the external form is broken after 3 attempts, report failure and stop.\n\n"
            f"BLOCKED SITES — if redirected to any of these, immediately call done with success=false: {', '.join(BLOCKED_DOMAINS)}"
            f"{otp_instructions}"
        )

    _agent_log_start("apply", f"{title} at {company}")

    progress_watchdog = _ProgressWatchdog()
    MAX_STEPS = progress_watchdog.MAX_STEPS

    def _on_step_with_limit(browser_state, agent_output, step_num):
        _agent_on_step(browser_state, agent_output, step_num)
        progress_watchdog.observe(browser_state, step_num)

    agent = Agent(
        task=(
            f"{apply_instructions}\n\n"
            f"PERSISTENCE & EFFICIENCY:\n"
            f"- FILE UPLOADS: For every Resume/CV upload, use the built-in upload_file action on the file input's current index with this exact absolute path: {resume_path}. This action attaches the file directly to the webpage; it does NOT open a native macOS file chooser. NEVER use click, coordinate clicking, evaluate, or keyboard input on a Choose File/Upload button. If a native file chooser is already visible, do not continue the application underneath it; close it, then inspect the page and use upload_file. After upload_file, inspect the page and continue only when the uploaded filename or a successful upload state is visible.\n"
            f"- At the start of an application, if the form offers an explicit 'Autofill from resume' or equivalent resume-autofill control, use that control first to upload the resume. If no explicit resume-autofill control exists, upload the resume through the ordinary Resume/CV upload control first, since it may autofill fields. After either upload, wait for processing and inspect the refreshed form; fill only fields that remain empty or incorrect.\n"
            f"- Never blindly retry an interaction: if an action does not cause the expected page, field, menu, or button state change, treat that exact method as failed and switch methods immediately.\n"
            f"- After every important field, selection, navigation, or submit action, inspect the next browser state. Request a screenshot only when the state does not clearly identify the target or confirm the result, or before a visual coordinate click. Continue only after confirming the intended value, menu state, validation result, or page transition actually changed.\n"
            f"- Before clicking any checkbox, radio button, toggle, or selectable option, inspect its current state. If the desired answer is already selected or correct, do not click it again.\n"
            f"- CONSENT CHECKBOXES: If the form says 'check the box', 'accept', 'agree', 'privacy', or 'terms', DO NOT click Submit/Create Account again. Re-inspect the CURRENT page state and target the visible consent checkbox or its text label only. If an attempted checkbox index clicks a button or does not visibly toggle the control, that index is stale: abandon it, inspect again, then use one distinct method (the associated label, a targeted evaluate on that visible checkbox, or a visual coordinate click). Verify a checked state/aria-checked value/checkmark before submitting. After two distinct failed consent-control methods, report the form as blocked instead of looping.\n"
            f"- A missing indexed element is NOT a blocker.\n"
            f"- Prefer normal indexed click and input actions first when an element is available.\n"
            f"- For a native dropdown, call dropdown_options, then select_dropdown with the exact desired option, and verify the selected value. For a custom combobox, click it once, use its visible input or send_keys to choose an option, then verify the displayed selection.\n"
            f"- For React-controlled text fields, click/focus and input the value once. If the value does not persist after inspection, use send_keys to replace or confirm it once, then inspect again.\n"
            f"- For a visible control that does not respond to one indexed interaction, use ONE targeted evaluate attempt for that currently visible control. If it does not change state, use one visual coordinate click; request a screenshot first only if its location is not already clear.\n"
            f"- If coordinate clicking fails or is unavailable, use ONE deeper iframe/shadow-DOM evaluate attempt when that is a plausible cause, then inspect the result. Do not cycle through multiple JavaScript selectors.\n"
            f"- If a button or menu is focused but still needs confirmation, use send_keys for Enter or Space once, then verify the result.\n"
            f"- Before reporting a control failure, try up to three distinct appropriate methods when available, but never repeat the same index, coordinates, keyboard command, or evaluate approach when the immediately preceding state shows it did nothing.\n"
            f"- Do NOT call done with success=false while the application form is still open unless there is a genuine blocker.\n"
            f"- Genuine blockers include: CAPTCHA that cannot be completed, missing required factual information that must not be guessed, an inaccessible/broken page after multiple recovery attempts, or an unrecoverable browser failure.\n"
            f"- You have a maximum of {MAX_STEPS} steps total. Budget your steps wisely.\n\n"
            f"TRACKING: Include in memory field after submission:\n"
            f'@@JOB_APPLIED: {{"title": "{title}", "company": "{company}", "location": "{job.get("location", "")}"}}\n'
            f"For each form question: @@QUESTION: {{\"question\": \"...\", \"answer\": \"...\", \"type\": \"...\"}}"
        ),
        llm=llm,
        max_actions_per_step=1,
        use_vision="auto",
        llm_call_timeout=300,  # 5 minutes per step
        max_failures=10,
        max_history_items=10,
        message_compaction=True,
        loop_detection_enabled=True,
        loop_detection_window=5,
        browser_session=browser,
        sensitive_data=agent_sensitive_data,
        extend_system_message=memory,
        available_file_paths=[resume_path],
        save_conversation_path=str(LOGS_DIR / f"apply_{company.replace(' ', '_')}_{title.replace(' ', '_')[:30]}"),
        calculate_cost=True,
        register_new_step_callback=_on_step_with_limit,
        register_done_callback=_agent_on_done,
        register_should_stop_callback=progress_watchdog.should_stop,
    )
    agent.tools.set_coordinate_clicking(True)

    try:
        result = await agent.run(max_steps=MAX_STEPS)

        # Extract Q&A from history
        _, new_questions = extract_from_history(result)
        domain = mem_store.extract_domain(url)
        await save_new_qa(new_questions, source_domain=domain)

        # Determine success/failure
        success = result.is_successful()
        errors = [e for e in result.errors() if e]

        # ── Memory extraction (self-learning) ─────────────────────────────
        mem_store = get_memory_store()
        # 1. Marker-based: extract @@LEARNING tags the agent emitted
        marker_learnings = extract_learnings_from_markers(result)
        if marker_learnings:
            store_learnings(mem_store, marker_learnings, url, success)
        # 2. LLM-based: use the configured LLM to summarise the run into procedural learnings
        llm_learnings = []
        try:
            def _llm_call(prompt):
                """Use the user's configured LLM for memory extraction."""
                import asyncio
                from browser_use.llm.messages import UserMessage

                async def _call():
                    extraction_llm = config.get_llm()
                    resp = await asyncio.wait_for(
                        extraction_llm.ainvoke([UserMessage(content=prompt)]),
                        timeout=30,
                    )
                    return resp.completion if hasattr(resp, 'completion') else (resp.content if hasattr(resp, 'content') else str(resp))

                return asyncio.run(_call())

            llm_learnings = await asyncio.to_thread(
                extract_learnings_via_llm,
                result,
                job_url=url,
                job_title=title,
                success=success,
                llm_call=_llm_call,
            )
            if llm_learnings:
                store_learnings(mem_store, llm_learnings, url, success)
        except Exception as mem_err:
            print(f"    ⚠️  [W{worker_id}] Memory extraction failed (non-fatal): {mem_err}")

        # ── Record metrics ────────────────────────────────────────────────
        run_finished_at = datetime.now(timezone.utc)
        memories_extracted_count = len(marker_learnings) + len(llm_learnings)
        try:
            MetricsStore().record_run(
                job_url=url, job_title=title, company=company,
                website_domain=domain, ats_platform=mem_store.detect_ats_platform(domain),
                success=success, started_at=run_started_at, finished_at=run_finished_at,
                step_count=len(result.history),
                memories_injected=memories_injected_count,
                memories_extracted=memories_extracted_count,
                error_message=(errors[-1][:2000] if errors else None) if not success else None,
            )
        except Exception as metrics_err:
            print(f"    ⚠️  [W{worker_id}] Metrics recording failed (non-fatal): {metrics_err}")

        if success:
            await save_job_status(url, "applied")
            print(f"  ✅ [W{worker_id}] Applied: {title} at {company}")
            return "applied"
        else:
            error_msg = errors[-1] if errors else (result.final_result() or "Agent reported failure")
            await save_job_status(url, "failed", error_msg[:2000])
            print(f"  ❌ [W{worker_id}] Failed: {title} at {company} — {error_msg[:100]}")
            return "failed"

    except Exception as e:
        error_str = str(e)
        if "security token" in error_str.lower() or "expired" in error_str.lower():
            refresh_credentials()
            await save_job_status(url, "pending", "credentials_expired_retry")
            print(f"  🔑 [W{worker_id}] Credentials expired on: {title} at {company} — refreshed, will retry")
            return "retry"
        await save_job_status(url, "failed", error_str[:2000])
        print(f"  ❌ [W{worker_id}] Error: {title} at {company} — {error_str[:100]}")
        return "failed"
    finally:
        try:
            await browser.close()
        except Exception as close_err:
            print(f"    ⚠️  [W{worker_id}] Browser cleanup error: {close_err}")


async def worker(name: str, worker_id: int, queue: asyncio.Queue, profile: dict, qa: dict, applied_labels: list, easy_apply: bool, stats: dict, cancel_flag: dict | None = None, browser_profile_dir: Path | None = None):
    """Worker that pulls jobs from queue and applies."""
    while True:
        if cancel_flag and cancel_flag.get("cancel_requested"):
            print(f"  🛑 [{name}] Stop requested — halting")
            break
        try:
            job = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        job_easy_apply = job.get("easy_apply", easy_apply) if job.get("easy_apply") is not None else easy_apply
        status = await apply_to_job(
            job, profile, qa, applied_labels, job_easy_apply, worker_id,
            browser_profile_dir=browser_profile_dir,
        )
        stats[status] = stats.get(status, 0) + 1

        # On retry, put back in queue
        if status == "retry":
            queue.put_nowait(job)

        queue.task_done()


async def main():
    parser = argparse.ArgumentParser(description="Apply to collected jobs")
    parser.add_argument("--workers", type=int, default=1, help="Number of concurrent workers")
    parser.add_argument("--limit", type=int, help="Max jobs to process")
    parser.add_argument("--easy-apply", dest="easy_apply", action="store_true", default=True)
    parser.add_argument("--no-easy-apply", dest="easy_apply", action="store_false")
    args = parser.parse_args()

    jobs = load_json(JOBS_FILE, {})
    profile = load_json(CANDIDATE_PROFILE, {})
    qa = load_json(QA_FILE, {})
    LOGS_DIR.mkdir(exist_ok=True)

    # Filter pending jobs by type
    pending = [
        j for j in jobs.values()
        if j.get("status") == "pending"
        and (j.get("easy_apply") is True) == args.easy_apply
    ]

    if args.limit:
        pending = pending[:args.limit]

    if not pending:
        print("No pending jobs to apply to. Run collect_jobs.py first.")
        return

    applied_labels = [
        f"{j.get('title','')} at {j.get('company','')}"
        for j in jobs.values() if j.get("status") == "applied"
    ]

    mode = "Easy Apply" if args.easy_apply else "Non-Easy Apply"
    print(f"Applying to {len(pending)} {mode} jobs with {args.workers} worker(s)\n")

    queue = asyncio.Queue()
    for job in pending:
        queue.put_nowait(job)

    stats = {}
    num_workers = min(args.workers, len(pending))

    print("Checking shared LinkedIn and Gmail login once before starting workers...")
    if not await verify_batch_logins():
        print("Login check did not complete. No jobs were started; sign in and run again.")
        return

    run_dir, worker_profiles = create_worker_profiles(num_workers)
    cred_task = asyncio.create_task(credential_refresh_loop(14))
    try:
        workers = []
        for i, worker_profile in enumerate(worker_profiles):
            if i > 0:
                await asyncio.sleep(5)  # stagger browser launches
            workers.append(asyncio.create_task(
                worker(
                    f"W{i+1}", i+1, queue, profile, qa, applied_labels,
                    args.easy_apply, stats, browser_profile_dir=worker_profile,
                )
            ))
        await asyncio.gather(*workers)
    finally:
        cred_task.cancel()
        await asyncio.gather(cred_task, return_exceptions=True)
        remove_worker_profiles(run_dir)

    print(f"\n{'='*60}")
    print(f"Results: {stats}")
    total_applied = sum(1 for j in load_json(JOBS_FILE, {}).values() if j.get("status") == "applied")
    print(f"Total applied across all runs: {total_applied}")

    # Memory stats
    mem_stats = get_memory_store().get_stats()
    print(f"🧠 Agent memory: {mem_stats['total_memories']} memories across {mem_stats['unique_domains']} domains")


if __name__ == "__main__":
    asyncio.run(main())
