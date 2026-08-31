"""Shared config, utilities, and credential management."""
import asyncio
import json
import re
import subprocess
import sys
import threading
from pathlib import Path

import boto3
from filelock import FileLock
from browser_use.llm import ChatAWSBedrock

try:
    from core.config import get_data_dir
    from memory import MemoryStore
except ImportError:
    from backend.core.config import get_data_dir
    from backend.memory import MemoryStore

_SOURCE_DIR = Path(__file__).resolve().parent.parent.parent  # project root (or temp dir if frozen)

DATA_DIR = get_data_dir()

# When frozen (PyInstaller), BASE_DIR would be a temp dir — use DATA_DIR instead
BASE_DIR = DATA_DIR if getattr(sys, 'frozen', False) else _SOURCE_DIR

# Use OS data dir for settings/profile (written by UI), project root for jobs/logs
JOBS_FILE = DATA_DIR / "jobs.json"
JOBS_LOCK = DATA_DIR / "jobs.json.lock"
QA_FILE = DATA_DIR / "qa_repository.json"
CANDIDATE_PROFILE = DATA_DIR / "candidate_profile.json"
LOGS_DIR = BASE_DIR / "logs"
RESUMES_DIR = BASE_DIR / "resumes"

# Browser profile ALWAYS in OS data dir (must match backend/main.py login endpoint)
BROWSER_PROFILE_DIR = DATA_DIR / "browser_profile"

AWS_PROFILE = "default"
AWS_REGION = "us-west-2"
MODEL_ID = "us.anthropic.claude-sonnet-4-6"
ADA_CMD: tuple[str, ...] = ()  # Only needed for Amazon internal credential refresh

# Load settings from OS data dir (written by desktop app UI) if available
_settings_file = DATA_DIR / "settings.json"
_ui_settings = json.loads(_settings_file.read_text()) if _settings_file.exists() else {}
SENSITIVE_DATA = _ui_settings.get("sensitive_data", {"email": "", "password": ""})

# Fall back to profile email if sensitive_data email is blank
if not SENSITIVE_DATA.get("email", "").strip():
    _profile_file = DATA_DIR / "candidate_profile.json"
    if _profile_file.exists():
        _profile_data = json.loads(_profile_file.read_text())
        _profile_email = _profile_data.get("email", "").strip()
        if _profile_email:
            SENSITIVE_DATA["email"] = _profile_email

RESUME_PATH = _ui_settings.get("resume_path", "")
BLOCKED_DOMAINS = _ui_settings.get("blocked_domains", ["meeboss.com"])

_PRIVATE_IP_PREFIXES = ("127.", "10.", "192.168.", "172.16.", "172.17.", "172.18.",
                        "172.19.", "172.20.", "172.21.", "172.22.", "172.23.",
                        "172.24.", "172.25.", "172.26.", "172.27.", "172.28.",
                        "172.29.", "172.30.", "172.31.", "0.", "169.254.")

def validate_job_url(url: str) -> bool:
    """Reject URLs pointing to private/internal networks (SSRF prevention)."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        if not host or not parsed.scheme.startswith("http"):
            return False
        if host in ("localhost", "0.0.0.0", "[::]", "[::1]"):
            return False
        if any(host.startswith(p) for p in _PRIVATE_IP_PREFIXES):
            return False
        return True
    except Exception:
        return False

# ── Singleton memory store ────────────────────────────────────────────────────
_memory_store: MemoryStore | None = None


def get_memory_store() -> MemoryStore:
    """Get or create the singleton memory store instance."""
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store


def load_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text())
    return default if default is not None else []


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def read_jobs() -> dict:
    """Read jobs.json with cross-process lock."""
    with FileLock(JOBS_LOCK):
        return load_json(JOBS_FILE, {})


def write_jobs(jobs: dict):
    """Write jobs.json with cross-process lock."""
    with FileLock(JOBS_LOCK):
        save_json(JOBS_FILE, jobs)


def update_job(url: str, **fields):
    """Atomically update a single job entry."""
    with FileLock(JOBS_LOCK):
        jobs = load_json(JOBS_FILE, {})
        if url in jobs:
            jobs[url].update(fields)
            save_json(JOBS_FILE, jobs)


_claim_lock = threading.Lock()


def claim_job(url: str) -> bool:
    """Atomically claim a pending job. Returns True if claimed, False if already taken.
    Uses both FileLock (cross-process) and threading lock (in-process workers)."""
    with _claim_lock:
        with FileLock(JOBS_LOCK):
            jobs = load_json(JOBS_FILE, {})
            if url in jobs and jobs[url].get("status") == "pending":
                jobs[url]["status"] = "in_progress"
                save_json(JOBS_FILE, jobs)
                return True
            return False


def refresh_credentials():
    """Run ada credentials update and return True on success."""
    if not ADA_CMD:
        # No credential refresh command configured — skip silently
        # Users should configure AWS credentials via the Settings UI or aws cli
        return True
    print("🔑 Refreshing AWS credentials...")
    try:
        subprocess.run(ADA_CMD, check=True, timeout=30)
        print("✅ Credentials refreshed")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"❌ Credential refresh failed: {e}", file=sys.stderr)
        return False


async def credential_refresh_loop(interval_minutes: int = 14):
    """Background task that refreshes credentials on a timer. Cancels with parent."""
    while True:
        await asyncio.sleep(interval_minutes * 60)
        refresh_credentials()


def get_llm(session_id: str | None = None) -> ChatAWSBedrock:
    """Create a fresh LLM client with current credentials."""
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return ChatAWSBedrock(model=MODEL_ID, session=session)


def normalize_question(q: str) -> str:
    return re.sub(r"[^\w\s]", "", q.lower()).strip()


def build_memory_context(
    profile: dict,
    qa: dict,
    applied_labels: list[str] | None = None,
    job_url: str | None = None,
) -> str:
    """Build the system message context with candidate profile, Q&A bank, and per-website learnings."""
    parts = []

    resume_context = (BASE_DIR / "resume.md").read_text(encoding="utf-8").strip()

    parts.append(
        f"""FULL CANDIDATE RESUME:
        {resume_context}

        Use this resume as factual candidate context when answering application questions.
        Do not invent experience not supported by the resume, candidate profile, or saved Q&A."""
            )

    # Salary formatting (country-aware)
    sal = profile.get('salary_expectation', {})
    sal_currency = sal.get('currency', 'USD') or 'USD'
    sal_min = sal.get('min', 0) or 0
    sal_max = sal.get('max', 0) or 0
    sal_period = sal.get('period', 'annual')
    salary_str = f"{sal_currency} {sal_min:,}-{sal_max:,} ({sal_period})" if sal_min else "Not specified"

    # Country-aware date format
    date_format = profile.get('date_format', '') or 'MM/DD/YYYY'

    profile_lines = [
        "CANDIDATE PROFILE:",

        f"Full Name: {profile.get('name', '')}",
        f"First Name: {profile.get('first_name', '')}",
        f"Middle Name: {profile.get('middle_name', '')}",
        f"Last Name: {profile.get('last_name', '')}",

        f"Email: {profile.get('email', '')}, Phone: {profile.get('phone_country_code', '')}{profile.get('phone', '')}",

        f"Location: {profile.get('address', {}).get('city', '')}, "
        f"{profile.get('address', {}).get('state', '')} "
        f"{profile.get('address', {}).get('zip', '')} "
        f"{profile.get('address', {}).get('country', '')}".strip(),

        f"Work Authorization: {profile.get('work_authorization', '')}, "
        f"Visa Sponsorship Needed: {profile.get('visa_sponsorship_needed', False)}",

        f"Willing to Relocate: {profile.get('willing_to_relocate', False)}, "
        f"Preferred Work Mode: {profile.get('preferred_work_mode', '')}",

        f"Years of Experience: {profile.get('years_of_experience', 0)}",
        f"Years of Machine Learning Experience: {profile.get('years_of_machine_learning_experience', 0)}",
        f"Years of Generative AI Experience: {profile.get('years_of_generative_ai_experience', 0)}",
        f"Years of Python Experience: {profile.get('years_of_python_experience', 0)}",
        f"Years of AWS Experience: {profile.get('years_of_aws_experience', 0)}",
        f"Years of Azure Experience: {profile.get('years_of_azure_experience', 0)}",

        f"Education: "
        f"{profile.get('education', {}).get('degree', '')} from "
        f"{profile.get('education', {}).get('school', '')} "
        f"({profile.get('education', {}).get('graduation', '')})",

        f"Undergraduate Education: "
        f"{profile.get('undergraduate_degree', '')} from "
        f"{profile.get('undergraduate_school', '')} "
        f"({profile.get('undergraduate_graduation', '')})",

        f"Current Role: {profile.get('current_role', '')}",
        f"Target Locations: {', '.join(profile.get('target_locations', []))}",
        f"Languages: {', '.join(profile.get('languages', []))}",
        f"Skills: {', '.join(profile.get('skills', []))}",

        f"Salary: {salary_str}",
    ]
    if profile.get('nationality'):
     profile_lines.append(f"Nationality: {profile['nationality']}")

    if profile.get('country_of_birth'):
        profile_lines.append(f"Country of Birth: {profile['country_of_birth']}")

    if profile.get('ethnicity'):
        profile_lines.append(f"Ethnicity: {profile['ethnicity']}")

    if profile.get('race'):
        profile_lines.append(f"Race: {profile['race']}")

    if profile.get('gender'):
        profile_lines.append(f"Gender: {profile['gender']}")

    if profile.get('marital_status'):
        profile_lines.append(f"Marital Status: {profile['marital_status']}")

    if profile.get('hispanic_latino'):
        profile_lines.append(
            f"Hispanic or Latino: {profile['hispanic_latino']}"
        )

    if profile.get('disability_status'):
        profile_lines.append(
            f"Disability Status: {profile['disability_status']}"
        )

    if profile.get('veteran_status'):
        profile_lines.append(
            f"Veteran Status: {profile['veteran_status']}"
        )

    if profile.get('date_of_birth'):
        profile_lines.append(
            f"Date of Birth: {profile['date_of_birth']}"
        )

    if profile.get('linkedin_url'):
        profile_lines.append(
            f"LinkedIn URL: {profile['linkedin_url']}"
        )

    if profile.get('github_url'):
        profile_lines.append(
            f"GitHub URL: {profile['github_url']}"
        )

    if profile.get('portfolio_url'):
        profile_lines.append(
            f"Portfolio URL: {profile['portfolio_url']}"
        )

    if profile.get('notice_period'):
        profile_lines.append(
            f"Notice Period: {profile['notice_period']}"
        )

    if profile.get('notes'):
        profile_lines.append(
            f"Notes: {profile['notes']}"
        )

    parts.append("\n".join(profile_lines))
    parts.append(
    "PROFILE SOURCE-OF-TRUTH INSTRUCTIONS:\n"
    "For factual personal, demographic, education, work authorization, "
    "contact, and years-of-experience questions, use the values in "
    "CANDIDATE PROFILE as the source of truth.\n"
    "Do not infer or override these values from the resume when an explicit "
    "profile value is provided.\n"
    "For demographic questions such as race, ethnicity, gender, Hispanic or "
    "Latino status, disability status, veteran status, marital status, and "
    "country of birth, use the saved profile value when available.\n"
    "Do not choose 'Prefer not to disclose', 'Decline to self-identify', or "
    "similar options when an explicit profile value is available."
)

    parts.append(
        "SCREENING-QUESTION ANSWERING INSTRUCTIONS:\n"
        "For open-ended application and technical screening questions, write a concise, recruiter-ready answer tailored to the question. "
        "Use the most relevant evidence from the resume, CANDIDATE PROFILE, and saved Q&A: name the specific technologies, services, responsibilities, outcomes, and scope that are actually supported by those sources. "
        "For example, when asked about AWS experience, answer whether the candidate has it and mention only the AWS services and work described in the candidate materials. "
        "Present supported experience clearly and confidently, but never invent, exaggerate, or imply hands-on experience with technologies, projects, metrics, or responsibilities that are not supported. "
        "Use a saved Q&A answer when it directly answers the question; otherwise synthesize the best accurate answer from the candidate materials."
    )

    # Country-specific instructions for the agent
    country_instructions = []
    country_instructions.append(f"DATE FORMAT: When filling date fields, use {date_format} format.")
    if profile.get('notice_period'):
        country_instructions.append(f"NOTICE PERIOD: If asked about notice period or availability, answer: {profile['notice_period']}")
    if profile.get('nationality'):
        country_instructions.append(f"NATIONALITY: If asked about nationality, answer: {profile['nationality']}")
    if profile.get('cover_letter'):
        country_instructions.append(f"COVER LETTER: If a cover letter is requested, use:\n{profile['cover_letter']}")
    if country_instructions:
        parts.append("COUNTRY-SPECIFIC INSTRUCTIONS:\n" + "\n".join(country_instructions))

    if applied_labels:
        parts.append("Already applied — SKIP:\n" + "\n".join(f"- {j}" for j in applied_labels))

    # Try SQLite Q&A first, fall back to passed-in dict
    qa_for_prompt = qa
    try:
        store = get_memory_store()
        if store:
            db_qa = store.qa_get_all_for_prompt()
            if db_qa:
                qa_for_prompt = db_qa
    except Exception:
        pass
    if qa_for_prompt:
        qa_list = "\n".join(f'Q: {q}\nA: {a}' for q, a in qa_for_prompt.items() if a)
        if qa_list:
            parts.append(f"Pre-filled answers for application questions:\n{qa_list}")

    # ── Per-website memory injection ──────────────────────────────────────
    if job_url:
        store = get_memory_store()
        memories = store.get_domain_memories(job_url, limit=20)
        if memories:
            domain = store.extract_domain(job_url)
            mem_count = len(memories)
            print(f"    🧠 Injecting {mem_count} memories for {domain}")
            parts.append(store.format_for_prompt(memories))

    parts.append(
        "TRACKING INSTRUCTIONS:\n"
        "After each successful application: @@JOB_APPLIED: {\"title\": \"...\", \"company\": \"...\", \"location\": \"...\"}\n"
        "For each form question encountered: @@QUESTION: {\"question\": \"...\", \"answer\": \"...\", \"type\": \"text|dropdown|radio|checkbox\"}\n\n"
        "SELF-LEARNING — report observations about THIS WEBSITE's UI/flow as you navigate:\n"
        "@@LEARNING: {\"domain\": \"<website domain>\", \"category\": \"navigation|form_strategy|element_interaction|failure_recovery|site_structure|qa_pattern\", \"insight\": \"<specific actionable observation>\"}\n"
        "Examples of good learnings:\n"
        "- Navigation: 'Easy Apply opens a modal overlay, don't navigate away from the page'\n"
        "- Element interaction: 'The checkbox is inside a scrollable div, must scroll to find it'\n"
        "- Form strategy: 'This ATS splits the form into 4 steps: Personal → Resume → Questions → Review'\n"
        "Report at least 2-3 learnings per application run."
    )
    return "\n\n".join(parts)


def extract_from_history(result):
    """Extract applied jobs and questions from agent history."""
    jobs, questions, seen = [], {}, set()
    for item in result.history:
        if not item.model_output:
            continue
        memory = item.model_output.memory or ""
        for m in re.finditer(r"@@JOB_APPLIED:\s*(\{[^}]{1,2000}\})", memory):
            try:
                j = json.loads(m.group(1))
                jobs.append(f"{j.get('title','')} at {j.get('company','')} - {j.get('location','')}")
            except json.JSONDecodeError:
                pass
        for m in re.finditer(r"@@QUESTION:\s*(\{[^}]{1,2000}\})", memory):
            try:
                q = json.loads(m.group(1))
                qtext, ans = q.get("question", "").strip(), q.get("answer", "").strip()
                norm = normalize_question(qtext)
                if qtext and norm not in seen:
                    seen.add(norm)
                    questions[qtext] = ans
            except json.JSONDecodeError:
                pass
        # Fallback
        if not jobs and any(kw in memory.lower() for kw in ["application submitted", "successfully applied"]):
            for pat in [r"applied to (.+?) via", r"Application submitted for (.+?) via"]:
                match = re.search(pat, memory, re.IGNORECASE)
                if match:
                    jobs.append(match.group(1).strip())
                    break
    return list(dict.fromkeys(jobs)), questions
