import json

from cli.collect_jobs import _job_from_step


def _marker(job_id="4463624853", **overrides):
    job = {
        "title": "Software Engineer",
        "company": "Avante",
        "location": "Seattle, WA",
        "url": f"https://www.linkedin.com/jobs/view/{job_id}/",
        "easy_apply": True,
        **overrides,
    }
    return "@@JOB_FOUND: " + json.dumps(job)


def test_collection_accepts_complete_marker_for_same_live_job():
    job = _job_from_step(
        _marker(),
        "https://www.linkedin.com/jobs/search/?currentJobId=4463624853",
    )
    assert job == {
        "title": "Software Engineer",
        "company": "Avante",
        "location": "Seattle, WA",
        "url": "https://www.linkedin.com/jobs/view/4463624853/",
        "easy_apply": True,
    }


def test_collection_rejects_metadata_marker_for_different_live_job():
    assert _job_from_step(
        _marker("4463624853"),
        "https://www.linkedin.com/jobs/search/?currentJobId=4999999999",
    ) is None


def test_collection_rejects_incomplete_or_multiple_markers():
    live_url = "https://www.linkedin.com/jobs/view/4463624853/"
    assert _job_from_step(_marker(company=""), live_url) is None
    assert _job_from_step(_marker() + "\n" + _marker(), live_url) is None


def test_collection_allows_missing_location():
    job = _job_from_step(
        _marker(location=""),
        "https://www.linkedin.com/jobs/view/4463624853/",
    )
    assert job is not None
    assert job["location"] == ""
