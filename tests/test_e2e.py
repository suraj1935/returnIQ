"""Playwright E2E: real server, real browser, real SQL tools. Skipped if browsers are unavailable."""
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")
ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def base_url():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--port", str(port)],
                            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    import httpx
    for _ in range(50):
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.2)
    else:
        proc.kill()
        pytest.fail("server did not start")
    yield f"http://127.0.0.1:{port}"
    proc.kill()


@pytest.fixture()
def page(base_url):
    with pw.sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as e:
            pytest.skip(f"chromium unavailable: {e}")
        pg = browser.new_page()
        pg.goto(base_url)
        yield pg
        browser.close()


def _ask(page, text):
    page.fill("#q", text)
    page.click("#ask")
    page.wait_for_function("document.querySelector('.answer') && document.querySelector('.answer').textContent !== 'Thinking...'")
    return page.locator(".answer").first.inner_text()


def test_grounded_answer_with_citation(page):
    ans = _ask(page, "What is the return rate in 2024?")
    assert re.search(r"\d+\.\d{2}%", ans)
    assert "Sources: return_rate" in page.locator(".cite").first.inner_text()


def test_out_of_scope_is_refused(page):
    ans = _ask(page, "what is the weather")
    assert "can't answer" in ans and page.locator(".cite").count() == 0


def test_empty_question_does_not_call_api(page):
    page.click("#ask")
    assert "Please enter a question" in page.locator(".msg").first.inner_text()


def test_no_data_period_stated_plainly(page):
    ans = _ask(page, "return rate in 2019")
    assert "can't be computed" in ans
