#!/usr/bin/env python3
"""Run an example workflow using the PlaywrightSession facade.

Executes a simple login flow on the local example.html page, demonstrates
persisting a profile to `./.playwright_profile` and exporting storage state.
"""
import asyncio
from pathlib import Path

from app.services.browser import PlaywrightSession

EXAMPLE_DIR = Path(__file__).parent
HTML_PATH = EXAMPLE_DIR / "example.html"
FILE_URI = HTML_PATH.resolve().as_uri()
PROFILE_DIR = EXAMPLE_DIR / ".playwright_profile"
STORAGE_JSON = EXAMPLE_DIR / "playwright_storage.json"

WORKFLOW = {
    "name": "login_example",
    "actions": [
        {"type": "navigate", "value": "https://erp.smartmes.com/platform/Frames/Login.aspx?ReturnUrl=%2fplatform"},
        {"type": "wait", "selector": "#btnLoginFederation"},
        {"type": "fill", "selector": "#username", "value": "alice"},
        {"type": "fill", "selector": "#password", "value": "secret"},
        {"type": "click", "selector": "#loginBtn"},
        {"type": "wait", "selector": "#welcome"},
        {"type": "extract", "selector": "#welcome", "output_key": "greeting"},
        {"type": "screenshot", "value": str((EXAMPLE_DIR / "after_login.png").resolve())},
    ],
}

async def main() -> None:
    async with PlaywrightSession(headless=False, user_data_dir=str(PROFILE_DIR)) as tool:
        print("Running login workflow (headful). This will save profile to", PROFILE_DIR)
        await tool.run_workflow(WORKFLOW)
        print("Exporting storage state to", STORAGE_JSON)
        await tool.export_storage_state(str(STORAGE_JSON))
        print("Done. Re-run with headless=True to verify profile reuse.")


if __name__ == "__main__":
    asyncio.run(main())
