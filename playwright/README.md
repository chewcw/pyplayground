Playwright example
===================

This folder shows a minimal example demonstrating how to programmatically use the `PlaywrightTool` facade in `app.services.browser`.

Quick steps:

1. Create a Python virtualenv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install playwright
playwright install
```

2. Run the example (headful to perform login and persist profile):

```bash
python playground/playwright/run_example.py
```

3. Re-run headless (modify the script to pass `headless=True`) to verify that the profile or exported `playwright_storage.json` restores the logged-in state.

Files:
- `example.html` — local test page with a simple login form and localStorage persistence.
- `run_example.py` — script that runs the sample workflow using `PlaywrightTool`.
- `.playwright_profile/` — created by the example when you run it; contains the persistent browser profile.
- `playwright_storage.json` — exported storage state for CI portability (created by the script).

Notes:
- For CI, prefer exporting `storage_state` and importing it in the CI run rather than relying on persistent profile folders.
