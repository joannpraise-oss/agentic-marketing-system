import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_main_imports_in_a_fresh_interpreter():
    # Railway crashed on import (NameError: Optional) while the suite passed: the tests had already
    # imported everything, and Python 3.14 doesn't evaluate annotations at import. A clean process
    # on the pinned version catches it.
    env = {**os.environ, "DATABASE_URL": "sqlite://", "OPENAI_API_KEY": "sk-test-not-real",
           "LANGCHAIN_TRACING_V2": "false", "LANGSMITH_TRACING": "false"}

    result = subprocess.run([sys.executable, "-c", "import main"], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=60)

    assert result.returncode == 0, result.stderr


def test_tests_run_on_the_pinned_python():
    pinned = (ROOT / ".python-version").read_text().strip()

    assert f"{sys.version_info.major}.{sys.version_info.minor}" == pinned
