"""Local checks for POSIX process interruption and graceful checkpoint preservation."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX signals required")


def test_sigterm_preserves_checkpoint_and_unwinds_async_execution(tmp_path):
    checkpoint = tmp_path / "work" / "family.json"
    cleanup = tmp_path / "closed.txt"
    child = textwrap.dedent("""\
        import asyncio
        import json
        import os
        from pathlib import Path
        import signal
        import sys
        from taxcalcbench import cli

        checkpoint, cleanup = map(Path, sys.argv[1:])
        cli.load_dotenv = lambda *args, **kwargs: None

        async def execute(args):
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text(json.dumps({"family_id": "A-F01", "status": "approved"}))
            try:
                asyncio.get_running_loop().call_soon(os.kill, os.getpid(), signal.SIGTERM)
                await asyncio.Event().wait()
            finally:
                cleanup.write_text("closed")

        cli.execute = execute
        cli.main(["run", "--country", "unused.json", "--output", str(checkpoint.parent)])
        """)
    result = subprocess.run(
        [sys.executable, "-c", child, str(checkpoint), str(cleanup)],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )

    assert result.returncode == 130, result.stderr
    assert "Interrupted; completed families remain" in result.stderr
    assert json.loads(checkpoint.read_text()) == {"family_id": "A-F01", "status": "approved"}
    assert cleanup.read_text() == "closed"

