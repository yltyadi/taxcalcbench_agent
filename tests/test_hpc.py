"""Local checks for Slurm invocation and interruption; no provider requests."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(os.name != "posix", reason="Slurm uses POSIX signals and Bash")


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


@pytest.fixture
def fake_slurm(tmp_path):
    bin_dir = tmp_path / "fake bin"
    bin_dir.mkdir()
    fake_srun = bin_dir / "srun"
    fake_srun.write_text(
        '#!/usr/bin/env bash\n'
        'printf "%s\\0" "$@" > "$CAPTURE_ARGS"\n'
        'pwd > "$CAPTURE_CWD"\n'
    )
    fake_srun.chmod(0o755)
    submitted_from = tmp_path / "project with spaces"
    submitted_from.mkdir()
    capture_args = tmp_path / "arguments"
    capture_cwd = tmp_path / "cwd"
    environment = {
        **os.environ,
        "PATH": str(bin_dir) + os.pathsep + os.environ.get("PATH", ""),
        "SLURM_SUBMIT_DIR": str(submitted_from),
        "CAPTURE_ARGS": str(capture_args),
        "CAPTURE_CWD": str(capture_cwd),
    }
    return environment, capture_args, capture_cwd, submitted_from


def test_slurm_forwards_countries_and_quotes_storage_and_python_paths(tmp_path, fake_slurm):
    environment, capture_args, capture_cwd, submitted_from = fake_slurm
    python = str(tmp_path / "conda environment" / "bin" / "python")
    sources = str(tmp_path / "shared source files")
    outputs = str(tmp_path / "shared output files")
    environment.update(
        TAXCALCBENCH_PYTHON=python,
        TAXCALCBENCH_DATA_ROOT=sources,
        TAXCALCBENCH_OUTPUT_ROOT=outputs,
    )
    countries_and_options = ["configs/cn.json", "country configs/id.json", "--single-family"]

    result = subprocess.run(
        ["bash", str(ROOT / "scripts/hpc.slurm"), *countries_and_options],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=10,
    )

    assert result.returncode == 0, result.stderr
    assert capture_args.read_bytes().decode().split("\0")[:-1] == [
        python, "-m", "taxcalcbench", "run",
        "--source-root", sources, "--output-root", outputs,
        "--countries", *countries_and_options,
    ]
    assert Path(capture_cwd.read_text().strip()).resolve() == submitted_from.resolve()


def test_slurm_requires_country_arguments_before_starting_srun(tmp_path, fake_slurm):
    environment, capture_args, _, _ = fake_slurm
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/hpc.slurm")],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=10,
    )

    assert result.returncode == 1
    assert "Usage:" in result.stderr
    assert not capture_args.exists()
