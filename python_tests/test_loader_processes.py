import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import contextlib
import signal
import sys

import pytest
import imgread

PROBE = Path(__file__).with_name("loader_process_probe.py")


def probe(*args):
    if os.environ.get("IMGREAD_REQUIRE_DIAGNOSTICS") == "1":
        assert hasattr(imgread.Loader, "_debug_state"), "diagnostic wheel required"
    process = subprocess.Popen([sys.executable, str(PROBE), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stdout + stderr
        return json.loads(stdout)
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@pytest.mark.parametrize("method", mp.get_all_start_methods())
@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("buffer", [False, True])
def test_process_methods(method, warm, buffer):
    result = probe("process", "--method", method, *(["--warm"] if warm else []),
                   *(["--buffer"] if buffer else []))
    assert result["parent_before"]["pid"] == result["parent_after"]["pid"]
    phases = result["child"]["phases"]
    assert phases[0]["pid"] == phases[1]["pid"]
    assert phases[0].get("native_creations") == phases[1].get("native_creations")


@pytest.mark.parametrize("mode", ["pickle", "overlap", "fifo", "constructor-no-io", "import-no-torch"])
def test_lifecycle(mode):
    assert probe(mode)


@pytest.mark.parametrize("method", mp.get_all_start_methods())
def test_pickle_preserves_path_spelling_and_errors(method):
    assert probe("path-spelling", "--method", method)
