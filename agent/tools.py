from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any, Dict


def run_python_sandboxed(code: str, timeout: int = 8) -> Dict[str, Any]:
    code = (code or "").strip() + "\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        script_path = os.path.join(tmpdir, "snippet.py")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(code)
        
        # isolate the code from the environment
        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONSAFEPATH"] = "1"

        # run the code in the sandbox
        cmd = [sys.executable, "-I", "-S", script_path]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
            )
            return {"ok": True, "exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
        except subprocess.TimeoutExpired as e:
            return {"ok": False, "error": "Timeout", "stdout": e.stdout or "", "stderr": e.stderr or ""}
        except Exception as e:
            return {"ok": False, "error": repr(e)}


TOOLS = {
    "code_interpreter": {
        "fn": run_python_sandboxed,
        "description": "Execute Python in a minimal sandbox. Args: code (str), timeout (int).",
    }
}


