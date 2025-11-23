from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional
import textwrap

from utils.code_postprocess import inject_main_for_testing, has_dunder_main

def run_python_sandboxed(code: str, timeout: int = 8, harness: Optional[str] = None) -> Dict[str, Any]:
    # normalize inputs
    code = textwrap.dedent(code or "").strip()
    harness = textwrap.dedent(harness or "").strip() if harness else None
    script: str
    if harness:
        # compose full script from function and provided harness
        script = (code + "\n\n" + harness).rstrip() + "\n"
    else:
        script = (code + "\n").rstrip() + "\n"

    # Ensure an entrypoint exists; if missing, inject one deterministically.
    if not has_dunder_main(script):
        if harness:
            # Wrap the provided harness under an injected __main__ block
            script = inject_main_for_testing(code, harness)
        else:
            # Minimal guard to satisfy execution; no-op body
            script = inject_main_for_testing(code, "pass")

    # validate
    code = script
    if not code.strip():
        return {"ok": False, "error": "EMPTY_CODE", "hint": "Provide 'code' with a complete Python script."}
    try:
        compile(code, "snippet.py", "exec")
    except SyntaxError as e:
        return {
            "ok": False,
            "error": "SYNTAX_ERROR",
            "stderr": f"{e.__class__.__name__}: {e}",
            "hint": "Ensure Action Input contains valid JSON and 'code' is a complete script with balanced quotes/indentation."
        }
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
        "description": "Execute Python in a minimal sandbox. Args: code (str), harness (optional str), timeout (int).",
    }
}


