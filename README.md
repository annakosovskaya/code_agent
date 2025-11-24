# Minimal ReAct Agent (LangGraph + Qwen2.5-Coder) for Buggy Python Fixing

The goal is to build an agent that automatically fixes buggy Python code and evaluates the correctness of the fix (locally running tests and reporting a metric over many tasks).

This repo provides a ReAct-style agent with a sandboxed Python tool. You can (a) run a minimal demo on a single task from `bigcode/humanevalpack`, and (b) evaluate pass@1 across many tasks with dataset-driven few‑shot (few‑shot items are excluded from scoring). The agent sends only the corrected function; we automatically add a small runner (with `if __name__ == '__main__':` and the tests) to execute it. See details about metrics, datasets, and approach below.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1.  **Install uv** (if not already installed):

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Install dependencies**:

    ```bash
    uv sync
    ```

    This will create a virtual environment and install all dependencies specified in `pyproject.toml` and `uv.lock`.

3.  **Optional environment variables**:

```bash
export MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"  # default 
export MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
```

For gated models, 
```bash
hf auth login
```

And get access [here](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main) (link for Llama-3).

### 🐙 Dataset
- We use the HumanEvalFix subset of HumanEvalPack ([HumanEvalPack](https://huggingface.co/datasets/bigcode/humanevalpack)).
- The dataset provides:
  - instruction: task description, including an example test inline.
  - example_test: the standalone example test code used to guide the agent.
  - test: hidden tests used for final verification and pass@1 scoring.
  - buggy_solution: the buggy Python implementation to be fixed.
  - canonical_solution: the correct reference implementation (used only for few‑shot construction; excluded from evaluation targets).
- In the demo, the agent receives the instruction and buggy solution; a dataset‑driven few‑shot prefix is included by default (number of shots k=3, excluding the current item); the initial harness contains only the example test (the main test is kept for evaluation).
- In evaluation, few‑shot examples are built from the dataset and excluded from scoring; the agent is prompted with the instruction + buggy solution + example_test, while the hidden test is used for scoring.

### Run the minimal demo

```bash
uv run examples/demo_minimal.py --index 0 --max-iterations 6
```

What this demo does:
- Loads one task from `bigcode/humanevalpack` (selectable via `--index`).
- Builds a few‑shot prefix (k=3 by default in the demo) and a harness with tests and an `if __name__ == '__main__':` runner.
- Starts a ReAct loop: the agent first calls the tool using the provided hint, then iterates LLM → tool until tests pass or the iteration limit is hit.
- On “Final Answer”, the agent verifies the function by running it with the harness; if it fails, the agent receives an Observation and continues instead of stopping.

Notes:
- Few-shot prompt is included by default in the demo and is dataset‑driven: we inject k=3 real examples from `bigcode/humanevalpack` (instruction (task description), buggy code, example_test, test, canonical_solution), excluding the current `--index`.
- The agent sends ONLY the corrected function; the harness (tests and `if __name__ == '__main__':`) is provided separately during execution.
- To disable few‑shot in the demo, remove the few‑shot construction in `examples/demo_minimal.py` (the `build_few_shot_prefix(...)` call) before the main task message.

### Evaluate pass@1 on 10 samples
The agent sees ONLY the example test; we score on the hidden test.

```bash
uv run examples/eval_pass1.py --num 10 --max-iterations 15
```

What this evaluation does:
- Loads the `test` split and builds a few‑shot prefix (size controlled by `--few-shot-k`).
- Excludes those few‑shot items from the evaluation pool to keep scoring fair.
- For each selected task, prompts the agent with the instruction + buggy code and ONLY the example test; the hidden `test` is used for scoring.
- Runs the same ReAct loop with the tool and performs final verification before accepting a “Final Answer”.
- Extracts the corrected function and executes the hidden tests to compute pass@1; prints per‑item results and a final summary JSON line.

This prints per-item results and a final JSON line with `pass@1`.

Few-shot control (k = number of few‑shot examples; defaults to 3 and excluded from scoring):
```bash
# disable few-shot
uv run examples/eval_pass1.py --num 10 --max-iterations 12 --few-shot-k 0
# use N few-shot examples
uv run examples/eval_pass1.py --num 10 --max-iterations 12 --few-shot-k 3
```

### Metric: pass@1
- Definition: share of tasks for which the first (and only) final solution produced by the agent passes the hidden tests.
- Prompting setup: the agent receives `instruction`, `buggy_solution`, and ONLY `example_test`. The hidden `test` is withheld for scoring. Few‑shot examples (k) may be provided for in‑context learning but are excluded from the evaluation pool.
- Procedure:
  1) Run the ReAct agent on each selected item (bounded by `--max-iterations`).
  2) Extract the corrected function from the agent’s “Final Answer” code fence; if missing, fall back to the last `Action Input.code` and strip any harness.
  3) Assemble a minimal evaluation script with the corrected function + hidden `test`.
  4) Execute in the sandbox (timeout enforced).
  5) Mark the item as passed iff the process succeeds (`ok=True` and `exit_code==0`).
- Output: per‑item JSON lines `{"index": <ds_idx>, "pass": true|false}` and a final summary `{"metric":"pass@1","num":N,"successes":S,"value":S/N}`.
- Reproducibility: set `MODEL_NAME` and keep `--few-shot-k` constant. Sampling is enabled with temperature≈0.2 by default; to reduce variance, you can set temperature to 0 (greedy) in `agent/llm.py`.

### Approach (how it works)

- ReAct loop (LangGraph):
  - The agent runs a ReAct cycle with two nodes: LLM (reasoning/decision) and tool (execution).
  - A small “start” router checks if a helper hint is present (“Action Input: …”) and, if so, injects an assistant Action and immediately routes to the tool on the first step to avoid long initial “thinking”.
  - Routing logic: if the last assistant message contains Final Answer → we first try to verify it by running the code with the harness; if verification passes, stop; otherwise feed the Observation back and continue. If it contains a valid Action for a known tool → run the tool; otherwise, go back to LLM.

- Strict output format:
  - While acting, the assistant must answer with two lines only:
    - Action: code_interpreter
    - Action Input: { "code": "\<ONLY_FUNCTION\>", "harness": "\<KEPT_BY_US\>" }
  - At the end, the assistant must produce:
    - Final Answer: \<only the corrected function\>
  - We add validation: if Action Input is missing/invalid, we do not run the tool but return an Observation telling the model to resend a valid JSON payload.

- Tooling (sandboxed Python):
  - `code_interpreter` executes Python in a subprocess with `-I -S` and resource limits. Args: `code` (str), `harness` (optional str), `timeout` (optional int).
  - Input normalization: we dedent and strip inputs and compose the final script as “function + harness” when provided.
  - Early checks: empty code and SyntaxError are detected up front and returned as Observations with actionable hints.
  - Auto entry point: if the final script does not contain a top‑level `if __name__ == '__main__':`, we automatically inject a minimal runner.
    - If `harness` is provided, it is wrapped under the injected `__main__` guard.
    - If `harness` is absent, we inject a no‑op main so the script can still execute deterministically.
  - The harness contains tests/asserts; an explicit entry point in the harness is optional (auto‑added if missing). The model sends only the function.

- Final verification before stopping:
  - When the model outputs “Final Answer”, we extract the function from the code fence and run it with the same harness that was provided in the hint.
  - If the script fails (non‑zero exit or assertion), we append an Observation and continue the loop instead of stopping.

- Few-shot prompting:
  - Demo: dataset-driven few-shot (k defaults to 3 in the demo); examples include instruction, buggy code, example_test, test, canonical_solution; the current example index is excluded.
  - Eval: dataset-driven few-shot is also supported via `--few-shot-k`; few-shot items are excluded from scoring.

- Generation controls:
  - Dynamic budget: when we expect “Action Input” with code, we temporarily increase generation limits for that step.
  - Anti-repeat: repetition penalty and no-repeat n-gram encourage concise, structured outputs.
  - Min-length: we use `min_new_tokens` to reduce extremely short/early-stopped “thought” responses.

- Evaluation (pass@1):
  - The agent only sees example_test (and may invent extra asserts to probe edge cases).
  - After the agent finishes, we extract the corrected function and run the hidden test `test` in isolation to score pass@1.

### Auto-injected __main__ and code post‑processing

- We provide small helpers in `utils/code_postprocess.py`:
  - `has_dunder_main(code)`: detect a top‑level `if __name__ == "__main__":`.
  - `inject_main_for_testing(code, main_body)`: add a minimal `__main__` block if missing (used by the tool to ensure execution).
  - `strip_injected_main(code)`: remove our injected block (marker‑based) to keep final answers clean.
- Behavior:
  - During tool execution, the runner is injected automatically when missing.
  - Final answers returned by the agent contain only the corrected function (no `__main__` block). You remain free to run your own evaluation harness to collect metrics.

### Files
- `agent/llm.py` – LLMChat wrapper (generic HF Transformers loader with optional HF token)
- `agent/tools.py` – sandboxed `code_interpreter` tool and registry
- `agent/react_agent.py` – ReAct agent graph (LangGraph), parsing, routing
- `examples/demo_minimal.py` – dataset loader + demo runner
- `examples/eval_pass1.py` – pass@1 evaluator (uses example test for prompting, hidden test for scoring)
- `minimal_agent.py` – thin wrapper that forwards to `examples/demo_minimal.py`
- `utils/code_postprocess.py` – helpers for injecting/stripping `__main__` in generated scripts
- `pyproject.toml`, `uv.lock` – dependencies and lockfile (managed via uv)

### Notes
- The sandbox is best-effort (`python -I -S` and a temp directory). Do not run untrusted code.

### Hyperparameters
- max_new_tokens: chosen to let the model finish full outputs. We use a dynamic budget: ~1024 tokens for short “thinking” steps and up to ~2048 tokens when an Action Input with code is expected.
- max_time: keep moderate to avoid long stalls; when running on CPU, we required max_time = 1200 min for code generation, otherwise the code wasn't generated fully. If you see truncated replies, raise the `max_time` values inside `agent/llm.py` (the dynamic `gen_kwargs` block in `LLMChat.invoke`).
- min_new_tokens: we set a lower bound (e.g., 64 for short steps, 128 for code steps) to reduce ultra‑short, early‑stopped replies.
- Other controls: temperature≈0.2 with sampling enabled, repetition penalty, and no‑repeat n‑gram to encourage the strict format.

## Current results

In the current form, the agent works well as an assistant to locate the root cause and suggest a fix, but it does not reliably complete the full “generate → test → confirm” loop autonomously. The agent usually identifies the bug correctly and proposes a reasonable fix. Although during run it often fails on formatting issues and new fixes introduce new errors. Using a more advanced model should make the performance better.

