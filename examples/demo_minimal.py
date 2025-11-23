from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from datasets import load_dataset

# Ensure project root is on sys.path when running this file directly
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.llm import ChatMessage
from agent.react_agent import build_minimal_agent


def assemble_test_script(buggy_solution: str, example_test: str, test: str) -> str:
    return (
        f"{buggy_solution}\n\n"
        "if __name__ == '__main__':\n"
        "    try:\n"
        "        " + example_test.replace("\n", "\n        ") + "\n"
        "        " + test.replace("\n", "\n        ") + "\n"
        "        print('ALL_TESTS_PASSED')\n"
        "    except AssertionError as e:\n"
        "        print('ASSERTION_FAILED:', e)\n"
        "        raise\n"
    )


def build_few_shot_prefix(ds, exclude_index: int, k: int = 3) -> str:
    parts: List[str] = []
    taken = 0
    # ensure we have needed columns
    columns = set(ds.column_names)
    need = {"buggy_solution", "example_test", "test", "instruction", "canonical_solution"}
    if not need.issubset(columns):
        ds = ds.select_columns(list(need.intersection(columns)))
    for idx in range(len(ds)):
        if idx == exclude_index:
            continue
        row = ds[idx]
        correct = row.get("canonical_solution")
        buggy = row.get("buggy_solution")
        ex_test = row.get("example_test")
        main_test = row.get("test")
        instr = row.get("instruction")
        if not (isinstance(correct, str) and correct.strip() and isinstance(buggy, str) and isinstance(ex_test, str) and isinstance(main_test, str) and isinstance(instr, str)):
            continue
        parts.append(
            "Instruction:\n"
            f"{instr}\n\n"
            "Buggy code:\n"
            "```python\n"
            f"{buggy}\n"
            "```\n"
            "Example test:\n"
            "```python\n"
            f"{ex_test}\n"
            "```\n"
            "Main test:\n"
            "```python\n"
            f"{main_test}\n"
            "```\n"
            "Correct code:\n"
            "```python\n"
            f"{correct}\n"
            "```\n"
        )
        taken += 1
        if taken >= k:
            break
    if not parts:
        return ""
    return "Below are few-shot examples from the dataset:\n\n" + "\n\n".join(parts)


def run_demo(index: int, max_iterations: int) -> List[ChatMessage]:
    print("[demo] Loading dataset...", flush=True)
    dataset = load_dataset("bigcode/humanevalpack", split="test")
    subset = dataset.select_columns(["buggy_solution", "test", "example_test", "instruction"])
    row = subset[index]

    code = row["buggy_solution"]
    example_test = row["example_test"]
    test = row["test"]
    instruction = row["instruction"]

    print(f"[demo] Example #{index} loaded.", flush=True)
    print("[demo] Building agent...", flush=True)
    app = build_minimal_agent(max_iterations=max_iterations)

    user_task = (
        "You will fix a buggy Python solution.\n"
        "Here is the instruction describing the task:\n"
        f"{instruction}\n\n"
        "Here is the current buggy solution (as Python code):\n"
        "```python\n"
        f"{code}\n"
        "```\n\n"
        "We can run tests in a sandbox via the tool `code_interpreter`.\n"
        "To test the code, compose a Python script that includes the solution and then runs these tests.\n"
        "You MAY also invent additional tests (asserts) to probe edge cases and help debugging. Keep provided tests intact.\n"
        "Example test:\n"
        "```python\n"
        f"{example_test}\n"
        "```\n"
        "Main test:\n"
        "```python\n"
        f"{test}\n"
        "```\n\n"
        "Use ReAct: think, call code_interpreter with the script, inspect failures, then propose a minimal fix. "
        "When you change the code, re-run the tests (including your extra tests if any). Finish with 'Final Answer' summarizing the fix."
    )

    # Provide a skeleton (harness) with example_test only to reduce length. Ask model to send only the function in code.
    indented_example = example_test.replace("\n", "\n        ")
    skeleton = (
        "if __name__ == '__main__':\n"
        "    try:\n"
        "        # Provided tests (example only on first run)\n"
        f"        {indented_example}\n"
        "        # Your additional tests (optional):\n"
        "        # e.g., assert <func>(...) == ...\n"
        "        print('ALL_TESTS_PASSED')\n"
        "    except AssertionError as e:\n"
        "        print('ASSERTION_FAILED:', e)\n"
        "        raise\n"
    )

    # Build dataset-driven few-shot prefix excluding current item
    try:
        ds_full = load_dataset("bigcode/humanevalpack", split="test")
        fewshot = build_few_shot_prefix(ds_full, exclude_index=index, k=3)
    except Exception:
        fewshot = ""
    inputs = {
        "messages": [
            *(([{"role": "user", "content": fewshot}] ) if fewshot else []),
            {"role": "user", "content": user_task},
            {"role": "user", "content": "Helper: Send only the corrected function in 'code'. The harness will be provided separately."},
            {"role": "user", "content": f"Action: code_interpreter\nAction Input: {json.dumps({'code': '<PUT_ONLY_FUNCTION_HERE>', 'harness': skeleton})}"},
        ],
        "max_iterations": max_iterations,
    }
    print("[demo] Invoking agent (this may take a while on first run)...", flush=True)
    final_state = app.invoke(inputs)
    print("[demo] Agent finished.", flush=True)
    return final_state.get("messages", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal LangGraph ReAct agent demo using Qwen")
    parser.add_argument("--index", type=int, default=0, help="Example index from humanevalpack test split")
    parser.add_argument("--max-iterations", type=int, default=6, help="Max ReAct steps")
    args = parser.parse_args()

    messages = run_demo(args.index, args.max_iterations)
    print(json.dumps(messages[-6:], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


