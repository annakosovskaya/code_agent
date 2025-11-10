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


def build_few_shot_prefix() -> str:
    return (
        "Below are three few-shot examples of fixing buggy Python code using tests.\n\n"
        "Example 1\n"
        "Docstring: Return the sum of two integers.\n"
        "Buggy code:\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a - b  # bug: subtraction instead of addition\n"
        "```\n"
        "Example test:\n"
        "```python\n"
        "assert add(2, 3) == 5\n"
        "```\n"
        "Main test:\n"
        "```python\n"
        "assert add(-1, 1) == 0\n"
        "assert add(10, 5) == 15\n"
        "```\n"
        "Correct code:\n"
        "```python\n"
        "def add(a, b):\n"
        "    return a + b\n"
        "```\n\n"
        "Example 2\n"
        "Docstring: Reverse a string.\n"
        "Buggy code:\n"
        "```python\n"
        "def reverse_string(s: str) -> str:\n"
        "    return ''.join(sorted(s))  # bug: sorts instead of reversing\n"
        "```\n"
        "Example test:\n"
        "```python\n"
        "assert reverse_string('abc') == 'cba'\n"
        "```\n"
        "Main test:\n"
        "```python\n"
        "assert reverse_string('racecar') == 'racecar'\n"
        "assert reverse_string('abcd') == 'dcba'\n"
        "```\n"
        "Correct code:\n"
        "```python\n"
        "def reverse_string(s: str) -> str:\n"
        "    return s[::-1]\n"
        "```\n\n"
        "Example 3\n"
        "Docstring: Return True if n is a prime number, else False.\n"
        "Buggy code:\n"
        "```python\n"
        "def is_prime(n: int) -> bool:\n"
        "    if n < 2:\n"
        "        return True  # bug: 0 and 1 are not prime\n"
        "    for i in range(2, n):\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "    return True\n"
        "```\n"
        "Example test:\n"
        "```python\n"
        "assert is_prime(7) is True\n"
        "```\n"
        "Main test:\n"
        "```python\n"
        "assert is_prime(1) is False\n"
        "assert is_prime(2) is True\n"
        "assert is_prime(9) is False\n"
        "```\n"
        "Correct code:\n"
        "```python\n"
        "def is_prime(n: int) -> bool:\n"
        "    if n < 2:\n"
        "        return False\n"
        "    i = 2\n"
        "    while i * i <= n:\n"
        "        if n % i == 0:\n"
        "            return False\n"
        "        i += 1\n"
        "    return True\n"
        "```\n"
    )


def run_demo(index: int, max_iterations: int) -> List[ChatMessage]:
    print("[demo] Loading dataset...", flush=True)
    dataset = load_dataset("bigcode/humanevalpack", split="test")
    subset = dataset.select_columns(["buggy_solution", "test", "example_test", "docstring"])
    row = subset[index]

    code = row["buggy_solution"]
    example_test = row["example_test"]
    test = row["test"]
    docstring = row["docstring"]

    print(f"[demo] Example #{index} loaded.", flush=True)
    print("[demo] Building agent...", flush=True)
    app = build_minimal_agent(max_iterations=max_iterations)

    user_task = (
        "You will fix a buggy Python solution.\n"
        "Here is the docstring describing the task:\n"
        f"{docstring}\n\n"
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

    # Provide a skeleton with a placeholder for extra tests
    skeleton = (
        f"{code}\n\n"
        "if __name__ == '__main__':\n"
        "    try:\n"
        "        # Provided tests\n"
        f"        {example_test.replace('\\n', '\\n        ')}\n"
        f"        {test.replace('\\n', '\\n        ')}\n"
        "        # Your additional tests (optional):\n"
        "        # e.g., assert <func>(...) == ...\n"
        "        print('ALL_TESTS_PASSED')\n"
        "    except AssertionError as e:\n"
        "        print('ASSERTION_FAILED:', e)\n"
        "        raise\n"
    )

    inputs = {
        "messages": [
            {"role": "user", "content": build_few_shot_prefix()},
            {"role": "user", "content": user_task},
            {"role": "user", "content": "Helper: To run tests, call code_interpreter with this JSON:"},
            {"role": "user", "content": f"Action: code_interpreter\nAction Input: {json.dumps({'code': skeleton})}"},
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


