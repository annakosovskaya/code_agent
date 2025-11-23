from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import List, Optional, Tuple

from datasets import load_dataset

# Ensure project root is on sys.path when running this file directly
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agent.react_agent import build_minimal_agent, parse_react
from agent.llm import ChatMessage, LLMChat
from agent.tools import run_python_sandboxed


def build_eval_prompt(few_shot_prefix: str, instruction: str, buggy_solution: str, example_test: str) -> str:
    header = (
        "You will fix a buggy Python solution.\n"
        "You are given ONLY an example test to guide your fix. Do NOT assume hidden tests.\n\n"
    )
    few_shot_block = (few_shot_prefix + "\n\n") if few_shot_prefix else ""
    task_block = (
        "Instruction (task description):\n"
        f"{instruction}\n\n"
        "Buggy solution (Python code):\n"
        "```python\n"
        f"{buggy_solution}\n"
        "```\n\n"
        "Example test (you may use this to validate your fix):\n"
        "```python\n"
        f"{example_test}\n"
        "```\n\n"
        "Instructions:\n"
        "- STRICT: On every step, unless finishing, output Action and Action Input. Never only Thought.\n"
        "- Use the tool `code_interpreter` to run a script that contains your CURRENT corrected solution + ONLY the example test.\n"
        "- You MAY invent additional tests (asserts) to probe edge cases; include them alongside the example test to validate robustness.\n"
        "- Do NOT copy tests from few-shot examples.\n"
        "- Iterate until the example test passes.\n"
        "- When you are FINISHED, output:\n"
        "  Final Answer: and then a single Python code block that contains ONLY the corrected solution (no tests, no prints).\n"
    )
    return header + few_shot_block + task_block


def extract_final_solution_from_messages(messages: List[ChatMessage]) -> Optional[str]:
    # 1) Prefer explicit code block in the latest assistant message
    for msg in reversed(messages):
        if msg["role"] != "assistant":
            continue
        content = msg["content"]
        # match triple backtick code block, prefer ```python ... ```
        m = re.search(r"```python\s+([\s\S]*?)```", content)
        if m:
            return m.group(1).strip()
        m = re.search(r"```\s+([\s\S]*?)```", content)
        if m:
            return m.group(1).strip()
        # also check if Final Answer provides raw code directly (rare)
    # 2) Fallback: take last Action Input.code that was sent to the interpreter and strip tests
    for msg in reversed(messages):
        if msg["role"] != "assistant":
            continue
        parsed = parse_react(msg["content"])
        if parsed.get("action") and parsed.get("action_input"):
            code = parsed["action_input"].get("code")
            if isinstance(code, str) and code.strip():
                # Heuristic: cut off any test harness starting with if __name__ guard
                parts = re.split(r"\nif __name__ == ['\"]__main__['\"]:\n", code, maxsplit=1)
                return parts[0].strip()
    return None


def assemble_hidden_eval_script(solution_code: str, hidden_test: str) -> str:
    return (
        f"{solution_code}\n\n"
        "if __name__ == '__main__':\n"
        "    try:\n"
        "        " + hidden_test.replace("\n", "\n        ") + "\n"
        "        print('ALL_TESTS_PASSED')\n"
        "    except AssertionError as e:\n"
        "        print('ASSERTION_FAILED:', e)\n"
        "        raise\n"
    )


def evaluate_single(dataset, index: int, few_shot_prefix: str, model: LLMChat, max_iterations: int = 8) -> Tuple[bool, List[ChatMessage], str]:
    subset = dataset.select_columns(["buggy_solution", "test", "example_test", "instruction"])
    row = subset[index]

    buggy_solution = row["buggy_solution"]
    hidden_test = row["test"]
    example_test = row["example_test"]
    instruction = row["instruction"]

    app = build_minimal_agent(model=model, max_iterations=max_iterations)

    user_task = build_eval_prompt(few_shot_prefix, instruction, buggy_solution, example_test)
    # Skeleton to encourage the agent to place optional extra tests
    indented_example = example_test.replace("\n", "\n    ")
    hint_script = (
        f"{buggy_solution}\n\n"
        "if __name__ == '__main__':\n"
        "    # Example test (provided)\n"
        f"    {indented_example}\n"
        "    # Your additional tests (optional):\n"
        "    # e.g., assert <func>(...) == ...\n"
    )

    inputs = {
        "messages": [
            {"role": "user", "content": user_task},
            {"role": "user", "content": "Helper: To validate, call code_interpreter with this JSON:"},
            {"role": "user", "content": f"Action: code_interpreter\nAction Input: {json.dumps({'code': hint_script})}"},
        ],
        "max_iterations": max_iterations,
    }
    print(f"[eval] Running agent on item {index} ...", flush=True)
    final_state = app.invoke(inputs)
    print(f"[eval] Agent finished for item {index}.", flush=True)
    messages: List[ChatMessage] = final_state.get("messages", [])

    candidate = extract_final_solution_from_messages(messages)
    if not candidate:
        return False, messages, "no_candidate_solution"

    eval_script = assemble_hidden_eval_script(candidate, hidden_test)
    obs = run_python_sandboxed(eval_script, timeout=12)
    passed = bool(obs.get("ok")) and int(obs.get("exit_code", 1)) == 0
    return passed, messages, obs.get("stdout", "") + obs.get("stderr", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute pass@1 on N examples (hidden tests held-out from prompt)")
    parser.add_argument("--num", type=int, default=10, help="Number of items to evaluate from test split")
    parser.add_argument("--max-iterations", type=int, default=12, help="Max ReAct steps per item")
    parser.add_argument("--few-shot-k", type=int, default=3, help="How many few-shot examples to include from dataset")
    args = parser.parse_args()

    # Load once and prepare few-shot examples with correct code if available
    full = load_dataset("bigcode/humanevalpack", split="test")
    few_shot_indices: List[int] = []
    parts: List[str] = []
    for idx in range(len(full)):
        row = full[idx]
        correct = row.get("canonical_solution") or ""
        if not correct:
            continue
        buggy = row.get("buggy_solution") or ""
        doc = row.get("instruction") or ""
        ex_test = row.get("example_test") or ""
        main_test = row.get("test") or ""
        part = (
            f"Instruction:\n{doc}\n\n"
            "Buggy code:\n```python\n" + buggy + "\n```\n"
            "Example test:\n```python\n" + ex_test + "\n```\n"
            "Main test:\n```python\n" + main_test + "\n```\n"
            "Correct code:\n```python\n" + correct + "\n```\n"
        )
        parts.append(part)
        few_shot_indices.append(idx)
        if len(few_shot_indices) >= args.few_shot_k:
            break
    few_shot_prefix = "Here are few-shot examples:\n\n" + "\n\n".join(parts) if parts else ""

    # Exclude few-shot indices from evaluation
    exclude = set(few_shot_indices)
    eval_indices = [i for i in range(len(full)) if i not in exclude][: args.num]

    total = len(eval_indices)
    # Build a single LLM instance and reuse across items to avoid reloading the model
    shared_llm = LLMChat()
    successes = 0
    for j, ds_idx in enumerate(eval_indices, start=1):
        print(f"[eval] ===== Item {j}/{total} (dataset idx {ds_idx}) =====", flush=True)
        ok, _, _ = evaluate_single(full, ds_idx, few_shot_prefix=few_shot_prefix, model=shared_llm, max_iterations=args.max_iterations)
        successes += 1 if ok else 0
        print(json.dumps({"index": ds_idx, "pass": ok}, ensure_ascii=False))
    pass_at_1 = successes / total if total > 0 else 0.0
    print(json.dumps({"metric": "pass@1", "num": total, "successes": successes, "value": pass_at_1}, ensure_ascii=False))


if __name__ == "__main__":
    main()


