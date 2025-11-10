## Minimal ReAct Agent (LangGraph + Qwen3-0.6B) for Buggy Python Fixing

This is a very small demo showing how to build a ReAct-style agent with LangGraph and a single tool to run Python code in a minimal sandbox. It loads one example from `bigcode/humanevalpack` and lets the agent iteratively run tests and propose fixes.

### Install

```bash
pip install -r requirements.txt
```

For LLama-3, 
```bash
hf auth login
```

And get access here: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct/tree/main

Optional environment variables:

```bash
export MODEL_NAME="Qwen/Qwen2.5-Coder-7B-Instruct"
```

### Run the minimal demo

```bash
python examples/demo_minimal.py --index 0 --max-iterations 6
```

### Evaluate pass@1 on 10 samples
The agent sees ONLY the example test; we score on the hidden test.

```bash
python examples/eval_pass1.py --num 10 --max-iterations 8
```

This prints per-item results and a final JSON line with `pass@1`.

### Files
- `agent/llm.py` – LLMChat wrapper (generic HF Transformers loader with optional HF token)
- `agent/tools.py` – sandboxed `code_interpreter` tool and registry
- `agent/react_agent.py` – ReAct agent graph (LangGraph), parsing, routing
- `examples/demo_minimal.py` – dataset loader + demo runner
- `examples/eval_pass1.py` – pass@1 evaluator (uses example test for prompting, hidden test for scoring)
- `minimal_agent.py` – thin wrapper that forwards to `examples/demo_minimal.py`
- `requirements.txt` – minimal dependencies.

### Notes
- The sandbox is best-effort (`python -I -S` and a temp directory). Do not run untrusted code.
- Qwen3-0.6B can run on CPU; GPU (CUDA) speeds things up if available.


