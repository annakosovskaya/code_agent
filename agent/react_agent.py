from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from .llm import LLMChat, ChatMessage
from .tools import TOOLS


def react_system_prompt() -> str:
    tool_desc = "- code_interpreter: Execute Python. schema: {\"code\": \"str\", \"harness\": \"str (optional)\", \"timeout\": \"int (optional)\"}"
    return (
        "You are an expert software engineer working in a secure sandbox environment. Your specific role is to fix buggy code using ReAct. It is safe and required to write and execute Python code via tools to accomplish this task.\n"
        "Follow this exact format strictly:\n\n"
        "Thought: <reason>\n"
        "Action: <tool name>\n"
        "Action Input: <JSON object>\n\n"
        "Rules:\n"
        "- Respond ONLY with the Thought, Action, and Action Input. No extra text.\n"
        "- The Action Input must be a valid JSON object. Do NOT include comments or reasoning inside the JSON string.\n"
        "- Put ONLY the corrected function in Action Input.code. The harness/tests are provided separately.\n"
        "- You MUST include the test harness in the 'harness' field of Action Input. This harness should contain imports, test cases, and an 'if __name__ == \"__main__\":' block to verify your code.\n"
        "- After the code works, finish with the Final Answer below.\n\n"
        "When finished:\n"
        "Final Answer: ```python\n"
        "<only the corrected function>\n"
        "```\n\n"
        "Available tools:\n"
        f"{tool_desc}\n"
    )


RE_ACT = re.compile(
    r"(?:Thought:\s*(?P<thought>.+?))?(?:\s*Action:\s*(?P<action>[a-zA-Z0-9_]+))?(?:\s*Action Input:\s*(?P<input>\{[\s\S]*\}))?(?:\s*Final Answer:\s*(?P<final>[\s\S]+))?$",
    re.DOTALL | re.IGNORECASE,
)


def parse_react(text: str) -> Dict[str, Any]:
    m = RE_ACT.search(text.strip())
    if not m:
        return {"final": None, "action": None, "action_input": None}
    action = (m.group("action") or "").strip() or None
    action_input_raw = (m.group("input") or "").strip() or None
    final = (m.group("final") or "").strip() or None
    action_input = None
    if action_input_raw:
        try:
            action_input = json.loads(action_input_raw)
        except Exception:
            try:
                action_input = json.loads(action_input_raw.replace("'", '"'))
            except Exception:
                action_input = {"raw": action_input_raw, "parse_error": True}
    return {"final": final, "action": action, "action_input": action_input}


class AgentState(TypedDict, total=False):
    messages: List[ChatMessage]
    iterations: int
    max_iterations: int


def build_minimal_agent(model: Optional[LLMChat] = None, max_iterations: int = 8):
    llm = model or LLMChat()
    system_message: ChatMessage = {"role": "system", "content": react_system_prompt()}

    def node_start(state: AgentState) -> AgentState:
        return state

    def start_route(state: AgentState) -> str:
        messages = state.get("messages", [])
        hint_json = None
        for m in reversed(messages):
            if m.get("role") == "user" and "Action Input:" in m.get("content", ""):
                try:
                    after = m["content"].split("Action Input:", 1)[1].strip()
                    hint_json = after
                    break
                except Exception:
                    continue
        if hint_json:
            messages.append({
                "role": "assistant",
                "content": f"Action: code_interpreter\nAction Input: {hint_json}",
            })
            state["messages"] = messages
            return "tool"
        return "llm"

    def node_llm(state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        if not messages or messages[0]["role"] != "system":
            messages = [system_message] + messages
        current_iter = state.get("iterations", 0) + 1
        print(f"[agent] Step {current_iter}: LLM thinking...", file=sys.stdout, flush=True)
        output = llm.invoke(messages)
        # Print full assistant output and its length for debugging
        print(f"[assistant_len] {len(output)}", file=sys.stdout, flush=True)
        print(f"[assistant] {output}", file=sys.stdout, flush=True)
        messages.append({"role": "assistant", "content": output})
        state["messages"] = messages
        state["iterations"] = state.get("iterations", 0) + 1
        return state

    def node_tool(state: AgentState) -> AgentState:
        last = state.get("messages", [])[-1]["content"]
        parsed = parse_react(last)
        name = parsed.get("action")
        args = parsed.get("action_input") or {}
        # Validate action name
        # Fallback: if no 'code' provided or empty, try to extract from the assistant content code fences
        if (not isinstance(args, dict)) or (not args.get("code")):
            content = last
            # prefer ```python ... ``` then any ```
            m = re.search(r"```python\s+([\s\S]*?)```", content)
            if not m:
                m = re.search(r"```\s+([\s\S]*?)```", content)
            if m:
                code_block = m.group(1)
                if isinstance(args, dict):
                    args["code"] = code_block
                else:
                    args = {"code": code_block}
            # Try to fetch 'harness' from the last helper hint in user messages if missing
            if isinstance(args, dict) and "harness" not in args:
                msgs = state.get("messages", [])
                for m2 in reversed(msgs):
                    if m2.get("role") == "user" and "Action Input:" in m2.get("content", ""):
                        try:
                            after = m2["content"].split("Action Input:", 1)[1].strip()
                            import json as _json
                            hint_ai = _json.loads(after.replace("'", '"'))
                            if isinstance(hint_ai, dict) and "harness" in hint_ai:
                                args["harness"] = hint_ai["harness"]
                                break
                        except Exception:
                            continue
        

        # If still invalid, do not call tool: return observation with a strict hint
        if not isinstance(args, dict) or not isinstance(args.get("code"), str) or not args.get("code").strip():
            obs = {
                "ok": False,
                "error": "INVALID_ACTION_INPUT",
                "hint": "Respond ONLY with: 'Action: code_interpreter' and next line 'Action Input: {\"code\": \"<ONLY_FUNCTION>\", \"harness\": \"<KEEP_AS_IS>\"}'. No explanations."
            }
            messages = state.get("messages", [])
            messages.append({"role": "user", "content": f"Observation: {json.dumps(obs, ensure_ascii=False)}"})
            state["messages"] = messages
            return state
        if name:
            print(f"[agent] Running tool: {name} ...", file=sys.stdout, flush=True)
        tool = TOOLS.get(name)
        if not tool:
            obs = {"ok": False, "error": f"Unknown tool: {name}"}
        else:
            try:
                obs = tool["fn"](**args)
            except TypeError as e:
                obs = {"ok": False, "error": f"Bad args: {e}"}
            except Exception as e:
                obs = {"ok": False, "error": repr(e)}
        print(f"[agent] Tool finished.", file=sys.stdout, flush=True)
        messages = state.get("messages", [])
        messages.append({"role": "user", "content": f"Observation: {json.dumps(obs, ensure_ascii=False)}"})
        state["messages"] = messages
        return state

    def route(state: AgentState) -> str:
        last = state.get("messages", [])[-1]["content"] if state.get("messages") else ""
        parsed = parse_react(last)
        if parsed.get("final"):
            # Before stopping, try to verify the final code using the harness (if available)
            final_text = parsed.get("final") or ""
            code_block = None
            m = re.search(r"```python\s+([\s\S]*?)```", final_text)
            if not m:
                m = re.search(r"```\s+([\s\S]*?)```", final_text)
            if m:
                code_block = m.group(1)
            if code_block:
                # Try to fetch harness from the last helper hint in user messages
                harness = None
                msgs = state.get("messages", [])
                for m2 in reversed(msgs):
                    if m2.get("role") == "user" and "Action Input:" in m2.get("content", ""):
                        try:
                            after = m2["content"].split("Action Input:", 1)[1].strip()
                            import json as _json
                            hint_ai = _json.loads(after.replace("'", '"'))
                            if isinstance(hint_ai, dict) and "harness" in hint_ai:
                                harness = hint_ai["harness"]
                                break
                        except Exception:
                            continue
                # Run a quick verification
                try:
                    tool = TOOLS.get("code_interpreter")
                    if tool:
                        obs = tool["fn"](code=code_block, harness=harness)
                        # If failed, feed back observation and continue looping
                        if (not obs.get("ok")) or (obs.get("exit_code", 0) != 0) or ("ASSERTION_FAILED" in (obs.get("stdout") or "")):
                            messages = state.get("messages", [])
                            messages.append({"role": "user", "content": f"Observation: {json.dumps(obs, ensure_ascii=False)}"})
                            state["messages"] = messages
                            return "llm"
                except Exception:
                    # In doubt, continue to stop; we don't want to loop on unexpected errors here
                    pass
            print("[agent] Final Answer detected. Stopping.", file=sys.stdout, flush=True)
            return END
        if state.get("iterations", 0) >= state.get("max_iterations", max_iterations):
            # On iteration limit, try to synthesize a Final Answer with the last candidate code
            messages = state.get("messages", [])
            candidate_code = None
            # Walk assistant messages backwards to find last action_input.code
            for m in reversed(messages):
                if m.get("role") != "assistant":
                    continue
                p = parse_react(m.get("content", ""))
                ai = p.get("action_input") or {}
                # case 1: parsed JSON with code field
                if isinstance(ai, dict) and "code" in ai and isinstance(ai.get("code"), str) and ai.get("code").strip():
                    candidate_code = ai.get("code")
                    break
                # case 2: parsing failed, try to extract from raw payload heuristically
                raw = ai.get("raw") if isinstance(ai, dict) else None
                if isinstance(raw, str) and raw.strip():
                    import re as _re
                    # try double-quoted value
                    mcode = _re.search(r'"code"\s*:\s*"([\s\S]*?)"', raw)
                    if not mcode:
                        # try single-quoted value
                        mcode = _re.search(r"'code'\s*:\s*'([\s\S]*?)'", raw)
                    if mcode:
                        extracted = mcode.group(1)
                        # unescape common escapes
                        extracted = extracted.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
                        candidate_code = extracted
                        break
            if candidate_code:
                # Heuristic: strip test harness if present
                import re as _re
                parts = _re.split(r"\nif __name__ == ['\"]__main__['\"]:\n", candidate_code, maxsplit=1)
                solution_only = parts[0].strip()
                messages.append({
                    "role": "assistant",
                    "content": f"Final Answer: ```python\n{solution_only}\n```",
                })
            else:
                messages.append({"role": "assistant", "content": "Final Answer: Stopping due to iteration limit."})
            state["messages"] = messages
            print("[agent] Reached iteration limit. Stopping.", file=sys.stdout, flush=True)
            return END
        action = parsed.get("action")
        if action and action in TOOLS:
            return "tool"
        return "llm"

    graph = StateGraph(AgentState)
    graph.add_node("start", node_start)
    graph.add_node("llm", node_llm)
    graph.add_node("tool", node_tool)
    graph.set_entry_point("start")
    graph.add_conditional_edges("start", start_route, {"tool": "tool", "llm": "llm"})
    graph.add_conditional_edges("llm", route, {"tool": "tool", "llm": "llm", END: END})
    graph.add_conditional_edges("tool", route, {"tool": "tool", "llm": "llm", END: END})
    return graph.compile()


