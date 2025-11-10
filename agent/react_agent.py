from __future__ import annotations

import json
import re
import sys
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, END

from .llm import LLMChat, ChatMessage
from .tools import TOOLS


def react_system_prompt() -> str:
    tool_desc = "- code_interpreter: Execute Python. schema: {\"code\": \"str\", \"timeout\": \"int (optional)\"}"
    return (
        "You are an assistant that fixes Python code using ReAct. You can run code via tools.\n"
        "Follow this exact format and rules:\n\n"
        "Thought: <reason>\n"
        "Action: <tool name>\n"
        "Action Input: <JSON object>\n\n"
        "Rules:\n"
        "- On every step (unless you are finishing), you MUST output Action and Action Input. Never output only Thought.\n"
        "- Always validate by calling code_interpreter with a script that contains your current corrected solution and tests.\n"
        "- Use ONLY tests provided for the current task; do not copy any tests from earlier examples.\n\n"
        "When finished:\n"
        "Final Answer: <short summary>\n\n"
        "Available tools:\n"
        f"{tool_desc}\n"
    )


RE_ACT = re.compile(
    r"(?:Thought:\s*(?P<thought>.+?))?(?:\s*Action:\s*(?P<action>[a-zA-Z0-9_]+))?(?:\s*Action Input:\s*(?P<input>\{[\s\S]*\}))?(?:\s*Final Answer:\s*(?P<final>[\s\S]+))?$",
    re.DOTALL,
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
    no_action_streak: int


def build_minimal_agent(model: Optional[LLMChat] = None, max_iterations: int = 8):
    llm = model or LLMChat()
    system_message: ChatMessage = {"role": "system", "content": react_system_prompt()}

    def node_llm(state: AgentState) -> AgentState:
        messages = state.get("messages", [])
        if not messages or messages[0]["role"] != "system":
            messages = [system_message] + messages
        current_iter = state.get("iterations", 0) + 1
        print(f"[agent] Step {current_iter}: LLM thinking...", file=sys.stdout, flush=True)
        output = llm.invoke(messages)
        messages.append({"role": "assistant", "content": output})
        state["messages"] = messages
        state["iterations"] = state.get("iterations", 0) + 1
        return state

    def node_tool(state: AgentState) -> AgentState:
        last = state.get("messages", [])[-1]["content"]
        parsed = parse_react(last)
        name = parsed.get("action")
        args = parsed.get("action_input") or {}
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
            print("[agent] Final Answer detected. Stopping.", file=sys.stdout, flush=True)
            return END
        if state.get("iterations", 0) >= state.get("max_iterations", max_iterations):
            msgs = state.get("messages", [])
            msgs.append({"role": "assistant", "content": "Final Answer: Stopping due to iteration limit."})
            state["messages"] = msgs
            print("[agent] Reached iteration limit. Stopping.", file=sys.stdout, flush=True)
            return END
        action = parsed.get("action")
        if action and action in TOOLS:
            state["no_action_streak"] = 0
            return "tool"
        # No action: nudge the model once or twice
        streak = state.get("no_action_streak", 0) + 1
        state["no_action_streak"] = streak
        if streak <= 2:
            messages = state.get("messages", [])
            messages.append({
                "role": "user",
                "content": (
                    "Reminder: Output an Action and Action Input. "
                    "Use Action: code_interpreter with JSON containing a 'code' field that includes your current corrected solution "
                    "and ONLY the tests provided for THIS task."
                ),
            })
            state["messages"] = messages
        return "llm"

    graph = StateGraph(AgentState)
    graph.add_node("llm", node_llm)
    graph.add_node("tool", node_tool)
    graph.set_entry_point("llm")
    graph.add_conditional_edges("llm", route, {"tool": "tool", "llm": "llm", END: END})
    graph.add_conditional_edges("tool", route, {"tool": "tool", "llm": "llm", END: END})
    return graph.compile()


