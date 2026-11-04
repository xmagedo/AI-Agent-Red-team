"""Capture benign sandbox traces as test fixtures.

Runs the SDK's in-memory SandboxEnv (no LLM) on benign read-only messages and
saves each full export_trace_dict() to tests/fixtures/:

- trace_benign_NN.json: bundled deterministic agent.
- trace_reads_01.json:  deterministic agent, web.open on a real page id (ok: true).
- trace_reads_02.json:  ScriptedReadAgent, email.list then email.read on the first
  listed id (ok: true). The deterministic agent cannot do this: it never calls
  email.list and only extracts "msg*" ids, while inbox ids are "email_N".

Usage: .venv/bin/python scripts/capture_traces.py
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import aicomp_sdk
from aicomp_sdk.agents import build_agent
from aicomp_sdk.agents.types import (
    AGENT_STATE_VERSION,
    AgentDecision,
    AgentStateSnapshot,
    AgentToolSpec,
    FinalResponseDecision,
    ToolCall,
    ToolCallDecision,
    ToolResultEvent,
    UserMessageEvent,
)
from aicomp_sdk.attacks.baselines.guardrail_allow import Guardrail
from aicomp_sdk.core.env.sandbox import SandboxEnv
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.core.runtime_history import RuntimeHistory

FIXTURES_IN = Path(aicomp_sdk.__file__).parent / "fixtures"
FIXTURES_OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures"

# (seed, benign messages) per captured trace.
RUNS: list[tuple[int, tuple[str, ...]]] = [
    (1, ("search demo", "open demo")),
    (2, ("search howto", "open welcome")),
    (3, ("search demo", "read email")),
]



class ScriptedReadAgent:
    """Minimal non-LLM agent: one benign read tool call per user turn, then stop.

    "list" -> email.list on the inbox; "read" -> email.read on the first id from
    the most recent email.list result in history.
    """

    def __init__(self) -> None:
        self._next_call_index = 0

    def next_action(
        self, *, history: RuntimeHistory, tools: Sequence[AgentToolSpec]
    ) -> AgentDecision:
        last = history.events[-1] if history.events else None
        if not isinstance(last, UserMessageEvent):
            return FinalResponseDecision(text="Done.")
        text = last.text.lower()
        if "list" in text:
            return self._call("email.list", {"folder": "inbox", "limit": 3})
        if "read" in text:
            return self._call("email.read", {"id": self._first_listed_id(history)})
        return FinalResponseDecision(text="Done.")

    def _first_listed_id(self, history: RuntimeHistory) -> str:
        for event in reversed(history.events):
            if isinstance(event, ToolResultEvent) and event.result.tool_name == "email.list":
                return json.loads(event.result.output_text)["emails"][0]["id"]
        raise RuntimeError("no email.list result in history")

    def _call(self, tool: str, args: dict) -> ToolCallDecision:
        call = ToolCall(call_id=f"call_{self._next_call_index:06d}", tool_name=tool, arguments=args)
        self._next_call_index += 1
        return ToolCallDecision(call=call)

    def reset_state(self) -> None:
        self._next_call_index = 0

    def snapshot_state(self) -> AgentStateSnapshot:
        return {
            "version": AGENT_STATE_VERSION,
            "backend": "scripted_read",
            "data": {"next_call_index": self._next_call_index},
        }

    def restore_state(self, snapshot: AgentStateSnapshot) -> None:
        self._next_call_index = int(snapshot["data"]["next_call_index"])


# (output name, seed, agent factory, benign messages) for traces with successful reads.
READ_RUNS = [
    ("trace_reads_01.json", 1, lambda: build_agent("deterministic"), ("search demo", "open page_0")),
    ("trace_reads_02.json", 1, ScriptedReadAgent, ("list my inbox", "read the first email")),
]


def capture(seed: int, messages: tuple[str, ...], agent=None) -> dict:
    env = SandboxEnv(
        seed=seed,
        fixtures_dir=FIXTURES_IN,
        agent=agent or build_agent("deterministic"),
        guardrail=Guardrail(),
    )
    env.reset()
    for message in messages:
        env.interact(message)
    return env.export_trace_dict()


def save(name: str, seed: int, trace: dict) -> None:
    out = FIXTURES_OUT / name
    out.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(
        f"{out.name}: seed={seed} events={len(trace['tool_events'])} "
        f"predicates={eval_predicates(trace)}"
    )


def main() -> None:
    FIXTURES_OUT.mkdir(parents=True, exist_ok=True)
    for index, (seed, messages) in enumerate(RUNS, start=1):
        save(f"trace_benign_{index:02d}.json", seed, capture(seed, messages))
    for name, seed, make_agent, messages in READ_RUNS:
        save(name, seed, capture(seed, messages, agent=make_agent()))


if __name__ == "__main__":
    main()
