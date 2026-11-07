"""Model-free fake environment mimicking the SDK's AttackEnvProtocol.

Serves pre-recorded trace dicts instead of running an agent, so the search engine
can be developed and tested without any model.
"""

from __future__ import annotations

import copy
from typing import Any

from aicomp_sdk.core.env.api import EnvInteractionResult


class _StubSnapshot:
    """Opaque snapshot token produced by StubEnv.snapshot()."""

    __slots__ = ("state",)

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state


class StubEnv:
    """Fake env serving scripted (by exact message) or pooled (round-robin) traces.

    Exactly one of ``scripted`` or ``pool`` must be given. Each interact() appends
    the served trace's ``tool_events`` to the accumulated trace.
    """

    def __init__(
        self,
        scripted: dict[str, dict] | None = None,
        pool: list[dict] | None = None,
        seed: int = 0,
    ) -> None:
        if (scripted is None) == (pool is None):
            raise ValueError("provide exactly one of scripted or pool")
        if pool is not None and not pool:
            raise ValueError("pool must not be empty")
        self._scripted = copy.deepcopy(scripted)
        self._pool = copy.deepcopy(pool)
        self._seed = int(seed)
        self._events: list[dict] = []
        self._user_messages: list[str] = []
        self._pool_index = 0

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int) -> None:
        self._seed = int(value)

    def _next_trace(self, user_message: str) -> dict:
        if self._scripted is not None:
            return self._scripted.get(user_message, {"tool_events": []})
        trace = self._pool[self._pool_index % len(self._pool)]
        self._pool_index += 1
        return trace

    def interact(self, user_message: str, max_tool_hops: int | None = None) -> EnvInteractionResult:
        """Record the message, serve the matching trace's events, return counters.

        agent_turns is reported as one turn per served tool event.
        """
        self._user_messages.append(user_message)
        served = copy.deepcopy(self._next_trace(user_message).get("tool_events", []))
        self._events.extend(served)
        succeeded = sum(1 for e in served if e.get("ok") is True)
        return EnvInteractionResult(
            seed=self._seed,
            total_tool_events=len(self._events),
            tool_events_added=len(served),
            agent_turns=len(served),
            successful_tool_calls=succeeded,
            failed_tool_calls=len(served) - succeeded,
            agent_refused=False,
        )

    def export_trace_dict(self) -> dict[str, Any]:
        """Return the accumulated trace in the same shape as the SDK trace dict."""
        return {
            "seed": self._seed,
            "user_messages": list(self._user_messages),
            "assistant_messages": [],
            "tool_events": copy.deepcopy(self._events),
            "egress": [],
        }

    def reset(self, *args: Any, **kwargs: Any) -> None:
        self._events = []
        self._user_messages = []
        self._pool_index = 0

    def snapshot(self) -> _StubSnapshot:
        return _StubSnapshot(
            copy.deepcopy(
                {
                    "events": self._events,
                    "user_messages": self._user_messages,
                    "pool_index": self._pool_index,
                }
            )
        )

    def restore(self, token: Any) -> None:
        if not isinstance(token, _StubSnapshot):
            raise TypeError("StubEnv.restore requires a token from StubEnv.snapshot()")
        state = copy.deepcopy(token.state)
        self._events = state["events"]
        self._user_messages = state["user_messages"]
        self._pool_index = state["pool_index"]
