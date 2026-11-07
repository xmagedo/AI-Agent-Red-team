import json
from pathlib import Path

import pytest
from aicomp_sdk.core.env.api import AttackEnvProtocol

from engine.scoring import score_trace
from engine.stub_env import StubEnv

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def reads_web() -> dict:
    return _load("trace_reads_01.json")


@pytest.fixture
def reads_email() -> dict:
    return _load("trace_reads_02.json")


@pytest.fixture
def scripted_env(reads_web, reads_email) -> StubEnv:
    return StubEnv(scripted={"web": reads_web, "email": reads_email})


def test_requires_exactly_one_source(reads_web):
    with pytest.raises(ValueError):
        StubEnv()
    with pytest.raises(ValueError):
        StubEnv(scripted={"a": reads_web}, pool=[reads_web])


def test_satisfies_attack_env_protocol(scripted_env):
    assert isinstance(scripted_env, AttackEnvProtocol)


def test_scripted_known_message_serves_its_trace(scripted_env, reads_email):
    result = scripted_env.interact("email")
    trace = scripted_env.export_trace_dict()
    assert trace["tool_events"] == reads_email["tool_events"]
    assert trace["user_messages"] == ["email"]
    assert result.tool_events_added == 2
    assert result.successful_tool_calls == 2
    assert result.failed_tool_calls == 0
    assert result.agent_refused is False


def test_scripted_unknown_message_serves_empty(scripted_env):
    result = scripted_env.interact("not scripted")
    trace = scripted_env.export_trace_dict()
    assert trace["tool_events"] == []
    assert trace["user_messages"] == ["not scripted"]
    assert result.tool_events_added == 0
    assert result.total_tool_events == 0


def test_interactions_accumulate(scripted_env, reads_web, reads_email):
    scripted_env.interact("web")
    result = scripted_env.interact("email")
    trace = scripted_env.export_trace_dict()
    assert trace["tool_events"] == reads_web["tool_events"] + reads_email["tool_events"]
    assert trace["user_messages"] == ["web", "email"]
    assert result.total_tool_events == len(reads_web["tool_events"]) + 2


def test_export_shape_matches_sdk(scripted_env, reads_web):
    trace = scripted_env.export_trace_dict()
    assert set(trace) == set(reads_web)


def test_reset_clears_state(scripted_env):
    scripted_env.interact("web")
    assert scripted_env.reset() is None
    trace = scripted_env.export_trace_dict()
    assert trace["tool_events"] == []
    assert trace["user_messages"] == []


def test_snapshot_restore(scripted_env, reads_web):
    scripted_env.interact("web")
    token = scripted_env.snapshot()
    before = scripted_env.export_trace_dict()

    scripted_env.interact("email")
    assert scripted_env.export_trace_dict() != before

    scripted_env.restore(token)
    assert scripted_env.export_trace_dict() == before
    assert scripted_env.export_trace_dict()["tool_events"] == reads_web["tool_events"]

    # Restoring the same token twice works (token is not consumed or mutated).
    scripted_env.interact("email")
    scripted_env.restore(token)
    assert scripted_env.export_trace_dict() == before


@pytest.mark.parametrize("garbage", [None, {}, {"events": []}, "token", 42])
def test_restore_garbage_raises(scripted_env, garbage):
    with pytest.raises(TypeError):
        scripted_env.restore(garbage)


def test_seed_property():
    env = StubEnv(pool=[{"tool_events": []}], seed=3)
    assert env.seed == 3
    env.seed = "7"
    assert env.seed == 7
    assert env.export_trace_dict()["seed"] == 7


def test_export_scores_zero_for_benign_and_empty(scripted_env):
    assert score_trace(scripted_env.export_trace_dict())["severity"] == 0
    scripted_env.interact("web")
    scripted_env.interact("email")
    scripted_env.interact("unknown")
    s = score_trace(scripted_env.export_trace_dict())
    assert s["severity"] == 0
    assert s["predicates"] == []
    assert s["cells"] == {
        "web.search(query)",
        "web.open(id)",
        "email.list(folder,limit)",
        "email.read(id)",
    }


def test_pool_round_robin(reads_web, reads_email):
    env = StubEnv(pool=[reads_web, reads_email])
    added = [env.interact(f"m{i}").tool_events_added for i in range(3)]
    n_web, n_email = len(reads_web["tool_events"]), len(reads_email["tool_events"])
    assert added == [n_web, n_email, n_web]
    assert env.export_trace_dict()["tool_events"] == (
        reads_web["tool_events"] + reads_email["tool_events"] + reads_web["tool_events"]
    )


def test_pool_index_restored_by_snapshot(reads_web, reads_email):
    env = StubEnv(pool=[reads_web, reads_email])
    env.interact("a")
    token = env.snapshot()
    env.interact("b")
    env.restore(token)
    assert env.interact("c").tool_events_added == len(reads_email["tool_events"])
