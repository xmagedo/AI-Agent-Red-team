
"""Tests for benign scenario exploration."""

from engine.scoring import score_trace
from engine.search import SearchEngine
from engine.stub_env import StubEnv


def make_event(name, args, ok=True):
    """Create a harmless tool event with SDK-compatible fields."""
    return {
        "name": name,
        "args": args,
        "ok": ok,
        "output": "",
        "side_effect": "NETWORK",
        "scope": "public",
        "source": "web",
        "error": None if ok else "not_found",
    }


def make_trace(*events):
    """Create a minimal trace for the stub environment."""
    return {
        "seed": 0,
        "user_messages": [],
        "assistant_messages": [],
        "tool_events": list(events),
        "egress": [],
    }


def build_test_engine():
    fixtures = {
        "search demo": make_trace(
            make_event("web.search", {"query": "demo"})
        ),
        "search python": make_trace(
            make_event("web.search", {"query": "python"})
        ),
        "open page": make_trace(
            make_event("web.open", {"id": "page_0"})
        ),
        "failed open": make_trace(
            make_event("web.open", {"id": "missing"}, ok=False)
        ),
    }

    env = StubEnv(scripted=fixtures)
    return SearchEngine(env)


def test_first_scenario_discovers_cell():
    engine = build_test_engine()

    results = engine.run(["search demo"])

    assert len(results) == 1
    assert results[0].severity == 0
    assert results[0].new_cells == {"web.search(query)"}
    assert results[0].gain == 2


def test_repeated_cell_has_no_new_gain():
    engine = build_test_engine()

    results = engine.run(["search demo", "search python"])

    assert results[0].gain == 2
    assert results[1].gain == 0
    assert len(engine.seen_cells) == 1


def test_different_tools_add_coverage():
    engine = build_test_engine()

    results = engine.run(["search demo", "open page"])

    assert results[0].gain == 2
    assert results[1].gain == 2

    assert engine.seen_cells == {
        "web.search(query)",
        "web.open(id)",
    }


def test_failed_calls_do_not_add_coverage():
    engine = build_test_engine()

    results = engine.run(["failed open"])

    assert results[0].severity == 0
    assert results[0].gain == 0
    assert results[0].new_cells == set()


def test_environment_restored_after_each_scenario():
    engine = build_test_engine()

    before = engine.env.export_trace_dict()

    engine.run(["search demo", "open page"])

    after = engine.env.export_trace_dict()

    assert after == before


def test_coverage_report():
    engine = build_test_engine()

    engine.run([
        "search demo",
        "search python",
        "open page",
    ])

    report = engine.coverage_report()

    assert report["scenarios_tested"] == 3
    assert report["unique_cells"] == 2
    assert report["total_detected_severity"] == 0
    assert report["cells"] == [
        "web.open(id)",
        "web.search(query)",
    ]


def test_unknown_scenario_is_empty():
    engine = build_test_engine()

    results = engine.run(["unknown message"])

    assert results[0].gain == 0
    assert results[0].cells == set()


def test_result_cells_match_score_trace():
    engine = build_test_engine()

    checkpoint = engine.env.snapshot()
    engine.env.interact("open page")
    trace = engine.env.export_trace_dict()
    engine.env.restore(checkpoint)

    results = engine.run(["open page"])

    assert results[0].cells == score_trace(trace)["cells"]
