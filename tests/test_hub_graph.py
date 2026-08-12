"""진입점 통합 테스트 (refs #10).

봉투 → 정책 → 라우팅 → 페이로드 → 실행 → 후처리 → 기록.
**1~4 에서 실패하면 LLM 을 부르지 않는다.** 이것이 순서의 존재 이유다.

통합 테스트지만 LLM 은 안 부른다 - ping 과 더미 TaskSpec 만 쓴다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from hub import observability, registry
from hub.graph import graph
from hub.registry import TaskSpec, build_registry

_PING = {"service": "sb", "task": "ping", "request_id": "r-1", "payload": {}}


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log = tmp_path / "calls.jsonl"
    monkeypatch.setattr(observability, "default_log_path", lambda: log)
    return log


def _entries(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]


# --- 정상 경로 ---


def test_ping_round_trips() -> None:
    out = graph.invoke(dict(_PING))
    assert out["output"] == {"pong": True}


def test_ping_is_recorded_as_ok(_isolated_log: Path) -> None:
    graph.invoke(dict(_PING))
    (entry,) = _entries(_isolated_log)
    assert entry["status"] == "ok"
    assert entry["task"] == "ping"
    assert entry["service"] == "sb"
    assert entry["request_id"] == "r-1"


def test_ping_accepts_missing_payload_keys() -> None:
    out = graph.invoke({**_PING, "payload": {}})
    assert out["output"] == {"pong": True}


# --- 미등록 task ---


def test_unknown_task_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown task"):
        graph.invoke({**_PING, "task": "nope"})


def test_unknown_task_error_lists_known_tasks() -> None:
    """호출자가 오타를 바로 고칠 수 있어야 한다."""
    with pytest.raises(ValueError) as exc:
        graph.invoke({**_PING, "task": "nope"})
    assert "ping" in str(exc.value)


def test_unknown_task_is_recorded_as_rejected(_isolated_log: Path) -> None:
    with pytest.raises(ValueError):
        graph.invoke({**_PING, "task": "nope"})
    (entry,) = _entries(_isolated_log)
    assert entry["status"] == "rejected"
    assert entry["task"] == "nope"


# --- 봉투 불량 ---


@pytest.mark.parametrize("missing", ["service", "task", "request_id", "payload"])
def test_missing_envelope_field_is_rejected(missing: str) -> None:
    bad = {k: v for k, v in _PING.items() if k != missing}
    with pytest.raises(Exception):  # noqa: B017 - ValidationError 계열
        graph.invoke(bad)


@pytest.mark.parametrize("missing", ["service", "task", "request_id", "payload"])
def test_missing_envelope_field_is_recorded(missing: str, _isolated_log: Path) -> None:
    """봉투가 불량이어도 호출 시도 자체는 기록에 남아야 한다."""
    bad = {k: v for k, v in _PING.items() if k != missing}
    with pytest.raises(Exception):  # noqa: B017
        graph.invoke(bad)
    entries = _entries(_isolated_log)
    assert len(entries) == 1
    assert entries[0]["status"] == "rejected"


# --- 페이로드 불량 ---


class _NeedsValue(BaseModel):
    value: str


class _EchoGraph:
    def invoke(self, payload: dict) -> dict:
        return {"echo": payload}


def _dummy(task: str = "dummy", **kw) -> TaskSpec:
    base = {
        "task": task,
        "graph": _EchoGraph(),
        "input_model": _NeedsValue,
        "timeout_s": 5.0,
        "description": "테스트용",
    }
    return TaskSpec(**{**base, **kw})


@pytest.fixture
def _with_dummy(monkeypatch: pytest.MonkeyPatch):
    def _install(spec: TaskSpec) -> None:
        monkeypatch.setattr(registry, "REGISTRY", build_registry([registry.REGISTRY["ping"], spec]))

    return _install


def test_payload_schema_violation_is_rejected(_with_dummy) -> None:
    _with_dummy(_dummy())
    with pytest.raises(Exception):  # noqa: B017
        graph.invoke({**_PING, "task": "dummy", "payload": {"wrong": 1}})


def test_payload_violation_is_recorded_as_rejected(_with_dummy, _isolated_log: Path) -> None:
    _with_dummy(_dummy())
    with pytest.raises(Exception):  # noqa: B017
        graph.invoke({**_PING, "task": "dummy", "payload": {"wrong": 1}})
    (entry,) = _entries(_isolated_log)
    assert entry["status"] == "rejected"


def test_valid_payload_reaches_the_graph(_with_dummy) -> None:
    _with_dummy(_dummy())
    out = graph.invoke({**_PING, "task": "dummy", "payload": {"value": "hello"}})
    assert out["output"] == {"echo": {"value": "hello"}}


# --- 그래프 실행 실패 ---


class _BoomGraph:
    def invoke(self, payload: dict) -> dict:
        raise RuntimeError("graph exploded")


def test_graph_exception_is_reraised(_with_dummy) -> None:
    _with_dummy(_dummy("boom", graph=_BoomGraph()))
    with pytest.raises(RuntimeError, match="graph exploded"):
        graph.invoke({**_PING, "task": "boom", "payload": {"value": "x"}})


def test_graph_exception_is_recorded_as_error(_with_dummy, _isolated_log: Path) -> None:
    _with_dummy(_dummy("boom", graph=_BoomGraph()))
    with pytest.raises(RuntimeError):
        graph.invoke({**_PING, "task": "boom", "payload": {"value": "x"}})
    (entry,) = _entries(_isolated_log)
    assert entry["status"] == "error"
    assert "RuntimeError" in entry["error"]


# --- 정책 훅 ---


def test_policy_check_is_called(monkeypatch: pytest.MonkeyPatch) -> None:
    """스텁이라도 호출 지점 존재를 고정한다.

    나중에 인증·쿼터를 붙일 때 hub/graph.py 를 안 고치기 위해서다.
    """
    from hub import graph as graph_mod

    seen: list[str] = []
    monkeypatch.setattr(graph_mod, "check", lambda env: seen.append(env.task))
    graph.invoke(dict(_PING))
    assert seen == ["ping"]


def test_policy_rejection_stops_before_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """정책이 거절하면 그래프에 닿지 않는다."""
    from hub import graph as graph_mod

    def _deny(env) -> None:
        raise PermissionError("denied")

    monkeypatch.setattr(graph_mod, "check", _deny)
    with pytest.raises(PermissionError):
        graph.invoke(dict(_PING))


# --- 후처리 훅 ---


def test_postprocess_is_applied(_with_dummy) -> None:
    def upper(output: dict, payload: BaseModel) -> dict:
        return {"echo": {k: v.upper() for k, v in output["echo"].items()}}

    _with_dummy(_dummy("post", postprocess=upper))
    out = graph.invoke({**_PING, "task": "post", "payload": {"value": "hi"}})
    assert out["output"] == {"echo": {"value": "HI"}}


def test_postprocess_receives_validated_payload(_with_dummy) -> None:
    captured: list[BaseModel] = []

    def spy(output: dict, payload: BaseModel) -> dict:
        captured.append(payload)
        return output

    _with_dummy(_dummy("spy", postprocess=spy))
    graph.invoke({**_PING, "task": "spy", "payload": {"value": "hi"}})
    assert isinstance(captured[0], _NeedsValue)
    assert captured[0].value == "hi"


# --- 경계 ---


def test_adding_a_task_requires_no_change_to_dispatch(_with_dummy) -> None:
    """그래프 추가에 라우터 로직이 바뀌면 경계가 잘못 그어진 것이다.

    레지스트리에 넣는 것만으로 동작해야 한다.
    """
    _with_dummy(_dummy("brand_new"))
    out = graph.invoke({**_PING, "task": "brand_new", "payload": {"value": "ok"}})
    assert out["output"] == {"echo": {"value": "ok"}}


def test_sample_request_matches_the_contract() -> None:
    """samples/ 가 계약의 실물이다. 계약이 바뀌면 여기서 깨진다."""
    repo = Path(__file__).resolve().parent.parent
    sample = json.loads((repo / "samples" / "ping" / "request.json").read_text(encoding="utf-8"))
    assert sample["assistant_id"] == "hub"
    out = graph.invoke(sample["input"])
    assert out["output"] == {"pong": True}


def test_langgraph_json_registers_hub() -> None:
    repo = Path(__file__).resolve().parent.parent
    cfg = json.loads((repo / "langgraph.json").read_text(encoding="utf-8"))
    assert cfg["graphs"]["hub"] == "./hub/graph.py:graph"


# --- 유출 회귀: 검증 실패 경로 ---

_SECRET = "TOP-SECRET-VALUE"


def test_invalid_envelope_does_not_leak_payload(_isolated_log: Path) -> None:
    """봉투가 불량이면 pydantic 이 input_value 에 입력 전체를 싣는다.

    그대로 기록하면 페이로드가 통째로 로그에 남는다. 실제로 유출됐던 경로다.
    """
    with pytest.raises(Exception):  # noqa: B017
        graph.invoke({"task": "ping", "request_id": "r", "payload": {"secret": _SECRET}})
    raw = _isolated_log.read_text(encoding="utf-8")
    assert _SECRET not in raw
    assert "service" in raw, "어느 필드가 잘못됐는지는 남아야 한다"


def test_invalid_payload_does_not_leak_values(_with_dummy, _isolated_log: Path) -> None:
    _with_dummy(_dummy())
    with pytest.raises(Exception):  # noqa: B017
        graph.invoke({**_PING, "task": "dummy", "payload": {"secret": _SECRET}})
    raw = _isolated_log.read_text(encoding="utf-8")
    assert _SECRET not in raw


def test_rejection_reason_still_names_the_bad_field(_isolated_log: Path) -> None:
    """값은 버리되 진단 가능성은 유지한다 - 어느 필드가 왜 틀렸는지."""
    with pytest.raises(Exception):  # noqa: B017
        graph.invoke({"task": "ping", "request_id": "r", "payload": {}})
    (entry,) = _entries(_isolated_log)
    assert "service" in entry["error"]
    assert "missing" in entry["error"]
