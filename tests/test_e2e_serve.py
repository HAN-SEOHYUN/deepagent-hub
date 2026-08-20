"""E2E - 와이어 호환 봉인 (refs #26).

samples/ 실물을 **무수정으로** 통과시키고, 계약표(docs/integration.md §5,
#21 본문)의 모든 셀을 잠근다. 여기 있는 기대값은 전부 2026-08-20 현행
langgraph dev 실측이다 - 바꾸려면 호출자(sb-be)와 함께 바꿔야 한다.

LLM 은 스텁(_stub_llm 컨벤션) - CI 에서 키·토큰 없이 돈다.
실 LLM 검증은 컨테이너에 대고 따로 실행해 이슈 코멘트에 기록한다 (DoD).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hub import observability
from serve import app as serve_app
from serve.app import app
from word_example import graph as we

_ROOT = Path(__file__).resolve().parent.parent

_GENERATED = {
    "example": "Thanks for your prompt reply to my message.",
    "example_ko": "제 메시지에 빠르게 답해 주셔서 고마워요.",
}


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    log = tmp_path / "calls.jsonl"
    monkeypatch.setattr(observability, "default_log_path", lambda: log)
    return log


@pytest.fixture
def _stub_llm(monkeypatch: pytest.MonkeyPatch):
    def _install(generated: dict | None = None, usage: dict | None = None, raises: Exception | None = None):
        payload = generated if generated is not None else _GENERATED

        class _Raw:
            usage_metadata = usage or {"input_tokens": 512, "output_tokens": 31}

        class _Stub:
            def invoke(self, messages):
                if raises is not None:
                    raise raises
                return {"parsed": we.Generated.model_validate(payload), "raw": _Raw()}

        monkeypatch.setattr(we, "_structured", lambda: _Stub())

    return _install


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


def _sample(name: str) -> dict:
    """samples/ 실물을 그대로 읽는다 - 여기서 한 글자도 고치지 않는 것이 계약이다."""
    return json.loads((_ROOT / "samples" / name / "request.json").read_text())


def _err(resp) -> dict:
    assert resp.status_code == 200, f"__error__ 는 200 이어야 한다 (실측): {resp.status_code}"
    return resp.json()["__error__"]


# --- 정상 2 - samples 무수정 --------------------------------------------------


def test_sample_ping_verbatim(client: TestClient) -> None:
    body = _sample("ping")
    r = client.post("/runs/wait", json=body)
    assert r.status_code == 200
    state = r.json()
    assert state["output"] == {"pong": True}
    for k, v in body["input"].items():
        assert state[k] == v, f"최종 상태가 봉투를 에코하지 않는다: {k}"


def test_sample_word_example_verbatim(client: TestClient, _stub_llm) -> None:
    _stub_llm()
    body = _sample("word_example")
    r = client.post("/runs/wait", json=body)
    assert r.status_code == 200
    out = r.json()["output"]
    assert out["english"] == body["input"]["payload"]["english"]  # 에코백 계약
    assert set(out) == {"english", "example", "example_ko", "model"}  # usage 없음


# --- __error__ (전부 HTTP 200) ------------------------------------------------


def test_error_invalid_envelope(client: TestClient) -> None:
    e = _err(client.post("/runs/wait", json={"assistant_id": "hub", "input": {}}))
    assert e["error"] == "ValidationError"


def test_error_unknown_task(client: TestClient) -> None:
    body = _sample("ping")
    body["input"]["task"] = "nope"
    e = _err(client.post("/runs/wait", json=body))
    assert e["error"] == "ValueError"
    assert "unknown task" in e["message"]


def test_error_invalid_payload(client: TestClient) -> None:
    body = _sample("word_example")
    del body["input"]["payload"]["meaning"]
    e = _err(client.post("/runs/wait", json=body))
    assert e["error"] == "ValidationError"


def test_error_missing_input(client: TestClient) -> None:
    e = _err(client.post("/runs/wait", json={"assistant_id": "hub"}))
    assert e["error"] == "EmptyInputError"


def test_error_non_dict_input(client: TestClient) -> None:
    e = _err(client.post("/runs/wait", json={"assistant_id": "hub", "input": [1, 2]}))
    assert e["error"] == "InvalidUpdateError"


def test_error_llm_failure(client: TestClient, _stub_llm) -> None:
    _stub_llm(raises=RuntimeError("LLM down"))
    e = _err(client.post("/runs/wait", json=_sample("word_example")))
    assert e["error"] == "RuntimeError"


def test_error_quality_violation(client: TestClient, _stub_llm) -> None:
    _stub_llm(generated={"example": "No target here at all.", "example_ko": "대상 없음."})
    e = _err(client.post("/runs/wait", json=_sample("word_example")))
    assert e["error"] == "ValueError"
    assert "품질 기준 위반" in e["message"]


def test_error_timeout(client: TestClient, monkeypatch: pytest.MonkeyPatch, _isolated_log: Path) -> None:
    class _Slow:
        def invoke(self, _input):
            time.sleep(1.0)

    class _FakeSpec:
        timeout_s = 0.05

    monkeypatch.setitem(serve_app.ASSISTANTS, "slow", _Slow())
    monkeypatch.setitem(serve_app.DIRECT_TASKS, "slow", "slow_task")
    monkeypatch.setattr(serve_app, "get_spec", lambda t: _FakeSpec if t == "slow_task" else None)
    e = _err(client.post("/runs/wait", json={"assistant_id": "slow", "input": {}}))
    assert e["error"] == "TimeoutError"
    assert not _isolated_log.exists() or _isolated_log.read_text() == ""


# --- 422 ----------------------------------------------------------------------


def test_422_unknown_assistant(client: TestClient) -> None:
    r = client.post("/runs/wait", json={"assistant_id": "nope", "input": {}})
    assert r.status_code == 422
    assert "Invalid assistant: 'nope'" in r.json()["detail"]


def test_422_missing_assistant(client: TestClient) -> None:
    r = client.post("/runs/wait", json={"input": {}})
    assert r.status_code == 422


# --- 직통 (D3 탈출구) ----------------------------------------------------------


def test_direct_includes_usage(client: TestClient, _stub_llm) -> None:
    _stub_llm()
    r = client.post(
        "/runs/wait",
        json={"assistant_id": "word-example", "input": _sample("word_example")["input"]["payload"]},
    )
    assert r.status_code == 200
    assert r.json()["usage"] == {"input_tokens": 512, "output_tokens": 31}


def test_hub_strips_usage(client: TestClient, _stub_llm) -> None:
    _stub_llm()
    r = client.post("/runs/wait", json=_sample("word_example"))
    assert "usage" not in r.json()["output"]


def test_direct_is_not_recorded(client: TestClient, _stub_llm, _isolated_log: Path) -> None:
    _stub_llm()
    client.post(
        "/runs/wait",
        json={"assistant_id": "word-example", "input": _sample("word_example")["input"]["payload"]},
    )
    assert not _isolated_log.exists() or _isolated_log.read_text() == ""


# --- 명단·문서 정합 ------------------------------------------------------------


def test_assistants_match_langgraph_json(client: TestClient) -> None:
    """serve 의 명시적 명단과 langgraph.json 의 graphs 키가 갈라지면 실패한다."""
    graphs = json.loads((_ROOT / "langgraph.json").read_text())["graphs"]
    assert set(serve_app.ASSISTANTS) == set(graphs)


def test_api_json_matches_serving_reality() -> None:
    doc = json.loads((_ROOT / "docs" / "api.json").read_text())
    assert doc["version"] == "0.3.0"
    assert "8000" in doc["transport"]["base_url"]["local"]
    assert "svc.cluster.local:8000" in doc["transport"]["base_url"]["server"]
    assert doc["transport"]["endpoint"] == "POST /runs/wait"
