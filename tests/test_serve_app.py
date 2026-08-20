"""serve/app.py 테스트 (refs #22).

**LLM 을 부르지 않는다** - `tests/test_word_example_graph.py` 와 같은 컨벤션으로
구조화 출력 호출을 스텁으로 갈아끼운다. API 키 없이 돈다.

에러 어댑터(`__error__`)·타임아웃 계약은 #24 에서 검증한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hub import observability
from serve.app import ASSISTANTS, RunRequest, app
from word_example import graph as we

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
    """구조화 출력만 갈아끼운다. 프롬프트 조립·검증·응답 조립은 실제 코드가 돈다."""

    def _install(generated: dict | None = None, usage: dict | None = None):
        payload = generated if generated is not None else _GENERATED

        class _Raw:
            usage_metadata = usage or {"input_tokens": 10, "output_tokens": 5}

        class _Stub:
            def invoke(self, messages):
                return {"parsed": we.Generated.model_validate(payload), "raw": _Raw()}

        monkeypatch.setattr(we, "_structured", lambda: _Stub())

    return _install


@pytest.fixture
def client() -> TestClient:
    # raise_server_exceptions=False: 어댑터가 없는 동안 그래프 예외가 500 으로
    # 떨어지는 것을 테스트가 관찰할 수 있게 한다 (#24 전까지의 중간 상태).
    return TestClient(app, raise_server_exceptions=False)


def _envelope(task: str = "ping", payload: dict | None = None) -> dict:
    return {
        "service": "sb",
        "task": task,
        "request_id": "test-serve",
        "payload": payload or {},
    }


# --- /healthz ---------------------------------------------------------------


def test_healthz_returns_200_without_llm(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


# --- /runs/wait - hub 경유 --------------------------------------------------


def test_hub_ping_returns_full_final_state(client: TestClient) -> None:
    """응답은 output 만이 아니라 **최종 상태 전체**다 (2026-08-20 현행 실측)."""
    r = client.post("/runs/wait", json={"assistant_id": "hub", "input": _envelope()})
    assert r.status_code == 200
    body = r.json()
    assert body["output"] == {"pong": True}
    for key in ("service", "task", "request_id", "payload"):
        assert key in body, f"최종 상태에 {key} 가 없다"


def test_hub_word_example_strips_usage(client: TestClient, _stub_llm) -> None:
    """허브 경유 응답에는 usage 가 없다 - 관측은 계약의 일부가 아니다."""
    _stub_llm()
    payload = {"english": "prompt", "meaning": "신속한, 즉각적인"}
    r = client.post(
        "/runs/wait",
        json={"assistant_id": "hub", "input": _envelope("word_example", payload)},
    )
    assert r.status_code == 200
    out = r.json()["output"]
    assert out["english"] == "prompt"
    assert out["example"] == _GENERATED["example"]
    assert "usage" not in out


def test_hub_call_is_recorded(client: TestClient, _isolated_log: Path) -> None:
    client.post("/runs/wait", json={"assistant_id": "hub", "input": _envelope()})
    lines = _isolated_log.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["task"] == "ping"


# --- /runs/wait - 직통 (D3 탈출구) -------------------------------------------


def test_direct_word_example_includes_usage(client: TestClient, _stub_llm) -> None:
    """직통 응답은 그래프 최종 상태 그대로 - usage 포함 (2026-08-20 실측 미러)."""
    _stub_llm(usage={"input_tokens": 512, "output_tokens": 31})
    r = client.post(
        "/runs/wait",
        json={
            "assistant_id": "word-example",
            "input": {"english": "prompt", "meaning": "신속한, 즉각적인"},
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["example"] == _GENERATED["example"]
    assert body["usage"] == {"input_tokens": 512, "output_tokens": 31}


def test_direct_call_is_not_recorded(client: TestClient, _stub_llm, _isolated_log: Path) -> None:
    """직통은 관측에 안 남는다 (api.json entrypoints.direct: records_calls false)."""
    _stub_llm()
    client.post(
        "/runs/wait",
        json={"assistant_id": "word-example", "input": {"english": "a", "meaning": "b"}},
    )
    assert not _isolated_log.exists() or _isolated_log.read_text() == ""


# --- 요청 모델 ---------------------------------------------------------------


def test_input_is_optional_in_request_model() -> None:
    """input 을 required 로 바꾸면 안 된다 - 누락은 422 가 아니라
    그래프의 EmptyInputError(200 + __error__) 여야 한다 (#24 에서 완성)."""
    req = RunRequest(assistant_id="hub")
    assert req.input is None


def test_unknown_assistant_returns_422_with_registered_list(client: TestClient) -> None:
    r = client.post("/runs/wait", json={"assistant_id": "nope", "input": {}})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "Invalid assistant: 'nope'" in detail
    assert "hub" in detail and "word-example" in detail


def test_assistants_roster_is_explicit() -> None:
    assert set(ASSISTANTS) == {"hub", "word-example"}
