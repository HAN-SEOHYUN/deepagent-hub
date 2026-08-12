"""word-example 그래프 테스트 (refs #17).

**LLM 을 부르지 않는다** - 구조화 출력 호출을 스텁으로 갈아끼운다.
API 키 없이 돌아야 CI 와 개발 초기에 쓸모가 있다.

실호출 검증(예문이 실제로 나오는지, 모델 교체가 되는지)은 키가 필요하며
이슈 #17 의 수용 기준에 따로 남아 있다.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hub import observability, registry
from hub.graph import graph as hub_graph
from word_example import graph as we
from word_example.schema import WordExampleIn

_PAYLOAD = {"english": "prompt", "meaning": "신속한, 즉각적인"}
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

    def _install(generated: dict | None = None, capture: list | None = None):
        payload = generated if generated is not None else _GENERATED

        class _Stub:
            def invoke(self, messages):
                if capture is not None:
                    capture.append(messages)
                return we.Generated.model_validate(payload)

        monkeypatch.setattr(we, "_structured", lambda: _Stub())

    return _install


def _envelope(**kw) -> dict:
    return {
        "service": "sb",
        "task": "word_example",
        "request_id": "t-1",
        "payload": {**_PAYLOAD, **kw},
    }


# --- 응답 조립 ---


def test_generates_example(_stub_llm) -> None:
    _stub_llm()
    out = we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())
    assert out["example"] == _GENERATED["example"]
    assert out["example_ko"] == _GENERATED["example_ko"]


def test_english_is_echoed_back_from_the_request(_stub_llm) -> None:
    """에코백은 **코드가 보장**한다. LLM 이 만들면 방어가 무의미해진다."""
    _stub_llm()
    out = we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())
    assert out["english"] == "prompt"


def test_echo_back_is_byte_identical_for_phrasal_verb(_stub_llm) -> None:
    _stub_llm()
    payload = {"english": "put off", "meaning": "미루다, 연기하다"}
    out = we.graph.invoke(WordExampleIn.model_validate(payload).model_dump())
    assert out["english"] == "put off"


def test_model_field_reports_the_configured_model(_stub_llm, monkeypatch) -> None:
    monkeypatch.setattr(we, "_MODEL", "anthropic:claude-sonnet-4-6")
    _stub_llm()
    out = we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())
    assert out["model"] == "anthropic:claude-sonnet-4-6"


def test_output_matches_the_contract(_stub_llm) -> None:
    _stub_llm()
    out = we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())
    assert set(out) == {"english", "example", "example_ko", "model"}


# --- 프롬프트 전달 ---


def test_prompt_carries_word_and_meaning(_stub_llm) -> None:
    seen: list = []
    _stub_llm(capture=seen)
    we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())
    text = json.dumps(seen, ensure_ascii=False, default=str)
    assert "prompt" in text
    assert "신속한" in text


def test_avoid_reaches_the_prompt(_stub_llm) -> None:
    seen: list = []
    _stub_llm(capture=seen)
    payload = {**_PAYLOAD, "avoid": ["Thanks for your prompt reply to my message."]}
    we.graph.invoke(WordExampleIn.model_validate(payload).model_dump())
    text = json.dumps(seen, ensure_ascii=False, default=str)
    assert "Thanks for your prompt reply to my message." in text


# --- 모델 설정 ---


def test_default_model_is_gpt_mini() -> None:
    assert we._default_model() == "openai:gpt-4.1-mini"


def test_blank_env_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """.env 에 키만 있고 값이 빈 경우까지 미설정으로 취급한다."""
    monkeypatch.setenv("WORD_EXAMPLE_MODEL", "")
    assert we._default_model() == "openai:gpt-4.1-mini"


def test_env_overrides_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORD_EXAMPLE_MODEL", "anthropic:claude-sonnet-4-6")
    assert we._default_model() == "anthropic:claude-sonnet-4-6"


def test_importing_the_graph_needs_no_api_key() -> None:
    """모듈 로드 시 LLM 을 만들면 키 없이는 허브 전체가 못 뜬다.

    registry 가 이 모듈을 import 하므로, 여기서 죽으면 ping 조차 안 된다.
    """
    assert we.graph is not None
    assert we.SPEC.task == "word_example"


# --- 허브 등록 ---


def test_registered_in_hub_registry() -> None:
    assert registry.get("word_example") is not None
    assert registry.get("word_example").input_model is WordExampleIn


def test_timeout_is_shorter_than_caller_budget() -> None:
    """호출자가 먼저 끊으면 고아 run 이 토큰만 태운다."""
    assert registry.get("word_example").timeout_s < 50


def test_reachable_through_the_hub_envelope(_stub_llm) -> None:
    _stub_llm()
    out = hub_graph.invoke(_envelope())
    assert out["output"]["english"] == "prompt"
    assert out["output"]["example"] == _GENERATED["example"]


def test_missing_meaning_is_rejected_before_the_llm(_stub_llm, _isolated_log: Path) -> None:
    """LLM 을 부르기 전에 끊는다 - 토큰을 태우지 않는다."""
    called: list = []
    _stub_llm(capture=called)
    with pytest.raises(Exception):  # noqa: B017
        hub_graph.invoke(
            {
                "service": "sb",
                "task": "word_example",
                "request_id": "t-2",
                "payload": {"english": "prompt"},
            }
        )
    assert called == [], "LLM 이 불렸다"
    entries = [
        json.loads(ln) for ln in _isolated_log.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert entries[0]["status"] == "rejected"


def test_langgraph_json_registers_word_example() -> None:
    repo = Path(__file__).resolve().parent.parent
    cfg = json.loads((repo / "langgraph.json").read_text(encoding="utf-8"))
    assert cfg["graphs"]["word-example"] == "./word_example/graph.py:graph"


def test_sample_request_matches_the_contract(_stub_llm) -> None:
    _stub_llm()
    repo = Path(__file__).resolve().parent.parent
    sample = json.loads(
        (repo / "samples" / "word_example" / "request.json").read_text(encoding="utf-8")
    )
    assert sample["assistant_id"] == "hub"
    out = hub_graph.invoke(sample["input"])
    assert set(out["output"]) == {"english", "example", "example_ko", "model"}


def test_env_example_documents_the_model_var() -> None:
    repo = Path(__file__).resolve().parent.parent
    assert "WORD_EXAMPLE_MODEL=" in (repo / ".env.example").read_text(encoding="utf-8")
