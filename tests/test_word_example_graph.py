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

    def _install(generated: dict | None = None, capture: list | None = None, usage: dict | None = None):
        payload = generated if generated is not None else _GENERATED

        class _Raw:
            usage_metadata = usage or {}

        class _Stub:
            def invoke(self, messages):
                if capture is not None:
                    capture.append(messages)
                # include_raw=True 형태를 흉내낸다
                return {"parsed": we.Generated.model_validate(payload), "raw": _Raw()}

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
    # 품질 게이트가 생긴 뒤로는 스텁 응답도 검증을 통과해야 한다.
    _stub_llm(
        {
            "example": "Let's put off the meeting until next Monday.",
            "example_ko": "회의는 다음 주 월요일로 미루죠.",
        }
    )
    payload = {"english": "put off", "meaning": "미루다, 연기하다"}
    out = we.graph.invoke(WordExampleIn.model_validate(payload).model_dump())
    assert out["english"] == "put off"


def test_model_field_reports_the_configured_model(_stub_llm, monkeypatch) -> None:
    monkeypatch.setattr(we, "_MODEL", "anthropic:claude-sonnet-4-6")
    _stub_llm()
    out = we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())
    assert out["model"] == "anthropic:claude-sonnet-4-6"


def test_graph_output_carries_usage_for_recording(_stub_llm) -> None:
    """그래프 출력에는 usage 가 실린다 - 허브가 기록용으로 가져간다."""
    _stub_llm()
    out = we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())
    assert set(out) == {"english", "example", "example_ko", "model", "usage"}


def test_hub_strips_usage_from_the_response(_stub_llm) -> None:
    """usage 는 계약의 일부가 아니다. 호출자가 볼 필드가 아니다."""
    _stub_llm()
    out = hub_graph.invoke(_envelope())
    assert set(out["output"]) == {"english", "example", "example_ko", "model"}


def test_usage_is_recorded(_stub_llm, _isolated_log: Path) -> None:
    """비용 환산의 근거다. 기록되지 않으면 #14 품질 실측이 불가능하다."""
    _stub_llm(usage={"input_tokens": 210, "output_tokens": 64})
    hub_graph.invoke(_envelope())
    entry = json.loads(_isolated_log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["tokens_in"] == 210
    assert entry["tokens_out"] == 64


def test_regeneration_accumulates_tokens(_stub_sequence, _isolated_log: Path) -> None:
    """재생성해도 실제로 쓴 만큼이 기록돼야 비용 환산이 맞는다."""
    _stub_sequence([_BAD_SHORT, _GENERATED], usage={"input_tokens": 100, "output_tokens": 30})
    hub_graph.invoke(_envelope())
    entry = json.loads(_isolated_log.read_text(encoding="utf-8").splitlines()[0])
    assert entry["tokens_in"] == 200
    assert entry["tokens_out"] == 60


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
    # 스텁이 내는 문장(_GENERATED)과 다른 문장을 피하라고 준다 - 같으면 중복 위반이 된다.
    payload = {**_PAYLOAD, "avoid": ["The store gave us a prompt refund without any fuss."]}
    we.graph.invoke(WordExampleIn.model_validate(payload).model_dump())
    text = json.dumps(seen, ensure_ascii=False, default=str)
    assert "The store gave us a prompt refund without any fuss." in text


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


# --- 재생성 루프 (refs #19) ---

_BAD_SHORT = {"example": "Prompt reply.", "example_ko": "빠른 답변."}
_BAD_NO_TARGET = {
    "example": "Thanks for your quick reply to my message.",
    "example_ko": "빠르게 답해 주셔서 고마워요.",
}


@pytest.fixture
def _stub_sequence(monkeypatch: pytest.MonkeyPatch):
    """호출마다 다른 응답을 주는 스텁. 호출 기록을 남긴다."""

    def _install(responses: list[dict], usage: dict | None = None):
        state = {"calls": 0, "messages": []}

        class _Raw:
            usage_metadata = usage or {}

        class _Stub:
            def invoke(self, messages):
                state["messages"].append(messages)
                i = min(state["calls"], len(responses) - 1)
                state["calls"] += 1
                return {"parsed": we.Generated.model_validate(responses[i]), "raw": _Raw()}

        monkeypatch.setattr(we, "_structured", lambda: _Stub())
        return state

    return _install


def test_passing_output_is_not_regenerated(_stub_sequence) -> None:
    st = _stub_sequence([_GENERATED])
    we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())
    assert st["calls"] == 1, "통과했는데 재생성했다"


def test_violation_triggers_one_regeneration(_stub_sequence) -> None:
    st = _stub_sequence([_BAD_SHORT, _GENERATED])
    out = we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())
    assert st["calls"] == 2
    assert out["example"] == _GENERATED["example"]


def test_second_violation_raises(_stub_sequence) -> None:
    _stub_sequence([_BAD_SHORT, _BAD_SHORT])
    with pytest.raises(ValueError):
        we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())


def test_at_most_two_llm_calls(_stub_sequence) -> None:
    """재생성은 1회만. 무한 루프가 나면 토큰이 무한히 탄다."""
    st = _stub_sequence([_BAD_SHORT, _BAD_SHORT])
    with pytest.raises(ValueError):
        we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())
    assert st["calls"] == 2


def test_error_names_the_violated_rule(_stub_sequence) -> None:
    _stub_sequence([_BAD_NO_TARGET, _BAD_NO_TARGET])
    with pytest.raises(ValueError) as exc:
        we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())
    assert "prompt" in str(exc.value), "어느 기준을 어겼는지 알 수 없다"


def test_failed_attempt_is_added_to_avoid(_stub_sequence) -> None:
    """재생성 프롬프트에 직전 실패작이 실려야 같은 답을 안 받는다."""
    st = _stub_sequence([_BAD_SHORT, _GENERATED])
    we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())
    retry_prompt = json.dumps(st["messages"][1], ensure_ascii=False, default=str)
    assert _BAD_SHORT["example"] in retry_prompt


def test_original_avoid_survives_the_retry(_stub_sequence) -> None:
    """재생성 때 원래 회피 목록이 사라지면 안 된다."""
    original = "The store gave us a prompt refund."
    st = _stub_sequence([_BAD_SHORT, _GENERATED])
    payload = {**_PAYLOAD, "avoid": [original]}
    we.graph.invoke(WordExampleIn.model_validate(payload).model_dump())
    retry_prompt = json.dumps(st["messages"][1], ensure_ascii=False, default=str)
    assert original in retry_prompt
    assert _BAD_SHORT["example"] in retry_prompt


def test_regeneration_failure_is_recorded_as_error(
    _stub_sequence, _isolated_log: Path
) -> None:
    """LLM 을 불렀으므로 rejected 가 아니라 error 다 - 토큰을 태웠다."""
    _stub_sequence([_BAD_SHORT, _BAD_SHORT])
    with pytest.raises(Exception):  # noqa: B017
        hub_graph.invoke(_envelope())
    entries = [
        json.loads(ln) for ln in _isolated_log.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert entries[0]["status"] == "error"


def test_duplicate_of_avoid_triggers_regeneration(_stub_sequence) -> None:
    dup = {"example": "The store gave us a prompt refund.", "example_ko": "바로 환불해 줬어요."}
    st = _stub_sequence([dup, _GENERATED])
    payload = {**_PAYLOAD, "avoid": ["The store gave us a prompt refund."]}
    we.graph.invoke(WordExampleIn.model_validate(payload).model_dump())
    assert st["calls"] == 2


def test_missing_korean_triggers_regeneration(_stub_sequence) -> None:
    bad_ko = {"example": _GENERATED["example"], "example_ko": "Thanks for the reply."}
    st = _stub_sequence([bad_ko, _GENERATED])
    we.graph.invoke(WordExampleIn.model_validate(_PAYLOAD).model_dump())
    assert st["calls"] == 2
