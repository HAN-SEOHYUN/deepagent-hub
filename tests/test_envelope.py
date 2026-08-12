"""봉투 검증 테스트 (refs #7).

봉투는 호출자가 지켜야 할 유일한 규약이다.
불량 봉투는 LLM 에 닿기 전에 여기서 끊는다 - 토큰을 태우지 않기 위해서다.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hub.envelope import Envelope

_VALID = {
    "service": "sb",
    "task": "word_example",
    "request_id": "sb-batch-20260812-1234",
    "payload": {"english": "prompt", "meaning": "신속한, 즉각적인"},
}


def test_valid_envelope_passes() -> None:
    env = Envelope.model_validate(_VALID)
    assert env.service == "sb"
    assert env.task == "word_example"
    assert env.request_id == "sb-batch-20260812-1234"
    assert env.payload == {"english": "prompt", "meaning": "신속한, 즉각적인"}


@pytest.mark.parametrize("missing", ["service", "task", "request_id", "payload"])
def test_missing_required_field_rejected(missing: str) -> None:
    data = {k: v for k, v in _VALID.items() if k != missing}
    with pytest.raises(ValidationError):
        Envelope.model_validate(data)


@pytest.mark.parametrize("blank", ["service", "task", "request_id"])
def test_blank_string_rejected(blank: str) -> None:
    """빈 문자열은 값이 없는 것과 같다. 기록에 빈 service 가 쌓이면 집계가 깨진다."""
    data = {**_VALID, blank: ""}
    with pytest.raises(ValidationError):
        Envelope.model_validate(data)


@pytest.mark.parametrize("bad", ["not a dict", 42, ["a"], None])
def test_payload_must_be_dict(bad: object) -> None:
    with pytest.raises(ValidationError):
        Envelope.model_validate({**_VALID, "payload": bad})


@pytest.mark.parametrize("bad", ["not a dict", 42, ["a"], None])
def test_input_itself_must_be_dict(bad: object) -> None:
    with pytest.raises(ValidationError):
        Envelope.model_validate(bad)


def test_empty_payload_is_allowed() -> None:
    """payload 가 비어도 봉투는 유효하다 - ping 이 그렇다."""
    env = Envelope.model_validate({**_VALID, "task": "ping", "payload": {}})
    assert env.payload == {}


@pytest.mark.parametrize(
    "payload",
    [
        {"anything": "goes"},
        {"nested": {"deep": {"value": [1, 2, 3]}}},
        {"unicode": "한글도 된다", "num": 3.14, "null": None, "bool": True},
    ],
)
def test_payload_contents_are_not_validated(payload: dict) -> None:
    """허브는 payload 의 내용을 모른다. 검증은 그래프의 몫이다."""
    env = Envelope.model_validate({**_VALID, "payload": payload})
    assert env.payload == payload


def test_extra_fields_are_ignored() -> None:
    """확장 규칙 - 필드 추가는 선택으로만. 여분 필드가 기존 호출자를 깨뜨리면 안 된다."""
    env = Envelope.model_validate({**_VALID, "trace_id": "x", "future_field": 1})
    assert env.service == "sb"
    assert not hasattr(env, "trace_id")


def test_envelope_is_immutable() -> None:
    """봉투는 검증 후 바뀌지 않는다. dispatch 도중 변조를 막는다."""
    env = Envelope.model_validate(_VALID)
    with pytest.raises(ValidationError):
        env.service = "other"  # type: ignore[misc]


def test_error_message_names_the_missing_field() -> None:
    """호출자가 무엇을 빠뜨렸는지 알 수 있어야 한다."""
    with pytest.raises(ValidationError) as exc:
        Envelope.model_validate({k: v for k, v in _VALID.items() if k != "task"})
    assert "task" in str(exc.value)
