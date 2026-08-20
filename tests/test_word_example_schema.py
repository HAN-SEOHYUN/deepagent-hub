"""word-example 입출력 스키마 테스트 (refs #15).

`meaning` 이 필수인 이유: `prompt` 는 「신속한」(형용사)이자 「자극하다」(동사)이자
「프롬프트」(명사)다. 뜻 없이는 어느 쪽인지 알 수 없다.

`avoid` 가 있는 이유: 재생성을 눌렀는데 같은 문장이 나오면 사용자는 고장으로 여긴다.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from word_example.schema import WordExampleIn, WordExampleOut

_IN = {"english": "prompt", "meaning": "신속한, 즉각적인"}
_OUT = {
    "english": "prompt",
    "example": "Thanks for your prompt reply to my message.",
    "example_ko": "제 메시지에 빠르게 답해 주셔서 고마워요.",
    "model": "openai:gpt-4.1-mini",
}


# --- 입력 ---


def test_minimal_input_passes() -> None:
    got = WordExampleIn.model_validate(_IN)
    assert got.english == "prompt"
    assert got.meaning == "신속한, 즉각적인"
    assert got.part_of_speech is None
    assert got.avoid == []


@pytest.mark.parametrize("field", ["english", "meaning"])
def test_required_field_missing_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        WordExampleIn.model_validate({k: v for k, v in _IN.items() if k != field})


@pytest.mark.parametrize("field", ["english", "meaning"])
def test_required_field_blank_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        WordExampleIn.model_validate({**_IN, field: ""})


@pytest.mark.parametrize("field", ["english", "meaning"])
def test_required_field_whitespace_only_is_rejected(field: str) -> None:
    """공백만 있는 값은 없는 것과 같다."""
    with pytest.raises(ValidationError):
        WordExampleIn.model_validate({**_IN, field: "   "})


def test_part_of_speech_is_optional() -> None:
    assert WordExampleIn.model_validate({**_IN, "part_of_speech": "adjective"}).part_of_speech == (
        "adjective"
    )


def test_avoid_defaults_to_empty_list() -> None:
    assert WordExampleIn.model_validate(_IN).avoid == []


def test_avoid_default_is_not_shared_between_instances() -> None:
    """가변 기본값이 공유되면 한 호출의 avoid 가 다른 호출에 샌다."""
    a = WordExampleIn.model_validate(_IN)
    b = WordExampleIn.model_validate(_IN)
    a.avoid.append("leaked")
    assert b.avoid == []


def test_avoid_accepts_strings() -> None:
    got = WordExampleIn.model_validate({**_IN, "avoid": ["one.", "two."]})
    assert got.avoid == ["one.", "two."]


@pytest.mark.parametrize("bad", ["not a list", 42, [1, 2], [None], {"a": 1}])
def test_avoid_must_be_list_of_strings(bad: object) -> None:
    with pytest.raises(ValidationError):
        WordExampleIn.model_validate({**_IN, "avoid": bad})


def test_avoid_drops_blank_entries() -> None:
    """빈 문자열은 회피 대상이 아니다. 프롬프트에 실리면 지시만 흐려진다."""
    got = WordExampleIn.model_validate({**_IN, "avoid": ["real.", "", "  "]})
    assert got.avoid == ["real."]


def test_input_extra_fields_are_ignored() -> None:
    got = WordExampleIn.model_validate({**_IN, "future_field": 1})
    assert not hasattr(got, "future_field")


def test_phrasal_verb_is_accepted() -> None:
    got = WordExampleIn.model_validate({"english": "put off", "meaning": "미루다, 연기하다"})
    assert got.english == "put off"


# --- 출력 ---


def test_valid_output_passes() -> None:
    got = WordExampleOut.model_validate(_OUT)
    assert got.english == "prompt"
    assert got.model == "openai:gpt-4.1-mini"


@pytest.mark.parametrize("field", ["english", "example", "example_ko", "model"])
def test_output_field_is_required(field: str) -> None:
    with pytest.raises(ValidationError):
        WordExampleOut.model_validate({k: v for k, v in _OUT.items() if k != field})


@pytest.mark.parametrize("field", ["english", "example", "example_ko", "model"])
def test_output_field_blank_is_rejected(field: str) -> None:
    """반쯤 성공한 응답을 만들지 않는다.

    빈 문자열을 200 으로 보내면 그 쓰레기가 호출자 DB 에 저장되고 화면에 뜬다.
    """
    with pytest.raises(ValidationError):
        WordExampleOut.model_validate({**_OUT, field: ""})


@pytest.mark.parametrize("field", ["example", "example_ko"])
def test_output_whitespace_only_is_rejected(field: str) -> None:
    with pytest.raises(ValidationError):
        WordExampleOut.model_validate({**_OUT, field: "   "})
