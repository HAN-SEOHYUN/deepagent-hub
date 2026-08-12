"""품질 검증 함수 테스트 (refs #18).

「자연스럽다」는 검증할 수 없다. **셀 수 있는 것만 여기서 강제한다.**
프롬프트에 적는 것만으로는 안 지켜지므로 코드가 확인하고 다시 시킨다.

전부 순수 함수 - LLM 을 부르지 않는다.
"""

from __future__ import annotations

import pytest

from word_example.checks import (
    contains_target,
    duplicates_any,
    has_korean,
    length_ok,
    violations,
    word_count,
)
from word_example.schema import WordExampleIn


def _payload(**kw) -> WordExampleIn:
    return WordExampleIn.model_validate({"english": "prompt", "meaning": "신속한", **kw})


# --- 대상 단어 포함 ---


def test_exact_word_is_found() -> None:
    assert contains_target("Thanks for your prompt reply.", "prompt")


@pytest.mark.parametrize(
    ("example", "target"),
    [
        ("She is running to the store.", "run"),
        ("He ran home in the rain.", "run"),
        ("She runs every morning.", "run"),
        ("We are thriving these days.", "thrive"),
        ("He lingered by the door.", "linger"),
    ],
)
def test_inflection_is_accepted(example: str, target: str) -> None:
    """어형 변화 허용 - 실제 쓰임을 보여주는 게 목적이다."""
    assert contains_target(example, target)


def test_case_difference_is_accepted() -> None:
    assert contains_target("Prompt replies are always welcome.", "prompt")


@pytest.mark.parametrize(
    "example",
    [
        "Let's put off the meeting until Monday.",
        "Let's put the meeting off until Monday.",
        "They put it off again this week.",
    ],
)
def test_phrasal_verb_is_found_even_when_split(example: str) -> None:
    """구동사는 흩어져도 인정한다. put off -> put the meeting off"""
    assert contains_target(example, "put off")


def test_missing_target_is_detected() -> None:
    assert not contains_target("Thanks for your quick reply.", "prompt")


@pytest.mark.parametrize(
    ("example", "target"),
    [
        ("I sorted them into a category.", "cat"),
        ("The management was slow.", "man"),
        ("He wore a nice hat today.", "at"),
    ],
)
def test_substring_does_not_count(example: str, target: str) -> None:
    """부분 문자열 오탐 방지 - cat 이 category 안에 있다고 통과시키면 안 된다."""
    assert not contains_target(example, target)


def test_phrasal_verb_requires_all_parts() -> None:
    assert not contains_target("Let's put the book on the table.", "put off")


# --- 길이 ---


@pytest.mark.parametrize(
    ("example", "expected"),
    [
        ("One two three four five six.", 6),
        ("A B C D E F G H I J K L M N.", 14),
        ("Short one.", 2),
    ],
)
def test_word_count(example: str, expected: int) -> None:
    assert word_count(example) == expected


def test_word_count_ignores_extra_whitespace() -> None:
    assert word_count("  Thanks   for  your  prompt   reply  now  ") == 6


@pytest.mark.parametrize("n", [6, 7, 10, 13, 14])
def test_length_boundary_inclusive(n: int) -> None:
    """6 과 14 는 통과한다 - 경계 포함."""
    assert length_ok(" ".join(["word"] * n))


@pytest.mark.parametrize("n", [1, 5, 15, 30])
def test_length_outside_range_fails(n: int) -> None:
    assert not length_ok(" ".join(["word"] * n))


# --- avoid 중복 ---


def test_identical_sentence_is_duplicate() -> None:
    s = "Thanks for your prompt reply to my message."
    assert duplicates_any(s, [s])


@pytest.mark.parametrize(
    "variant",
    [
        "thanks for your prompt reply to my message.",  # 대소문자
        "Thanks for your prompt reply to my message",  # 구두점 없음
        "Thanks  for your   prompt reply to my message.",  # 공백
        "Thanks, for your prompt reply to my message!",  # 구두점 차이
    ],
)
def test_punctuation_and_case_variants_are_duplicates(variant: str) -> None:
    """재생성했는데 마침표만 다른 같은 문장이 나오면 사용자는 고장으로 여긴다."""
    original = "Thanks for your prompt reply to my message."
    assert duplicates_any(variant, [original])


def test_different_sentence_is_not_duplicate() -> None:
    assert not duplicates_any(
        "The store gave us a prompt refund.",
        ["Thanks for your prompt reply to my message."],
    )


def test_empty_avoid_skips_the_check() -> None:
    assert not duplicates_any("Anything at all.", [])


def test_duplicate_against_any_entry() -> None:
    s = "The store gave us a prompt refund."
    assert duplicates_any(s, ["Something else.", s, "Another one."])


# --- 한글 ---


@pytest.mark.parametrize("text", ["제 메시지에 빠르게 답해 주셔서 고마워요.", "빠른 답변 감사합니다"])
def test_korean_is_detected(text: str) -> None:
    assert has_korean(text)


@pytest.mark.parametrize("text", ["Thanks for the reply.", "12345", "!!!", ""])
def test_non_korean_is_detected(text: str) -> None:
    assert not has_korean(text)


def test_mixed_text_counts_as_korean() -> None:
    assert has_korean("prompt 는 신속한 이라는 뜻입니다")


# --- 통합: violations() ---


_GOOD = "Thanks for your prompt reply to my message."
_GOOD_KO = "제 메시지에 빠르게 답해 주셔서 고마워요."


def test_no_violations_for_good_output() -> None:
    assert violations(_GOOD, _GOOD_KO, _payload()) == []


def test_missing_target_is_reported() -> None:
    got = violations("Thanks for your quick reply to my message.", _GOOD_KO, _payload())
    assert len(got) == 1
    assert "prompt" in got[0]


def test_too_short_is_reported() -> None:
    got = violations("Prompt reply.", _GOOD_KO, _payload())
    assert any("6" in v or "길이" in v for v in got)


def test_duplicate_is_reported() -> None:
    got = violations(_GOOD, _GOOD_KO, _payload(avoid=[_GOOD]))
    assert any("avoid" in v or "중복" in v for v in got)


def test_missing_korean_is_reported() -> None:
    got = violations(_GOOD, "Thanks for your reply.", _payload())
    assert any("한글" in v for v in got)


def test_multiple_violations_are_all_reported() -> None:
    """어느 기준을 어겼는지 전부 알 수 있어야 프롬프트를 고칠 근거가 된다."""
    got = violations("Quick.", "Thanks.", _payload())
    assert len(got) >= 3  # 대상 단어 없음 + 길이 + 한글 없음


def test_violation_messages_are_human_readable() -> None:
    got = violations("Quick.", "Thanks.", _payload())
    assert all(isinstance(v, str) and v.strip() for v in got)
