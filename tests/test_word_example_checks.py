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


# --- 표제어 표기 정규화 (refs #27) -------------------------------------------
#
# 단어장 표기는 사람이 읽는 것이지 매칭 사양이 아니다. 아래 셋은 전부 BBC 정상
# 표제어인데 예문 생성이 **항상** 실패했다 - 검증기가 원문 표기를 글자 그대로
# 요구했기 때문이다.
#
# 이 절의 테스트는 판정을 **느슨하게** 만드는 변경을 지킨다. 그래서 통과 케이스와
# 같은 수의 거부 케이스를 함께 둔다 - 통과만 늘리면 이 검사는 없는 것과 같다.


@pytest.mark.parametrize(
    "example",
    [
        "I finally paid off my debt.",
        "She paid back the loan last month.",
        "He cleared his debt in two years.",
    ],
)
def test_phrase_slash_accepts_any_single_variant(example: str) -> None:
    """`A / B / C` 는 셋 중 **하나만** 쓰이면 된다. 한 문장에 셋을 다 넣을 수 없다."""
    assert contains_target(example, "pay back / pay off / clear (debt)")


@pytest.mark.parametrize(
    "example",
    ["She set a good example for us all.", "He set a bad example yesterday."],
)
def test_word_slash_accepts_either_word(example: str) -> None:
    """공백 없는 `good/bad` 는 낱말 단위 대안이다. 나머지 낱말은 살아 있어야 한다."""
    assert contains_target(example, "set a good/bad example")


def test_parenthesised_object_is_optional() -> None:
    """괄호 안은 **전형적인 목적어를 알려주는 주석**이지 필수 낱말이 아니다.

    `clear (debt)` 에서 `debt` 를 요구하면 "She cleared her student loan." 이 떨어진다.
    자리표시자 제거만으로는 이 케이스가 안 잡힌다 - 괄호 안이 자리표시자가 아니기 때문.
    """
    assert contains_target(
        "She finally cleared her student loan.", "pay back / pay off / clear (debt)"
    )


def test_placeholder_is_replaced_in_a_real_sentence() -> None:
    """`someone`/`something` 은 예문에서 실제 대상으로 바뀐다."""
    assert contains_target("She set her brother up for success.", "set someone up for something")


@pytest.mark.parametrize(
    "example",
    ["He put me in my place.", "They put the new intern in her place."],
)
def test_possessive_moves_with_the_placeholder(example: str) -> None:
    """소유격은 자리표시자와 **함께** 바뀐다: their -> my / her.

    이걸 놓치면 괄호만 지워도 #120 은 계속 떨어진다.
    """
    assert contains_target(example, "put (someone) in their place")


def test_bare_possessive_headword_is_still_exact() -> None:
    """유연한 소유격은 **구·관용구일 때만**이다.

    표제어가 `their` 한 낱말이면 그건 그 단어를 배우는 것이므로 그대로 요구한다.
    """
    assert contains_target("Their house is very old.", "their")
    assert not contains_target("My house is very old.", "their")


def test_prefix_headword_matches_a_hyphenated_word() -> None:
    """접두사 표제어는 붙어 나온다: `pro-` -> `pro-recycling`."""
    assert contains_target("The pro-recycling campaign worked well.", "pro- (something)")
    assert not contains_target("The campaign worked well last year.", "pro- (something)")


def test_headword_that_is_itself_a_placeholder_falls_back_to_the_original() -> None:
    """정규화가 표제어를 통째로 지우면 원문으로 되돌린다.

    되돌리지 않으면 후보가 비어 **무조건 실패**가 되고, 재생성만 반복하다 run 이 죽는다.
    """
    assert contains_target("I need something to eat right now.", "something")
    assert not contains_target("The weather is nice today outside.", "something")


@pytest.mark.parametrize(
    "english",
    [
        "pay back / pay off / clear (debt)",
        "set someone up for something",
        "put (someone) in their place",
    ],
)
def test_normalization_still_rejects_an_unrelated_sentence(english: str) -> None:
    """⚠️ 이 절의 핵심 가드.

    정규화는 판정을 느슨하게 만든다. 대상 단어가 **정말로 안 들어간** 문장까지
    통과하기 시작하면 `contains_target` 은 아무것도 안 지키는 함수가 된다.
    """
    assert not contains_target("The weather is nice today outside.", english)
