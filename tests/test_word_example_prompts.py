"""프롬프트 회귀 테스트 (refs #16).

프롬프트는 계약이다. 공용 서버에서 프롬프트를 고치면 남의 서비스가 조용히 깨진다.
여기서 바이트 동일성을 지킨다.

참조 구현(dp-agent)의 같은 테스트가 이유를 밝힌다:
"다른 서비스가 이미 이 프롬프트로 검증을 마쳤으므로 변경 금지"
"""

from __future__ import annotations

import pytest

from word_example import prompts
from word_example.schema import WordExampleIn

_IN = {"english": "prompt", "meaning": "신속한, 즉각적인"}


def _payload(**kw) -> WordExampleIn:
    return WordExampleIn.model_validate({**_IN, **kw})


# --- 바이트 동일성 ---


def test_system_prompt_is_frozen() -> None:
    """이 테스트가 실패하면 프롬프트를 의도적으로 바꾼 것인지 확인하라.

    의도적이면 상수를 V2 로 새로 만들고 이 테스트도 같이 옮긴다.
    테스트를 지우면서 프롬프트를 고치면 안 된다.
    """
    assert prompts.system_prompt() == prompts.WORD_EXAMPLE_SYSTEM_V1


# --- 스펙 §5 프롬프트 책임분이 실제로 담겼는가 ---


@pytest.mark.parametrize(
    "must_contain",
    [
        "6~14",  # 길이
        "어형 변화",  # run -> running 허용
        "일상",  # 상황
        "쉬운 단어",  # 어휘 수준
        "고유명사",  # 금지
        "번역투",  # 한글 품질
    ],
)
def test_system_prompt_carries_the_spec_rules(must_contain: str) -> None:
    assert must_contain in prompts.WORD_EXAMPLE_SYSTEM_V1


def test_system_prompt_mentions_the_audience() -> None:
    """장년층이라는 전제가 여러 결정을 좌우한다. 프롬프트에도 남긴다."""
    assert "장년층" in prompts.WORD_EXAMPLE_SYSTEM_V1


# --- 사용자 메시지 ---


def test_user_message_carries_word_and_meaning() -> None:
    msg = prompts.build_user_message(_payload())
    assert "prompt" in msg
    assert "신속한, 즉각적인" in msg


def test_part_of_speech_is_included_when_given() -> None:
    msg = prompts.build_user_message(_payload(part_of_speech="adjective"))
    assert "adjective" in msg


def test_part_of_speech_line_is_absent_when_omitted() -> None:
    """없는 값을 빈 줄로 넣으면 지시가 흐려진다."""
    msg = prompts.build_user_message(_payload())
    assert "품사:" not in msg


def test_no_avoid_section_when_avoid_is_empty() -> None:
    """자동 생성 경로에서는 회피 지시가 나가면 안 된다."""
    msg = prompts.build_user_message(_payload())
    assert "겹치지" not in msg


def test_avoid_sentences_are_included_verbatim() -> None:
    """원문 그대로 실어야 LLM 이 무엇을 피할지 안다."""
    sentences = ["Thanks for your prompt reply to my message.", "The store gave us a prompt refund."]
    msg = prompts.build_user_message(_payload(avoid=sentences))
    for s in sentences:
        assert s in msg


def test_avoid_section_appears_only_with_avoid() -> None:
    with_avoid = prompts.build_user_message(_payload(avoid=["x."]))
    without = prompts.build_user_message(_payload())
    assert len(with_avoid) > len(without)
    assert "겹치지" in with_avoid


def test_blank_avoid_entries_do_not_reach_the_prompt() -> None:
    """스키마가 걸러내므로 프롬프트에 빈 항목이 실리지 않는다."""
    msg = prompts.build_user_message(_payload(avoid=["real.", "", "  "]))
    assert msg.count("- ") == 1


def test_user_message_is_stable_for_same_input() -> None:
    """같은 입력에 같은 메시지. 재현 가능해야 디버깅이 된다."""
    assert prompts.build_user_message(_payload()) == prompts.build_user_message(_payload())
