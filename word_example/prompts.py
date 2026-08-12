"""word-example 프롬프트 (refs #16).

⚠️ **프롬프트는 계약이다.** 공용 서버에서 프롬프트 변경은 남의 서비스를 조용히 깬다.
호출자가 남이면 깨진 걸 알아채기까지 오래 걸린다.

상수로 두고 `tests/test_word_example_prompts.py` 가 바이트 동일성을 지킨다.
의도적으로 바꿀 때는 상수를 새로 만들고(`..._V2`) 테스트도 같이 옮긴다.
**테스트를 지우면서 프롬프트를 고치면 안 된다** - 그 테스트가 유일한 방어선이다.

여기 담긴 것은 스펙 §5 중 **프롬프트 책임분**이다.
셀 수 있는 것(길이·단어 포함·avoid 중복·한글 여부)은 코드가 따로 강제한다.
"""

from __future__ import annotations

from word_example.schema import WordExampleIn

WORD_EXAMPLE_SYSTEM_V1 = """당신은 영어 학습용 예문을 만드는 도우미입니다.
주 사용자층은 장년층이며, 예문이 길거나 어려우면 읽지 않습니다.

주어진 영어 단어와 한글 뜻을 받아, 일상에서 그대로 써먹을 수 있는
영어 문장 하나와 자연스러운 한글 해석을 만드세요.

## 반드시 지킬 것

- 대상 단어를 문장에 반드시 포함하세요. 어형 변화는 허용됩니다 (run -> running, ran).
- 문장 길이는 6~14 단어로 맞추세요.
- 한글 뜻이 가리키는 품사로 쓰세요. 다른 품사로 쓰지 마세요.

## 문장 성격

- 일상 상황을 쓰세요 - 집, 직장, 가게, 약속, 가족, 날씨 같은 장면입니다.
- 뉴스 기사체나 논문체를 쓰지 마세요.
- 대상 단어를 제외하면 쉬운 단어만 쓰세요.
  예문을 읽으려고 또 사전을 찾아야 한다면 실패입니다.
- 고유명사를 쓰지 마세요. 특정 인물, 상표, 지명이 들어가면 문장이 금방 낡습니다.

## 한글 해석

- 번역투를 쓰지 마세요.
  "나는 그 신속한 응답을 받았다"가 아니라 "빠르게 답해 주셔서 고마워요"처럼 씁니다.
- 실제로 한국 사람이 말하듯 옮기세요."""

_AVOID_HEADER = """
## 이미 쓴 문장 (이것들과 겹치지 마세요)

아래 문장과 같거나 비슷한 문장을 만들지 마세요. 다른 장면, 다른 표현으로 쓰세요."""


def system_prompt() -> str:
    """시스템 프롬프트. 지금은 분기가 없다.

    분기가 생기면 공통 구간의 바이트 동일성을 테스트로 지킨다.
    """
    return WORD_EXAMPLE_SYSTEM_V1


def build_user_message(payload: WordExampleIn) -> str:
    """사용자 메시지. `avoid` 가 비어 있으면 회피 지시 자체가 안 나간다."""
    lines = [
        f"단어: {payload.english}",
        f"뜻: {payload.meaning}",
    ]
    if payload.part_of_speech:
        lines.append(f"품사: {payload.part_of_speech}")

    text = "\n".join(lines)
    if payload.avoid:
        text += "\n" + _AVOID_HEADER + "\n" + "\n".join(f"- {s}" for s in payload.avoid)
    return text
