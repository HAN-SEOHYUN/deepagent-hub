"""word-example 그래프 - 단어와 뜻을 받아 일상 예문을 만든다 (refs #17).

## 모델 교체

`WORD_EXAMPLE_MODEL` env 로 바꾼다. `init_chat_model` 이 `openai:` / `anthropic:`
접두사로 프로바이더까지 바꾸므로 **코드 변경이 없다.**
시작은 `openai:gpt-4.1-mini`, 연산·사고가 무거워지면 `anthropic:claude-sonnet-*` 로 승격.

## LLM 을 지연 생성하는 이유

모듈 로드 시점에 만들면 **API 키 없이는 허브 전체가 못 뜬다** -
`hub/registry.py` 가 이 모듈을 import 하므로 `ping` 조차 안 된다.

## english 는 코드가 채운다

응답의 `english` 는 요청을 그대로 복사한다. LLM 이 만들게 하면
"응답이 요청과 다르면 저장하지 않는다"는 호출자 방어가 무의미해진다.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from hub.spec import TaskSpec
from word_example.checks import violations
from word_example.prompts import build_user_message, system_prompt
from word_example.schema import WordExampleIn, WordExampleOut

_DEFAULT_MODEL = "openai:gpt-4.1-mini"

# 호출자(sb) 예산이 50s 이므로 그보다 짧게 - 호출자가 먼저 끊으면
# 고아 run 이 아무도 안 받아갈 응답을 만들며 토큰을 태운다.
_TIMEOUT_S = 45.0
# 기본 2회는 45s 예산 내 초과 위험.
_MAX_RETRIES = 1
# 품질 위반 시 재생성 횟수. 1회만 - 무한 루프가 나면 토큰이 무한히 탄다.
_MAX_REGENERATIONS = 1

# 테스트가 monkeypatch 로 갈아끼울 수 있게 모듈 변수로 둔다.
_MODEL = None


class Generated(BaseModel):
    """LLM 이 만드는 것. `english` 와 `model` 은 코드가 채운다."""

    example: str = Field(min_length=1, description="영어 예문 한 문장")
    example_ko: str = Field(min_length=1, description="자연스러운 한글 해석")


class WordExampleState(TypedDict, total=False):
    """내부 상태 - 입력 필드가 그대로 흐른다."""

    english: str
    meaning: str
    part_of_speech: str | None
    avoid: list[str]
    example: str
    example_ko: str
    model: str


class WordExampleOutState(TypedDict):
    """출력 스키마.

    분리하지 않으면 입력 필드(meaning, avoid, part_of_speech)가 응답에 섞여
    계약이 깨진다. 호출자는 계약에 없는 필드를 보게 된다.
    """

    english: str
    example: str
    example_ko: str
    model: str


def _default_model() -> str:
    """`os.getenv(x) or 기본값` - `.env` 에 키만 있고 값이 빈 경우까지 미설정으로 취급한다."""
    return os.getenv("WORD_EXAMPLE_MODEL") or _DEFAULT_MODEL


_cached = None


def _structured():
    """구조화 출력 LLM. 첫 호출에 만든다 (키 없이도 import 가 되도록)."""
    global _cached
    if _cached is None:
        llm = init_chat_model(
            _MODEL or _default_model(),
            temperature=0.3,
            timeout=_TIMEOUT_S,
            max_retries=_MAX_RETRIES,
        )
        _cached = llm.with_structured_output(Generated)
    return _cached


def _call(payload: WordExampleIn) -> Generated:
    return _structured().invoke(
        [
            ("system", system_prompt()),
            ("human", build_user_message(payload)),
        ]
    )


def generate(state: WordExampleState) -> dict:
    """프롬프트 조립 -> LLM -> 품질 검증 -> (위반 시) 재생성 1회 -> 응답 조립.

    ⛔ **절삭·보정으로 때우지 않는다.** 잘라낸 문장은 예문이 아니고,
    반쯤 성공한 응답은 호출자 DB 에 저장되어 화면에 뜬다.
    허브는 저장을 안 하므로 되돌릴 수단이 없다.
    """
    payload = WordExampleIn.model_validate(state)
    attempt = payload
    bad: list[str] = []

    for _ in range(_MAX_REGENERATIONS + 1):
        gen = _call(attempt)
        bad = violations(gen.example, gen.example_ko, payload)
        if not bad:
            out = WordExampleOut(
                english=payload.english,  # 에코백은 코드가 보장한다
                example=gen.example,
                example_ko=gen.example_ko,
                model=_MODEL or _default_model(),
            )
            return out.model_dump()
        # 직전 실패작을 회피 목록에 얹어 같은 답을 다시 받지 않는다.
        # 원래 avoid 는 유지한다 - 호출자가 피하라고 준 것이다.
        attempt = attempt.model_copy(update={"avoid": [*attempt.avoid, gen.example]})

    raise ValueError("품질 기준 위반 (재생성 후에도): " + "; ".join(bad))


graph = (
    StateGraph(WordExampleState, output_schema=WordExampleOutState)
    .add_node("generate", generate)
    .add_edge(START, "generate")
    .add_edge("generate", END)
).compile()


SPEC = TaskSpec(
    task="word_example",
    graph=graph,
    input_model=WordExampleIn,
    timeout_s=_TIMEOUT_S,
    description="영어 단어와 뜻을 받아 일상 예문(영어+한글)을 만든다",
)
