"""word-example 입출력 계약 (refs #15).

허브는 이 스키마의 내용을 모른다. `TaskSpec.input_model` 로 넘겨주면
허브가 "검증하라"만 실행한다.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class WordExampleIn(BaseModel):
    """입력.

    `meaning` 이 필수인 이유: `prompt` 는 「신속한」(형용사)이자 「자극하다」(동사)이자
    「프롬프트」(명사)다. 뜻 없이는 어느 쪽인지 알 수 없고, 단어장에 이미 있으므로 그대로 받는다.
    """

    model_config = ConfigDict(extra="ignore")

    english: str = Field(min_length=1, description="대상 단어·구문. 원문 그대로")
    meaning: str = Field(min_length=1, description="한글 뜻. 품사를 가르는 근거")
    part_of_speech: str | None = Field(default=None, description="품사 힌트. 없으면 meaning 으로 추론")
    avoid: list[str] = Field(
        default_factory=list,
        description="이미 있는 예문들. 재생성에서 같은 문장이 나오면 사용자는 고장으로 여긴다",
    )

    @field_validator("english", "meaning")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("공백만 있는 값은 없는 것과 같다")
        return v

    @field_validator("avoid")
    @classmethod
    def _drop_blank(cls, v: list[str]) -> list[str]:
        # 빈 문자열은 회피 대상이 아니다. 프롬프트에 실리면 지시만 흐려진다.
        return [s for s in v if s.strip()]


class WordExampleOut(BaseModel):
    """출력.

    `english` 를 에코백하는 이유: 호출자가 병렬로 던졌을 때 응답이 뒤섞이면
    조용히 다른 단어에 저장된다. 호출자는 불일치를 감지하면 저장하지 않는다.

    `model` 을 돌려주는 이유: 모델은 `.env` 로 바뀐다. 호출자가 화면 문구에
    박아두면 모델을 바꾼 순간 그 문구가 거짓말이 된다.
    """

    model_config = ConfigDict(extra="ignore")

    english: str = Field(min_length=1, description="요청한 단어 그대로 (에코백)")
    example: str = Field(min_length=1, description="영어 예문 한 문장")
    example_ko: str = Field(min_length=1, description="자연스러운 한글")
    model: str = Field(min_length=1, description="실제로 쓴 모델")

    @field_validator("example", "example_ko")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        # 반쯤 성공한 응답을 만들지 않는다 - 그 쓰레기가 호출자 DB 에 저장된다.
        if not v.strip():
            raise ValueError("빈 예문은 예문이 아니다")
        return v
