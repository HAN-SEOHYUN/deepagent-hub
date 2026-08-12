"""봉투 (Envelope) - 호출자가 지켜야 할 유일한 규약 (refs #7).

허브는 봉투만 알고 내용물(payload)은 모른다.
payload 검증은 그래프가 정의한 스키마가 맡는다 - 허브는 "검증하라"만 알고
"무엇을 검증하는지"는 모른다.

    {"service": "sb", "task": "word_example",
     "request_id": "sb-batch-20260812-1234",
     "payload": {...}}

`task` 가 라우팅 키이고 `service` 는 라우팅에 쓰지 않는다.
한 서비스가 여러 task 를 부르기 때문이다(sb 가 예문 생성과 시험 문제 생성을 둘 다 부른다).
`service` 는 기록의 축이고, 나중에 붙을 인증·쿼터의 축이다.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Envelope(BaseModel):
    """모든 호출이 이 모양으로 들어온다.

    - 네 필드 모두 필수
    - 여분 필드는 무시한다 (확장 규칙: 필드 추가는 선택으로만)
    - 검증 후 불변 - dispatch 도중 변조를 막는다
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    service: str = Field(
        min_length=1,
        description="누가 부르는가. 기록·정책의 축. 라우팅에는 쓰지 않는다",
    )
    task: str = Field(
        min_length=1,
        description="무엇을 시키는가. 라우팅 키",
    )
    request_id: str = Field(
        min_length=1,
        description="호출자가 만든 추적 ID. 호출자 로그와 대조하는 데 쓴다",
    )
    payload: dict = Field(
        description="그래프별 입력. 허브는 내용을 모른다",
    )
