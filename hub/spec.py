"""TaskSpec - 그래프가 자기를 설명하는 descriptor (refs #17).

**registry 와 분리한 이유는 순환 import 다.**
`hub/registry.py` 는 각 그래프 모듈을 import 하고(명단), 그래프 모듈은 `TaskSpec` 이 필요하다.
같은 모듈에 두면 그래프를 먼저 import 할 때 깨진다 -
`langgraph.json` 이 그래프 직통 진입점도 등록하므로 실제로 일어날 수 있는 순서다.

이 모듈은 **아무것도 import 하지 않는다.** 그래야 잎(leaf)으로 남는다.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class TaskSpec:
    """그래프가 자기를 설명하는 descriptor.

    `input_model` 을 그래프가 정의하는 것이 중요하다 - 허브는 "검증하라"만 알고
    "무엇을 검증하는지"는 모른다.
    """

    task: str
    """라우팅 키. 봉투의 `task` 와 일치한다."""

    graph: Any
    """컴파일된 subgraph. `.invoke(payload: dict) -> dict` 를 갖는다."""

    input_model: type[BaseModel]
    """페이로드 스키마. 그래프가 정의한다."""

    timeout_s: float
    """호출자 예산보다 짧게. 호출자가 먼저 끊으면 고아 run 이 토큰만 태운다."""

    description: str
    """task 목록을 사람이 읽을 때. 미등록 오류 메시지에도 쓰인다."""

    postprocess: Callable[[dict, BaseModel], dict] | None = field(default=None)
    """출력 후처리. 검증 가능한 품질 기준을 코드가 강제하는 자리."""
