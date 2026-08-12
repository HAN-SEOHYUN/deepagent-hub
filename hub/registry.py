"""레지스트리 - `task` 를 그래프로 잇는다 (refs #8).

**핵심 원칙: 그래프를 추가할 때 라우터의 로직은 안 고치고 명단만 고친다.**

각 그래프 모듈이 `SPEC` 상수를 노출하고 여기서 명시적으로 import 한다.
pkgutil 자동 탐색은 쓰지 않는다 - Docker 에서 조용히 깨지는 부류이고,
로컬에서 되는데 서버에서 그래프가 사라지면 원인 찾는 데 반나절이 든다.

경계가 무너지는 순간을 알아채는 법: 그래프를 추가하는데 `hub/graph.py` 의
로직을 고쳐야 하면 잘못 그어진 것이다. 아래 튜플에 이름이 한 줄 느는 것은 정상이다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
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


def build_registry(specs: Iterable[TaskSpec]) -> dict[str, TaskSpec]:
    """`TaskSpec` 목록을 `task` -> spec dict 로 만든다.

    구성 오류는 **로드 시점에** 잡는다. 조용히 덮어쓰면 어느 그래프가 도는지 알 수 없다.
    """
    registry: dict[str, TaskSpec] = {}
    for spec in specs:
        if not spec.task:
            raise ValueError("TaskSpec.task 가 빈 문자열이다")
        if spec.task in registry:
            raise ValueError(f"task 이름이 중복됐다: {spec.task!r}")
        registry[spec.task] = spec
    return registry


# 그래프가 늘면 여기 import 한 줄 + 아래 튜플에 이름 한 줄.
# 이것이 "명단"이고, hub/graph.py 의 "로직"은 그대로다.
from hub.builtin import PING  # noqa: E402

REGISTRY: dict[str, TaskSpec] = build_registry(
    (
        PING,
    )
)


def get(task: str, registry: dict[str, TaskSpec] | None = None) -> TaskSpec | None:
    """등록된 spec 을 돌려준다. 없으면 `None`.

    예외를 던지지 않는 이유: 미등록 `task` 를 어떻게 처리할지는 호출측이 정한다
    (허브는 `rejected` 로 기록하고 등록된 목록을 에러에 실어 돌려준다).
    """
    return (REGISTRY if registry is None else registry).get(task)


def known_tasks(registry: dict[str, TaskSpec] | None = None) -> list[str]:
    """등록된 task 이름. 미등록 오류 메시지에 실어 호출자가 바로 고치게 한다."""
    return sorted(REGISTRY if registry is None else registry)
