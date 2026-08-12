"""내장 task - 그래프 없이도 왕복을 확인한다 (refs #10).

`ping` 은 LLM 을 부르지 않는다. 배포 헬스체크와 개발 중 연결 확인에 쓴다.
이것이 성공하면 그래프 로드·봉투 검증·라우팅·기록이 전부 살아 있다는 뜻이다.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict
from typing_extensions import TypedDict

from hub.registry import TaskSpec


class PingIn(BaseModel):
    """입력이 없다. 여분 필드는 무시한다."""

    model_config = ConfigDict(extra="ignore")


class PingState(TypedDict, total=False):
    pong: bool


def _pong(state: PingState) -> dict:
    return {"pong": True}


_ping_graph = (
    StateGraph(PingState).add_node("pong", _pong).add_edge(START, "pong").add_edge("pong", END)
).compile()


PING = TaskSpec(
    task="ping",
    graph=_ping_graph,
    input_model=PingIn,
    timeout_s=5.0,
    description="LLM 없이 왕복만 확인한다. 배포 헬스체크용",
)
