"""메인 에이전트 - 봉투를 받아 그래프로 돌린다 (refs #10).

**오케스트레이터에 LLM 이 없다.** 호출자는 프로그램이고 자기가 뭘 원하는지 안다.
여기에 LLM 을 끼우면 요청마다 비용·레이턴시가 붙고, 결정론적이어야 할 곳에 확률이 들어온다.

    POST /runs/wait {"assistant_id": "hub", "input": {봉투}}

## 처리 순서 (이 순서가 계약이다)

    1. 봉투 검증      Envelope
    2. 정책           policy.check      <- 지금은 통과만
    3. 라우팅         registry.get
    4. 페이로드 검증  spec.input_model
    5. 실행           spec.graph.invoke
    6. 후처리         spec.postprocess
    7. 기록           record() 가 finally 로 처리

**1~4 에서 실패하면 LLM 을 부르지 않는다.** 불량 입력이 토큰을 태우고 나서 실패하지 않게 한다.

## 그래프를 추가할 때

`hub/registry.py` 의 명단에 한 줄. 이 파일은 **바뀌지 않는다** -
바뀌어야 한다면 경계가 잘못 그어진 것이다.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from hub.envelope import Envelope
from hub.observability import record
from hub.policy import check
from hub.registry import get, known_tasks

_PLACEHOLDER = "<invalid>"

# 그래프 출력의 이 키는 **기록용**이다. 허브가 가져가고 응답에서 제거한다.
# 호출 계약에 안 넣는 이유: 호출자가 쓸 값이 아니고, 계약에 넣으면 모든 그래프가
# 이 필드를 만들어야 한다. 관측은 허브의 관심사이지 계약의 일부가 아니다.
_USAGE_KEY = "usage"


class HubState(TypedDict, total=False):
    """진입 상태 = 봉투 그대로 + 출력."""

    service: str
    task: str
    request_id: str
    payload: dict
    output: dict


def _safe_reason(prefix: str, e: Exception) -> str:
    """검증 실패 사유를 **입력값 없이** 만든다.

    pydantic 의 `str(ValidationError)` 는 `input_value={...}` 에 입력 전체를 싣는다.
    그대로 기록하면 봉투가 불량일 때 페이로드가 통째로 로그에 남는다.
    필드 위치와 오류 종류만 남기고 값은 버린다.
    """
    errors = getattr(e, "errors", None)
    if not callable(errors):
        return f"{prefix}: {type(e).__name__}"
    try:
        parts = [
            f"{'.'.join(str(x) for x in err.get('loc', ())) or '<root>'}: {err.get('type', '?')}"
            for err in errors()
        ]
    except Exception:  # noqa: BLE001 - 사유 생성 실패가 호출을 죽이면 안 된다
        return f"{prefix}: {type(e).__name__}"
    return f"{prefix}: " + "; ".join(parts)


def _salvage(raw: Any) -> Envelope:
    """봉투가 불량이어도 기록은 남긴다.

    실패한 호출이야말로 봐야 할 것이다. 다만 기록에 쓸 식별자가 없으므로
    건질 수 있는 것만 건지고 나머지는 자리표시자로 채운다.
    """
    src = raw if isinstance(raw, dict) else {}

    def _str(key: str) -> str:
        v = src.get(key)
        return v if isinstance(v, str) and v else _PLACEHOLDER

    return Envelope(
        service=_str("service"),
        task=_str("task"),
        request_id=_str("request_id"),
        payload={},
    )


def dispatch(state: HubState) -> dict:
    """봉투 하나를 그래프 하나로 돌린다."""
    try:
        env = Envelope.model_validate(state)
    except Exception as e:
        # 봉투가 불량이면 record() 에 넘길 Envelope 자체가 없다. 건져서 남기고 그대로 던진다.
        with record(_salvage(state)) as rec:
            rec.reject(_safe_reason("invalid envelope", e))
        raise

    with record(env) as rec:
        check(env)

        spec = get(env.task)
        if spec is None:
            reason = f"unknown task: {env.task!r} (known: {known_tasks()})"
            rec.reject(reason)
            raise ValueError(reason)

        try:
            payload = spec.input_model.model_validate(env.payload)
        except Exception as e:
            rec.reject(_safe_reason(f"invalid payload for {env.task!r}", e))
            raise

        # 여기서부터 LLM 이 돈다.
        output = spec.graph.invoke(payload.model_dump())
        if spec.postprocess is not None:
            output = spec.postprocess(output, payload)

        rec.ok(output)
        return {"output": {k: v for k, v in output.items() if k != _USAGE_KEY}}


graph = (
    StateGraph(HubState).add_node("dispatch", dispatch).add_edge(START, "dispatch").add_edge("dispatch", END)
).compile()
