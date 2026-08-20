"""FastAPI 서빙 - langgraph-api 를 대체한다 (refs #22).

**와이어 형식은 langgraph 서버와 동일하다** - 호출자(sb-be)는 아무것도 고치지 않는다.

    POST /runs/wait {"assistant_id": "hub", "input": {봉투}}
    GET  /healthz

D22: `langchain/langgraph-api` 는 self-hosted 에 Enterprise 라이선스가 필요해
기동되지 않는다. 허브는 무상태라 체크포인팅·스레드·큐도 안 쓴다 - 직접 서빙한다.

## 응답 형태 (2026-08-20 현행 실측)

- 성공: 그래프 최종 상태 **전체** (허브 경유는 봉투 + output. usage 는 그래프가 제거)
- 직통(word-example): 최종 상태 그대로 - **usage 포함** (탈출구이지 계약이 아니다, D3)
- 미등록 assistant: 422 + detail
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from hub.graph import graph as hub_graph
from word_example.graph import graph as word_example_graph

# env 주입은 앱이 하지 않는다. word_example 이 import 시점에 env 를 읽으므로
# 프로세스 시작 전에 서 있어야 한다:
#   로컬:  uvicorn serve.app:app --env-file .env   (uvicorn 이 import 전에 로드)
#   운영:  k8s envFrom (컨테이너 시작 시 주입)
# 앱 안에서 load_dotenv() 를 부르면 pytest 프로세스까지 .env 로 오염된다.

# assistant 명단. langgraph.json 의 graphs 키 집합과 일치해야 한다 - 테스트가 강제한다.
# 동적 경로 import 를 쓰지 않는 이유: Docker 에서 조용히 깨지는 부류다 (D10).
# 표기 주의: assistant 는 hyphen(`word-example`), 봉투의 task 는 underscore(`word_example`).
ASSISTANTS: dict[str, Any] = {
    "hub": hub_graph,
    "word-example": word_example_graph,
}

app = FastAPI()


class RunRequest(BaseModel):
    """`/runs/wait` 요청. langgraph 서버 스키마에서 허브가 쓰는 부분만.

    `input` 이 optional 인 것이 계약이다 - required 로 두면 누락 시 FastAPI 가
    422 를 던지는데, 현행 서버는 200 + `__error__` EmptyInputError 를 준다.
    input 은 검증하지 않고 그래프에 넘겨 같은 예외가 자연 발생하게 한다 (#24).
    """

    assistant_id: str
    input: Any = None


@app.get("/healthz")
def healthz() -> dict:
    """k8s 프로브용. LLM 을 부르지 않는다.

    그래프 로드는 import 시점이다 - 로드가 깨지면 앱 자체가 안 뜨고
    startup probe 가 잡는다. 여기서 다시 확인할 것이 없다.
    """
    return {"ok": True}


@app.post("/runs/wait")
async def runs_wait(req: RunRequest) -> Any:
    """봉투 하나를 그래프 하나로 돌린다 - 동기 응답만 (D6)."""
    graph = ASSISTANTS.get(req.assistant_id)
    if graph is None:
        registered = ", ".join(sorted(ASSISTANTS))
        # 현행 langgraph 서버의 422 detail 형태를 미러한다 (2026-08-20 실측)
        return JSONResponse(
            status_code=422,
            content={
                "detail": (
                    f"Invalid assistant: '{req.assistant_id}'. Must be either:\n"
                    f"- A valid assistant UUID, or\n"
                    f"- One of the registered graphs: {registered}"
                )
            },
        )
    # 동기 그래프를 스레드로 - LLM 대기가 이벤트 루프를 막지 않게.
    # 타임아웃 집행과 __error__ 어댑터는 #24 에서 얹는다.
    return await asyncio.to_thread(graph.invoke, req.input)
