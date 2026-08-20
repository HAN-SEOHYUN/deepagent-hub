# deepagent-hub

여러 서비스가 함께 쓰는 **무상태 AI 생성 서버**.
호출자는 "무엇을 만들어 달라"만 보내고, 허브는 만들어서 돌려준다.

**허브는 DB 를 건드리지 않는다** — 생성만 한다. 저장은 호출자 몫이다.

## 호출

```
POST /runs/wait
{
  "assistant_id": "hub",
  "input": {
    "service": "sb",
    "task": "ping",
    "request_id": "call-0001",
    "payload": {}
  }
}
```

| 봉투 필드 | 뜻 |
|---|---|
| `service` | **누가** 부르는가 — 기록·정책용 (라우팅에 쓰지 않는다) |
| `task` | **무엇을** 시키는가 — 라우팅 키 |
| `request_id` | 호출자가 만든 추적 ID |
| `payload` | 그래프별 입력 — **허브는 내용을 모른다** |

## 개발

```bash
uv sync --all-extras        # langgraph dev 는 dev extra 에 있다 (운영 이미지 ELv2 배제, #25)
cp .env.example .env        # 값은 직접 채운다
uv run langgraph dev        # 포트 2024
```

운영과 같은 서빙(FastAPI)으로 띄우려면:

```bash
uv run uvicorn serve.app:app --port 8000 --env-file .env
```

```bash
uv run pytest
uv run ruff check .
```

## 구조

```
hub/          진입·라우팅·정책·기록
<graph>/      얹히는 그래프 (형제 패키지)
samples/      호출 계약 실물
tests/
```

그래프를 추가할 때는 `hub/registry.py` 명단, `langgraph.json`,
`pyproject.toml` 의 `packages.find`, `.env.example` 을 함께 갱신한다.
