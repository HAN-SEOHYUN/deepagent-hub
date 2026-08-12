# 호출자 연동 가이드

다른 서버(예: `sb-be`)에서 deepagent-hub 를 부르는 방법.
기계가 읽을 명세는 [`api.json`](api.json), 이 문서는 **연동하는 사람이 읽는 것**이다.

---

## ⚠️ 먼저 알아야 할 것 — 실패해도 HTTP 200 이 온다

LangGraph 서버는 그래프 안에서 난 예외를 **HTTP 200 + `__error__` 본문**으로 돌려준다.

```
성공          HTTP 200  {"output": {...}}
그래프 실패    HTTP 200  {"__error__": {"error": "...", "message": "..."}}
그래프 이름 오타 HTTP 422  {"detail": "Invalid assistant: ..."}
```

**`raise_for_status()` 만으로는 실패를 못 잡는다.** 반드시 `__error__` 키를 확인한다.
이걸 놓치면 실패 응답이 성공으로 처리되어 잘못된 데이터가 저장된다.

---

## 1. 로컬에서 둘 다 띄우기

### 허브

```bash
cd ~/Documents/proj/deepagent-hub
uv run langgraph dev --no-browser        # http://localhost:2024
```

기동 확인:

```bash
curl -sf localhost:2024/ok && echo " 허브 살아있음"
```

### 포트

| | 포트 |
|---|---|
| 허브 (`langgraph dev`) | **2024** |
| 허브 (`langgraph up`, 도커) | 8123 |
| sb-be | 8000 (기본) |

겹치지 않는다. 호출자 `.env` 에 허브 주소를 둔다.

```
DEEPAGENT_HUB_URL=http://localhost:2024
```

> 서버 배포 후에는 `http://<타일넷_IP>:8123` 으로 바꾼다. **주소를 코드에 박지 않는다.**

---

## 2. 엔드포인트

```
POST {DEEPAGENT_HUB_URL}/runs/wait
Content-Type: application/json
```

동기 호출 하나뿐이다. 비동기·스트리밍은 지원하지 않는다.

### 요청

```json
{
  "assistant_id": "hub",
  "input": {
    "service": "sb",
    "task": "word_example",
    "request_id": "sb-batch-20260812-1234",
    "payload": {
      "english": "prompt",
      "meaning": "신속한, 즉각적인",
      "part_of_speech": "adjective",
      "avoid": []
    }
  }
}
```

| 봉투 필드 | 필수 | 뜻 |
|---|---|---|
| `service` | ✅ | **누가** 부르는가. `"sb"` 로 고정 |
| `task` | ✅ | **무엇을** 시키는가. `"word_example"` |
| `request_id` | ✅ | 호출자가 만드는 추적 ID. 로그 대조용 |
| `payload` | ✅ | task 별 입력 |

`payload` (word_example):

| 필드 | 필수 | 뜻 |
|---|---|---|
| `english` | ✅ | 대상 단어·구문. 원문 그대로 |
| `meaning` | ✅ | 한글 뜻. **없으면 품사를 알 수 없다** |
| `part_of_speech` | ❌ | 품사 힌트 |
| `avoid` | ❌ | 피할 문장들. 자동 생성은 `[]`, 재생성에서 채운다 |

### 응답

**응답은 봉투가 그대로 돌아오고 `output` 이 붙는 형태다.**

```json
{
  "service": "sb",
  "task": "word_example",
  "request_id": "sb-batch-20260812-1234",
  "payload": { "...": "요청 그대로" },
  "output": {
    "english": "prompt",
    "example": "Thank you for your prompt reply to my email.",
    "example_ko": "이메일에 빨리 답해 주셔서 고마워요.",
    "model": "openai:gpt-4.1-mini-2025-04-14"
  }
}
```

⚠️ **결과는 `output` 안에 있다.** 최상위가 아니다.

---

## 3. curl 로 먼저 확인

```bash
curl -s localhost:2024/runs/wait -H 'content-type: application/json' -d '{
  "assistant_id": "hub",
  "input": {
    "service": "sb", "task": "word_example", "request_id": "manual-1",
    "payload": {"english": "prompt", "meaning": "신속한, 즉각적인"}
  }
}' | jq .output
```

연결만 확인 (LLM 미호출, 토큰 0):

```bash
curl -s localhost:2024/runs/wait -H 'content-type: application/json' \
  -d '{"assistant_id":"hub","input":{"service":"sb","task":"ping","request_id":"probe","payload":{}}}' \
  | jq .output
```

---

## 4. Python 클라이언트 (sb-be 용)

```python
"""deepagent-hub 클라이언트.

허브는 실패해도 HTTP 200 을 준다 - __error__ 키를 반드시 확인한다.
"""

from __future__ import annotations

import os
import httpx

HUB_URL = os.getenv("DEEPAGENT_HUB_URL", "http://localhost:2024")
SERVICE = "sb"

# 허브의 word_example 타임아웃이 45s 다. 그보다 길게 잡아야
# 허브가 먼저 끊고 명확한 에러를 준다.
TIMEOUT_S = 50.0


class HubError(RuntimeError):
    """허브가 생성에 실패했다. 호출자는 이 항목만 건너뛴다."""


async def generate_word_example(
    *, word_id: int, english: str, meaning: str,
    part_of_speech: str | None = None, avoid: list[str] | None = None,
) -> dict:
    body = {
        "assistant_id": "hub",
        "input": {
            "service": SERVICE,
            "task": "word_example",
            "request_id": f"sb-{word_id}",
            "payload": {
                "english": english,
                "meaning": meaning,
                **({"part_of_speech": part_of_speech} if part_of_speech else {}),
                "avoid": avoid or [],
            },
        },
    }

    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        r = await client.post(f"{HUB_URL}/runs/wait", json=body)

    # 422 등 - 그래프 이름 오타, 서버 설정 문제
    r.raise_for_status()

    data = r.json()

    # ⚠️ 실패해도 200 이 온다. 이 확인을 빼먹으면 안 된다.
    if "__error__" in data:
        err = data["__error__"]
        raise HubError(f"{err.get('error')}: {err.get('message', '')[:200]}")

    out = data.get("output")
    if not out:
        raise HubError("output 이 비었다")

    # ⚠️ 응답이 요청과 다른 단어면 저장하지 않는다.
    # 병렬 호출에서 응답이 뒤섞이면 조용히 잘못된 데이터가 쌓인다.
    if out.get("english") != english:
        raise HubError(f"단어 불일치: 요청={english!r} 응답={out.get('english')!r}")

    return out
```

### 배치에서 쓸 때

```python
import asyncio

# 허브는 배치(다건) 요청을 지원하지 않는다. 단건을 병렬로 돌린다.
# 하루 11건 규모에서는 3~5 동시면 배치와 시간 차이가 없다.
_sem = asyncio.Semaphore(5)


async def _one(word) -> None:
    async with _sem:
        try:
            out = await generate_word_example(
                word_id=word.id, english=word.english, meaning=word.meaning
            )
        except Exception as e:
            # ⚠️ 단어별 실패 격리. 하나 실패해도 수집 잡이 죽으면 안 된다.
            # warm_tts.py 에서 이 격리를 제거했더니 한 단어 실패가
            # 수집 잡 전체를 RuntimeError 로 죽였다 (뮤테이션 검증).
            log.warning("예문 생성 실패 word_id=%s: %s", word.id, e)
            return
        save_example(word.id, out)  # 저장은 호출자의 몫


async def generate_for_new_words(words) -> None:
    await asyncio.gather(*(_one(w) for w in words))
```

---

## 5. 실패 처리

| 상황 | HTTP | 본문 | LLM 호출 | 호출자 대응 |
|---|---|---|---|---|
| 정상 | 200 | `output` | — | 저장 |
| 봉투 필드 누락 | 200 | `__error__` ValidationError | **없음** | **호출자 버그.** 코드를 고친다 |
| 미등록 `task` | 200 | `__error__` ValueError | **없음** | **호출자 버그** |
| `meaning` 누락 | 200 | `__error__` ValidationError | **없음** | **호출자 버그** |
| 품질 기준 위반 (재생성 후에도) | 200 | `__error__` ValueError | 있음 | 그 단어만 건너뜀 |
| LLM 타임아웃·rate limit | 200 | `__error__` | 있음 | 그 단어만 건너뜀 |
| 그래프 이름 오타 | **422** | `detail` | — | 설정 오류 |
| 허브 다운 · 네트워크 | — | 연결 예외 | — | **로그만 남기고 잡은 성공으로 끝냄** |

### 두 가지 원칙

**허브가 죽어 있어도 수집은 성공해야 한다.** 예문은 부가 정보이지 필수가 아니다.

**재생성 실패가 기존 예문을 날리면 안 된다.** 실패하면 아무것도 안 바꾼다.

### 회복은 멱등성이 대신한다

실패한 단어는 예문이 `NULL` 로 남고, **다음 배치가 「예문 없는 단어」를 다시 시도**한다.
별도 재시도 큐가 필요 없다.

> ⚠️ **백필은 안 하기로 했다.** 「예문 없는 단어」 조건에 기존 단어가 전부 걸리므로
> 날짜 조건을 함께 건다 — `WHERE example IS NULL AND entry_date >= '<연동 시작일>'`

---

## 6. 재생성 (사용자 버튼)

지금 저장된 예문을 `avoid` 에 실어 보낸다. **같은 문장이 다시 나오지 않는다.**

```python
out = await generate_word_example(
    word_id=word.id, english=word.english, meaning=word.meaning,
    avoid=[word.example] if word.example else [],
)
```

구두점·대소문자 차이도 동일로 판정하므로, 마침표만 다른 문장이 돌아오는 일은 없다.

⚠️ 브라우저는 타일넷에 못 들어온다. **FE 는 반드시 BE 를 거쳐 부른다** —
어차피 LLM 키가 브라우저에 노출되면 안 되므로 이것이 맞는 구조다.

---

## 7. 주의할 것 두 가지

### 에러 메시지를 그대로 로그에 남기지 말 것

`payload` 검증 실패 메시지에는 pydantic 이 **입력값 전체**를 싣는다.

```
1 validation error for WordExampleIn
meaning
  Field required [type=missing, input_value={'english': 'prompt'}, input_type=dict]
```

허브 자신의 로그에서는 이 값을 지우지만, **HTTP 응답에는 그대로 온다.**
호출자가 이 메시지를 통째로 로그에 남기면 페이로드가 호출자 로그에 쌓인다.
앞의 클라이언트 예제가 `[:200]` 으로 자르는 이유다.

### 응답의 `english` 를 반드시 대조할 것

병렬 호출에서 응답이 뒤섞이면 **조용히 다른 단어에 저장된다.**
허브가 요청 값을 그대로 에코백하므로 대조만 하면 잡힌다.

---

## 8. 연동이 됐는지 확인하는 법

허브 쪽 호출 기록을 본다.

```bash
cd ~/Documents/proj/deepagent-hub
jq -r 'select(.service=="sb") | [.ts, .status, .task, .latency_ms, (.tokens_in//"-")] | @tsv' logs/calls.jsonl
```

```
status 값
  ok        정상
  rejected  LLM 에 닿기 전 거절 - 토큰 미소모. 대개 호출자 버그다
  error     그래프 실행 중 실패 - 토큰 소모됨
```

**`rejected` 가 보이면 호출자 코드를 먼저 의심한다.** 봉투나 페이로드가 규약과 다르다는 뜻이다.

---

## 관련

- 기계가 읽을 명세: [`api.json`](api.json)
- 계약 실물: `samples/word_example/request.json`, `samples/ping/request.json`
