"""호출 기록 - 호출 1건이 JSONL 1줄 (refs #9).

비용 관측은 소급이 안 된다. 인증·쿼터를 미루면서도 이것만은 1일차에 만드는 이유다.
쿼터 상한을 정할 근거가 여기서 나온다.

허브는 DB 를 안 건드리므로(재사용의 전제) 기록도 파일로 간다.

**남기지 않는 것**: 페이로드 원문, LLM 출력 원문.
필요하면 `request_id` 로 호출자 로그를 본다. 한 번 남기기 시작하면
보존 기간·접근 통제가 따라오므로 안 남기는 편에서 시작한다.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from hub.envelope import Envelope

_ERROR_MAX = 200

# 상대경로를 쓰지 않는다 - 실행 디렉터리가 바뀌면 조용히 다른 곳을 본다.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def default_log_path() -> Path:
    return _REPO_ROOT / "logs" / "calls.jsonl"


class Recorder:
    """`record()` 컨텍스트가 넘겨주는 핸들. 결과를 기록 항목에 얹는다."""

    def __init__(self, entry: dict[str, Any]) -> None:
        self._entry = entry
        self._settled = False
        """결과가 명시적으로 정해졌는가.

        거절(`rejected`)은 예외로 전파되는 것이 정상이므로, 그 예외가
        `record()` 의 except 절에서 `error` 로 덮어쓰이면 안 된다.
        `rejected` 와 `error` 를 가르는 것이 기록의 핵심이다 -
        전자는 호출자 버그이고 토큰을 안 태웠다.
        """

    @property
    def settled(self) -> bool:
        return self._settled

    def ok(self, output: dict[str, Any]) -> None:
        """정상 종료. 출력에서 모델·토큰만 뽑는다 (원문은 안 남긴다)."""
        self._settled = True
        self._entry["status"] = "ok"
        model = output.get("model")
        if isinstance(model, str):
            self._entry["model"] = model
        usage = output.get("usage")
        if isinstance(usage, dict):
            self._entry["tokens_in"] = usage.get("input_tokens")
            self._entry["tokens_out"] = usage.get("output_tokens")

    def reject(self, reason: str) -> None:
        """LLM 에 닿기 전 거절. 호출자 버그를 뜻하며 토큰을 안 태웠다."""
        self._settled = True
        self._entry["status"] = "rejected"
        self._entry["error"] = reason[:_ERROR_MAX]


def _append(entry: dict[str, Any], log_path: Path) -> None:
    """기록 실패는 삼킨다 - 관측이 본업을 죽이면 안 된다."""
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # ensure_ascii=True: 로그는 ASCII 만. 이모지·비ASCII 가 인코딩 충돌로
        # 그래프 로드를 실패시킨 전례가 있다(참조 구현 cp949 사고).
        line = json.dumps(entry, ensure_ascii=True)
        with log_path.open("a", encoding="ascii") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001 - 기록 실패로 호출을 죽이지 않는다
        pass


@contextmanager
def record(env: Envelope, log_path: Path | None = None) -> Iterator[Recorder]:
    """호출 1건을 기록한다.

    `finally` 에서 쓴다 - 예외가 나도 기록은 남는다.
    실패한 호출이야말로 봐야 할 것인데, 성공 경로에서만 쓰면 그게 안 남는다.

    지연은 `time.monotonic()` 으로 잰다. 벽시계는 NTP 보정에 뒤로 갈 수 있다.
    """
    path = default_log_path() if log_path is None else log_path
    started = time.monotonic()
    entry: dict[str, Any] = {
        "ts": datetime.now().astimezone().isoformat(),
        "request_id": env.request_id,
        "service": env.service,
        "task": env.task,
        "status": "error",  # 아무도 갱신 안 하면 실패로 본다
        "latency_ms": 0,
        "model": None,
        "tokens_in": None,
        "tokens_out": None,
        "error": None,
    }
    rec = Recorder(entry)
    try:
        yield rec
    except Exception as e:
        # 이미 결과가 정해졌으면(대개 reject) 덮어쓰지 않는다.
        # 거절은 예외로 전파되는 것이 정상이고, 그것을 error 로 바꾸면
        # "토큰을 태웠는가"를 기록으로 못 가른다.
        if not rec.settled:
            entry["status"] = "error"
            entry["error"] = f"{type(e).__name__}: {e}"[:_ERROR_MAX]
        raise
    finally:
        entry["latency_ms"] = int((time.monotonic() - started) * 1000)
        _append(entry, path)
