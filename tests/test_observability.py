"""호출 기록 테스트 (refs #9).

비용 관측은 소급이 안 된다. 1일차에 넣지 않으면
"언제부터 이랬는지 모르겠다"로 시작하는 사고를 겪는다.

두 가지 방어가 핵심이고 둘 다 뮤테이션으로 검증한다:
  1. 기록 실패가 호출을 죽이지 않는다
  2. 페이로드·LLM 출력 원문이 기록에 새지 않는다
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hub.envelope import Envelope
from hub.observability import record

_ENV = Envelope.model_validate(
    {
        "service": "sb",
        "task": "word_example",
        "request_id": "req-1",
        "payload": {"english": "prompt", "meaning": "신속한", "secret_note": "SENSITIVE-VALUE"},
    }
)


def _lines(log: Path) -> list[dict]:
    return [json.loads(ln) for ln in log.read_text(encoding="utf-8").splitlines() if ln.strip()]


# --- status 세 값 ---


def test_success_is_recorded_as_ok(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    with record(_ENV, log_path=log) as rec:
        rec.ok({"example": "x", "model": "openai:gpt-4.1-mini"})
    (entry,) = _lines(log)
    assert entry["status"] == "ok"
    assert entry["service"] == "sb"
    assert entry["task"] == "word_example"
    assert entry["request_id"] == "req-1"
    assert entry["model"] == "openai:gpt-4.1-mini"


def test_reject_is_recorded_as_rejected(tmp_path: Path) -> None:
    """LLM 에 닿기 전 거절. 호출자 버그를 뜻한다."""
    log = tmp_path / "calls.jsonl"
    with record(_ENV, log_path=log) as rec:
        rec.reject("unknown task: nope")
    (entry,) = _lines(log)
    assert entry["status"] == "rejected"
    assert "unknown task" in entry["error"]


def test_exception_is_recorded_as_error(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    with pytest.raises(RuntimeError):
        with record(_ENV, log_path=log):
            raise RuntimeError("boom")
    (entry,) = _lines(log)
    assert entry["status"] == "error"
    assert "RuntimeError" in entry["error"]
    assert "boom" in entry["error"]


def test_exception_is_reraised_not_swallowed(tmp_path: Path) -> None:
    """기록은 관찰이지 처리가 아니다. 예외를 삼키면 호출자가 실패를 모른다."""
    log = tmp_path / "calls.jsonl"
    with pytest.raises(ValueError, match="original"):
        with record(_ENV, log_path=log):
            raise ValueError("original")


# --- envelope 필드 ---


def test_entry_has_required_fields(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    with record(_ENV, log_path=log) as rec:
        rec.ok({})
    (entry,) = _lines(log)
    for key in ("ts", "request_id", "service", "task", "status", "latency_ms"):
        assert key in entry, f"{key} 누락"


def test_latency_is_non_negative_int(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    with record(_ENV, log_path=log) as rec:
        rec.ok({})
    (entry,) = _lines(log)
    assert isinstance(entry["latency_ms"], int)
    assert entry["latency_ms"] >= 0


def test_ts_is_iso8601_with_timezone(tmp_path: Path) -> None:
    from datetime import datetime

    log = tmp_path / "calls.jsonl"
    with record(_ENV, log_path=log) as rec:
        rec.ok({})
    (entry,) = _lines(log)
    parsed = datetime.fromisoformat(entry["ts"])
    assert parsed.tzinfo is not None, "타임존 없는 시각은 나중에 해석이 안 된다"


def test_token_counts_are_extracted(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    with record(_ENV, log_path=log) as rec:
        rec.ok({"usage": {"input_tokens": 210, "output_tokens": 64}})
    (entry,) = _lines(log)
    assert entry["tokens_in"] == 210
    assert entry["tokens_out"] == 64


def test_missing_usage_leaves_tokens_null(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    with record(_ENV, log_path=log) as rec:
        rec.ok({"example": "x"})
    (entry,) = _lines(log)
    assert entry["tokens_in"] is None
    assert entry["tokens_out"] is None


# --- 방어 1: 기록 실패가 호출을 죽이지 않는다 ---


def test_unwritable_log_does_not_break_the_call(tmp_path: Path) -> None:
    """관측이 본업을 죽이면 안 된다."""
    unwritable = tmp_path / "nope"
    unwritable.mkdir(mode=0o500)
    log = unwritable / "calls.jsonl"
    try:
        with record(_ENV, log_path=log) as rec:
            rec.ok({"example": "x"})  # 예외가 새면 안 된다
    finally:
        unwritable.chmod(0o700)


def test_unwritable_log_still_reraises_real_exception(tmp_path: Path) -> None:
    """기록 실패를 삼키되, 원래 예외까지 삼키면 안 된다."""
    unwritable = tmp_path / "nope2"
    unwritable.mkdir(mode=0o500)
    log = unwritable / "calls.jsonl"
    try:
        with pytest.raises(RuntimeError, match="real"):
            with record(_ENV, log_path=log):
                raise RuntimeError("real")
    finally:
        unwritable.chmod(0o700)


def test_log_parent_directory_is_created(tmp_path: Path) -> None:
    log = tmp_path / "deep" / "nested" / "calls.jsonl"
    with record(_ENV, log_path=log) as rec:
        rec.ok({})
    assert log.exists()


# --- 방어 2: 원문 유출 금지 ---


def test_no_payload_leak(tmp_path: Path) -> None:
    """페이로드 원문이 기록에 들어가면 개인정보가 파일에 쌓인다."""
    log = tmp_path / "calls.jsonl"
    with record(_ENV, log_path=log) as rec:
        rec.ok({"example": "Thanks for your prompt reply.", "model": "openai:gpt-4.1-mini"})
    raw = log.read_text(encoding="utf-8")
    assert "SENSITIVE-VALUE" not in raw
    assert "신속한" not in raw
    assert "Thanks for your prompt reply." not in raw


def test_error_message_is_truncated(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    with pytest.raises(RuntimeError):
        with record(_ENV, log_path=log):
            raise RuntimeError("x" * 5000)
    (entry,) = _lines(log)
    assert len(entry["error"]) <= 200


# --- JSONL 형식 ---


def test_each_call_is_one_parseable_line(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    for i in range(5):
        env = Envelope.model_validate(
            {"service": "sb", "task": "ping", "request_id": f"r{i}", "payload": {}}
        )
        with record(env, log_path=log) as rec:
            rec.ok({})
    entries = _lines(log)
    assert len(entries) == 5
    assert [e["request_id"] for e in entries] == [f"r{i}" for i in range(5)]


def test_entries_are_appended_not_overwritten(tmp_path: Path) -> None:
    log = tmp_path / "calls.jsonl"
    with record(_ENV, log_path=log) as rec:
        rec.ok({})
    with record(_ENV, log_path=log) as rec:
        rec.ok({})
    assert len(_lines(log)) == 2


def test_log_is_ascii_safe_json(tmp_path: Path) -> None:
    """로그는 ASCII 만. 이모지가 인코딩 충돌로 그래프 로드를 실패시킨 전례가 있다."""
    log = tmp_path / "calls.jsonl"
    env = Envelope.model_validate(
        {"service": "한글서비스", "task": "ping", "request_id": "r", "payload": {}}
    )
    with record(env, log_path=log) as rec:
        rec.ok({})
    raw = log.read_text(encoding="utf-8")
    raw.encode("ascii")  # 예외가 나면 실패
    assert json.loads(raw.strip())["service"] == "한글서비스"


# --- 기본 경로 ---


def test_default_log_path_is_cwd_independent() -> None:
    """상대경로를 쓰면 실행 디렉터리가 바뀔 때 조용히 다른 곳을 본다."""
    from hub.observability import default_log_path

    p = default_log_path()
    assert p.is_absolute()
    assert p.name == "calls.jsonl"


def test_reject_survives_the_propagating_exception(tmp_path: Path) -> None:
    """거절은 예외로 전파되는 것이 정상이다. 그 예외가 rejected 를 error 로 덮으면 안 된다.

    둘을 가르는 것이 기록의 핵심이다 - rejected 는 호출자 버그이고 토큰을 안 태웠다.
    """
    log = tmp_path / "calls.jsonl"
    with pytest.raises(ValueError):
        with record(_ENV, log_path=log) as rec:
            rec.reject("unknown task: nope")
            raise ValueError("unknown task: nope")
    (entry,) = _lines(log)
    assert entry["status"] == "rejected"
    assert "unknown task" in entry["error"]


def test_ok_survives_a_late_exception(tmp_path: Path) -> None:
    """ok 이후 뒷정리에서 터져도 성공은 성공으로 남는다."""
    log = tmp_path / "calls.jsonl"
    with pytest.raises(RuntimeError):
        with record(_ENV, log_path=log) as rec:
            rec.ok({"model": "openai:gpt-4.1-mini"})
            raise RuntimeError("teardown")
    (entry,) = _lines(log)
    assert entry["status"] == "ok"


# --- stdout 싱크 (refs #23) ---------------------------------------------------


def _env() -> Envelope:
    return Envelope(service="sb", task="ping", request_id="stdout-test", payload={})


def test_record_emits_one_jsonl_line_to_stdout(tmp_path, capsys) -> None:
    """호출 1건 = stdout JSONL 1줄. k8s(Loki) 수집 경로다."""
    log = tmp_path / "calls.jsonl"
    with record(_env(), log_path=log) as rec:
        rec.ok({"pong": True})
    out_lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(out_lines) == 1
    entry = json.loads(out_lines[0])
    assert entry["task"] == _env().task
    assert entry["status"] == "ok"
    # 파일과 같은 내용이어야 한다
    assert out_lines[0] == log.read_text().splitlines()[0]


def test_stdout_line_is_ascii_only(tmp_path, capsys) -> None:
    """비ASCII 는 이스케이프된다 - 인코딩 충돌로 그래프 로드가 깨진 전례(D17)."""
    log = tmp_path / "calls.jsonl"
    env = Envelope(service="sb", task="word_example", request_id="한글-id", payload={})
    with record(env, log_path=log) as rec:
        rec.reject("한글 사유")
    out = capsys.readouterr().out
    assert out.strip()
    assert out.isascii()


def test_broken_stdout_does_not_kill_the_call_and_file_survives(tmp_path, monkeypatch) -> None:
    """stdout 이 막혀도 본업(호출)과 파일 싱크는 산다."""
    log = tmp_path / "calls.jsonl"

    class _Broken:
        def write(self, *a):
            raise OSError("stdout is closed")

        def flush(self):
            raise OSError("stdout is closed")

    monkeypatch.setattr("hub.observability.sys.stdout", _Broken())
    with record(_env(), log_path=log) as rec:
        rec.ok({"pong": True})
    lines = log.read_text().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "ok"
