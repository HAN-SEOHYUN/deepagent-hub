"""레지스트리 테스트 (refs #8).

핵심 원칙: 그래프를 추가할 때 라우터의 *로직* 은 안 고치고 *명단* 만 고친다.
그래서 조회는 예외가 아니라 None 을 돌려주고, 미등록 task 처리는 호출측(graph.py)이 정한다.

실제 그래프나 LLM 을 부르지 않는다 - 더미 TaskSpec 만 쓴다.
"""

from __future__ import annotations

import dataclasses

import pytest
from pydantic import BaseModel

from hub.registry import TaskSpec, build_registry, get, known_tasks


class _DummyIn(BaseModel):
    value: str


class _DummyGraph:
    """Runnable 흉내 - invoke 만 있으면 된다."""

    def invoke(self, payload: dict) -> dict:
        return {"echo": payload}


def _spec(task: str = "dummy", **kw) -> TaskSpec:
    base = {
        "task": task,
        "graph": _DummyGraph(),
        "input_model": _DummyIn,
        "timeout_s": 45.0,
        "description": f"{task} 테스트용",
    }
    return TaskSpec(**{**base, **kw})


# --- TaskSpec 자체 ---


def test_taskspec_has_six_fields() -> None:
    names = {f.name for f in dataclasses.fields(TaskSpec)}
    assert names == {"task", "graph", "input_model", "timeout_s", "description", "postprocess"}


def test_postprocess_defaults_to_none() -> None:
    assert _spec().postprocess is None


def test_postprocess_can_be_supplied() -> None:
    def hook(output: dict, payload: BaseModel) -> dict:
        return output

    assert _spec(postprocess=hook).postprocess is hook


def test_taskspec_is_immutable() -> None:
    spec = _spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.task = "other"  # type: ignore[misc]


# --- 조회 ---


def test_get_returns_spec_for_registered_task() -> None:
    reg = build_registry([_spec("alpha")])
    assert get("alpha", reg).task == "alpha"


def test_get_returns_none_for_unknown_task() -> None:
    """예외가 아니라 None. 미등록 처리는 호출측이 정한다."""
    reg = build_registry([_spec("alpha")])
    assert get("nope", reg) is None


def test_known_tasks_lists_registered_names() -> None:
    reg = build_registry([_spec("alpha"), _spec("beta")])
    assert known_tasks(reg) == ["alpha", "beta"]


def test_known_tasks_is_sorted() -> None:
    """에러 메시지에 실리므로 순서가 안정적이어야 한다."""
    reg = build_registry([_spec("zulu"), _spec("alpha"), _spec("mike")])
    assert known_tasks(reg) == ["alpha", "mike", "zulu"]


# --- 구성 오류는 로드 시점에 잡는다 ---


def test_duplicate_task_fails_at_build_time() -> None:
    """같은 task 가 둘 등록되면 조용히 덮어쓰지 않고 즉시 실패한다."""
    with pytest.raises(ValueError, match="중복"):
        build_registry([_spec("alpha"), _spec("alpha")])


def test_registry_key_matches_spec_task() -> None:
    reg = build_registry([_spec("alpha"), _spec("beta")])
    assert all(key == spec.task for key, spec in reg.items())


def test_blank_task_name_fails_at_build_time() -> None:
    with pytest.raises(ValueError, match="빈"):
        build_registry([_spec("")])


def test_empty_registry_is_valid() -> None:
    """그래프가 하나도 없어도 레지스트리는 성립한다."""
    reg = build_registry([])
    assert reg == {}
    assert known_tasks(reg) == []
    assert get("anything", reg) is None


# --- 기본 REGISTRY ---


def test_module_level_registry_exists() -> None:
    from hub import registry

    assert isinstance(registry.REGISTRY, dict)


def test_get_falls_back_to_module_registry() -> None:
    """registry 인자를 생략하면 모듈 기본 REGISTRY 를 본다."""
    from hub import registry

    for task in registry.REGISTRY:
        assert get(task) is registry.REGISTRY[task]
