"""패키징 회귀 테스트 (refs #6).

참조 구현(dp-agent)은 서브패키지 6개 중 1개만 `pyproject.toml` 에 선언해,
나머지가 "레포 루트에서 실행하기 때문에" 우연히 동작하는 상태다.
로컬에서는 CWD 덕에 돌고 Docker 에서만 깨지므로 발견이 늦다.

그래프를 추가하면서 `packages.find` 등재를 빠뜨리는 것이 이 프로젝트에서
가장 재현되기 쉬운 사고다. 여기서 자동으로 잡는다.
"""

from __future__ import annotations

import fnmatch
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 패키지가 아닌 최상위 디렉터리 (선언 대상에서 제외)
_NOT_PACKAGES = {"tests", "docs", "samples", "logs"}


def _declared_patterns() -> list[str]:
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        cfg = tomllib.load(f)
    return cfg["tool"]["setuptools"]["packages"]["find"]["include"]


def _top_level_packages() -> set[str]:
    """디스크에 실재하는 최상위 파이썬 패키지 (`__init__.py` 보유)."""
    return {
        d.name
        for d in REPO_ROOT.iterdir()
        if d.is_dir()
        and not d.name.startswith((".", "_"))
        and d.name not in _NOT_PACKAGES
        and (d / "__init__.py").exists()
    }


def test_hub_is_importable() -> None:
    import hub

    assert hub.__doc__, "hub 패키지에 docstring 이 있어야 한다"


def test_every_package_on_disk_is_declared() -> None:
    """실재하는 패키지가 전부 packages.find 에 잡히는지.

    그래프를 추가하고 pyproject.toml 등재를 잊으면 여기서 실패한다.
    """
    patterns = _declared_patterns()
    undeclared = [
        pkg
        for pkg in sorted(_top_level_packages())
        if not any(fnmatch.fnmatch(pkg, pat) for pat in patterns)
    ]
    assert not undeclared, (
        f"pyproject.toml [tool.setuptools.packages.find] include 에 없는 패키지: {undeclared}\n"
        f"현재 include={patterns} - 로컬은 CWD 덕에 돌지만 Docker 에서 깨진다."
    )


def test_dependencies_all_have_upper_bound() -> None:
    """상한 없는 의존성은 마이너 버전이 통째로 떠올라도 아무도 모른다."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as f:
        cfg = tomllib.load(f)
    deps = cfg["project"]["dependencies"]
    unbounded = [d for d in deps if "<" not in d]
    assert not unbounded, f"상한이 없는 의존성: {unbounded}"


def test_env_example_has_no_values() -> None:
    """.env.example 에는 키 이름만 담는다. 값이 들어가면 시크릿이 커밋된다."""
    lines = (REPO_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    with_values = [
        ln for ln in lines if "=" in ln and not ln.lstrip().startswith("#") and ln.split("=", 1)[1]
    ]
    assert not with_values, f".env.example 에 값이 있는 줄: {with_values}"


def test_langgraph_json_is_consistent() -> None:
    """langgraph.json 이 유효하고 python_version 이 pyproject 와 맞는지."""
    import json

    cfg = json.loads((REPO_ROOT / "langgraph.json").read_text(encoding="utf-8"))
    assert cfg["dependencies"] == ["."]
    assert cfg["env"] == ".env"
    assert cfg["python_version"] == "3.13"
    assert "graphs" in cfg, "graphs 키는 비어 있어도 존재해야 한다"


def test_graph_modules_import_in_any_order() -> None:
    """순환 import 회귀 방지 (refs #17).

    hub/registry.py 는 각 그래프 모듈을 import 하고(명단), 그래프 모듈은 TaskSpec 이 필요하다.
    TaskSpec 을 registry 에 두면 그래프를 먼저 import 할 때 깨진다 -
    langgraph.json 이 그래프 직통 진입점(`word-example`)도 등록하므로 실제로 일어날 수 있다.

    각 진입 모듈을 **깨끗한 인터프리터에서 단독으로** import 해 본다.
    """
    import subprocess
    import sys

    for module in ("hub.graph", "hub.registry", "hub.builtin", "word_example.graph"):
        r = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"{module} 단독 import 실패:\n{r.stderr[-600:]}"


def test_spec_module_is_a_leaf() -> None:
    """hub/spec.py 는 프로젝트 내부 모듈을 import 하지 않는다 - 그래야 순환이 안 생긴다."""
    src = (REPO_ROOT / "hub" / "spec.py").read_text(encoding="utf-8")
    for bad in ("from hub.", "import hub", "from word_example", "import word_example"):
        assert bad not in src, f"hub/spec.py 가 내부 모듈을 import 한다: {bad!r}"
