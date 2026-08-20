"""word-example 품질 실측 (refs #14).

실물 단어를 돌려 길이 분포·재생성 발생률·토큰·지연을 표로 뽑는다.
「길이 6~14 는 출발점이지 결론이 아니다」 - 여기서 근거를 만든다.

    uv run python scripts/quality_check.py                 # 기본 단어 집합
    uv run python scripts/quality_check.py --words w.tsv   # 파일 (english<TAB>meaning)
    uv run python scripts/quality_check.py --model openai:gpt-5-nano
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# 이슈 #14 시나리오 매트릭스의 8가지 단어 유형을 고르게 담는다.
DEFAULT_WORDS: list[tuple[str, str, str]] = [
    # (english, meaning, 유형)
    ("prompt", "신속한, 즉각적인", "형용사/다의어"),
    ("resilient", "회복력 있는", "형용사"),
    ("hectic", "정신없이 바쁜", "형용사"),
    ("thorough", "철저한", "형용사"),
    ("sturdy", "튼튼한", "형용사"),
    ("bargain", "싼 물건, 특가품", "명사"),
    ("hassle", "귀찮은 일, 번거로움", "명사"),
    ("chore", "허드렛일, 집안일", "명사"),
    ("remedy", "치료법, 해결책", "명사"),
    ("glimpse", "언뜻 봄", "명사"),
    ("linger", "오래 머무르다", "동사"),
    ("thrive", "잘 자라다, 번창하다", "동사"),
    ("postpone", "연기하다", "동사"),
    ("tidy", "정리하다", "동사"),
    ("mend", "고치다, 수선하다", "동사"),
    ("barely", "간신히, 거의 ~않다", "부사"),
    ("eventually", "결국, 마침내", "부사"),
    ("rarely", "좀처럼 ~않는", "부사"),
    ("thoroughly", "철저히", "부사"),
    ("put off", "미루다, 연기하다", "구동사"),
    ("run into", "우연히 마주치다", "구동사"),
    ("look after", "돌보다", "구동사"),
    ("give up", "포기하다", "구동사"),
    ("come across", "우연히 발견하다", "구동사"),
    ("charge", "충전하다", "다의어(동사)"),
    ("book", "예약하다", "다의어(동사)"),
    ("light", "가벼운", "다의어(형용사)"),
    ("resilience", "회복력", "추상명사"),
    ("insight", "통찰력", "추상명사"),
    ("patience", "인내심", "추상명사"),
    ("courtesy", "예의, 호의", "추상명사"),
    ("once in a while", "가끔씩", "관용표현"),
    ("on purpose", "일부러", "관용표현"),
]


def _load(path: str | None) -> list[tuple[str, str, str]]:
    if not path:
        return DEFAULT_WORDS
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        rows.append((parts[0], parts[1], parts[2] if len(parts) > 2 else "-"))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", help="english<TAB>meaning<TAB>유형 파일")
    ap.add_argument("--model", help="WORD_EXAMPLE_MODEL 덮어쓰기")
    ap.add_argument("--out", default="/tmp/quality.tsv")
    args = ap.parse_args()

    if args.model:
        os.environ["WORD_EXAMPLE_MODEL"] = args.model

    # 모델을 바꾼 뒤에 import 해야 한다 (모듈이 로드 시점에 기본값을 읽는다).
    from hub.graph import graph
    from word_example.checks import word_count

    words = _load(args.words)
    rows = []
    failures = []

    for i, (english, meaning, kind) in enumerate(words, 1):
        started = time.monotonic()
        try:
            out = graph.invoke(
                {
                    "service": "quality-check",
                    "task": "word_example",
                    "request_id": f"qc-{i:03d}",
                    "payload": {"english": english, "meaning": meaning},
                }
            )["output"]
            ms = int((time.monotonic() - started) * 1000)
            rows.append(
                {
                    "english": english,
                    "kind": kind,
                    "example": out["example"],
                    "example_ko": out["example_ko"],
                    "words": word_count(out["example"]),
                    "ms": ms,
                }
            )
            print(f"  [{i:2}/{len(words)}] {english:16} {word_count(out['example']):2}w  {out['example']}")
        except Exception as e:  # noqa: BLE001 - 실측이므로 실패도 데이터다
            failures.append({"english": english, "kind": kind, "error": f"{type(e).__name__}: {e}"[:160]})
            print(f"  [{i:2}/{len(words)}] {english:16} FAILED  {type(e).__name__}")

    out_path = Path(args.out)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("english\tkind\twords\tms\texample\texample_ko\n")
        for r in rows:
            f.write(f"{r['english']}\t{r['kind']}\t{r['words']}\t{r['ms']}\t{r['example']}\t{r['example_ko']}\n")

    counts = sorted(r["words"] for r in rows)
    print()
    print(f"  성공 {len(rows)} / 실패 {len(failures)}")
    if counts:
        print(f"  길이: min={counts[0]} max={counts[-1]} 중앙값={counts[len(counts) // 2]}")
        dist: dict[int, int] = {}
        for c in counts:
            dist[c] = dist.get(c, 0) + 1
        print("  분포: " + " ".join(f"{k}w×{v}" for k, v in sorted(dist.items())))
    if failures:
        print("  실패:")
        for fl in failures:
            print(f"    {fl['english']:16} {fl['error']}")
    print(f"  표: {out_path}")
    print(json.dumps({"ok": len(rows), "failed": len(failures)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
