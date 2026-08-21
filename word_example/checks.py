"""품질 검증 - 셀 수 있는 것만 코드가 강제한다 (refs #18).

「자연스럽다」는 검증할 수 없다. 아래 넷으로 바꿔 놓는다.
프롬프트가 맡는 것(일상 상황·어휘 난이도·고유명사·번역투)은 여기서 보지 않는다 -
판정 규칙이 자의적이라 코드로 못 박으면 오탐이 더 해롭다.

전부 순수 함수다. LLM 도 IO 도 없어 테스트가 빠르고 공짜다.
"""

from __future__ import annotations

import itertools
import re

from word_example.schema import WordExampleIn

LENGTH_LO = 6
LENGTH_HI = 14

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_HANGUL_RE = re.compile(r"[가-힣]")

# 표제어에만 나타나는 표기들. 예문에는 그대로 안 나온다 (refs #27).
#
# 단어장 표기는 **사람이 읽는 것**이지 매칭 사양이 아니다. `set someone up for
# something` 의 `someone` 은 예문에서 실명·대명사로 바뀌고, `pay back / pay off /
# clear` 는 셋 중 하나만 쓰인다. 원문을 글자 그대로 요구하면 **정상 예문이 전부
# 떨어져** 재생성만 반복하다 run 이 실패한다.
_ELLIPSIS = re.compile(r"…|\.\.\.")
_PARENS = re.compile(r"\([^)]*\)")
_PHRASE_SLASH = re.compile(r"\s+/\s+")

# 자리표시자 — 예문에서 실제 대상으로 치환된다.
_PLACEHOLDERS = frozenset(
    {"someone", "somebody", "something", "oneself", "one's", "ones", "sb", "sth"}
)

# 소유격 한정사는 자리표시자와 **함께 바뀐다**:
#   `put (someone) in their place` -> "He put me in **my** place."
# 그래서 `their` 를 글자 그대로 요구하면 #120 이 계속 떨어진다.
_POSSESSIVES = frozenset({"my", "your", "his", "her", "its", "our", "their"})

# 한 표제어가 펴지는 변형 수 상한. 슬래시가 여럿 겹쳐도 폭발하지 않게.
_MAX_VARIANTS = 12

# 영어 굴절 접미사. 긴 것부터 벗겨야 running -> runn 이 아니라 run 이 된다.
_SUFFIXES = ("ingly", "ings", "ing", "edly", "ies", "ied", "es", "ed", "s", "'s")

# 불규칙 변화는 어간 추출로 못 잡는다. 흔한 것만 표로 둔다.
#
# 여기 없는 불규칙 동사는 "대상 단어 없음"으로 판정돼 재생성이 일어난다.
# 그것이 안전한 실패다 - 잘못 통과시키는 것(오탐)보다 낫고, 재생성하면
# 대개 규칙형이 나온다. 표는 낭비를 줄일 뿐 정확성의 근거가 아니다.
_IRREGULAR = {
    "ran": "run", "run": "run",
    "went": "go", "gone": "go", "goes": "go",
    "ate": "eat", "eaten": "eat",
    "took": "take", "taken": "take",
    "gave": "give", "given": "give",
    "came": "come",
    "saw": "see", "seen": "see",
    "made": "make",
    "found": "find",
    "told": "tell",
    "kept": "keep",
    "left": "leave",
    "lost": "lose",
    "met": "meet",
    "paid": "pay",
    "put": "put",
    "said": "say",
    "sent": "send",
    "sat": "sit",
    "spent": "spend",
    "stood": "stand",
    "brought": "bring",
    "bought": "buy",
    "caught": "catch",
    "chose": "choose", "chosen": "choose",
    "drove": "drive", "driven": "drive",
    "fell": "fall", "fallen": "fall",
    "felt": "feel",
    "got": "get", "gotten": "get",
    "grew": "grow", "grown": "grow",
    "held": "hold",
    "knew": "know", "known": "know",
    "wrote": "write", "written": "write",
    "began": "begin", "begun": "begin",
    "broke": "break", "broken": "break",
    "spoke": "speak", "spoken": "speak",
    "wore": "wear", "worn": "wear",
    "threw": "throw", "thrown": "throw",
    "thought": "think",
    "understood": "understand",
    "was": "be", "were": "be", "been": "be", "am": "be", "is": "be", "are": "be",
    "had": "have", "has": "have",
    "did": "do", "does": "do", "done": "do",
}


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def word_count(example: str) -> int:
    """단어 수. 여분 공백·구두점에 흔들리지 않는다."""
    return len(_tokens(example))


def length_ok(example: str, lo: int = LENGTH_LO, hi: int = LENGTH_HI) -> bool:
    """경계 포함. 6 단어와 14 단어는 통과한다."""
    return lo <= word_count(example) <= hi


def _stem(word: str) -> str:
    """굴절 접미사를 벗긴 어간 근사값.

    형태소 분석기를 쓰지 않는다 - 의존성을 하나 늘릴 만한 정확도 차이가 아니고,
    여기서 필요한 건 "대상 단어가 쓰였는가"의 판정이지 언어학적 정확성이 아니다.
    """
    w = word.lower()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    for suf in _SUFFIXES:
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            base = w[: -len(suf)]
            # running -> runn -> run (자음 중복 되돌리기)
            if len(base) >= 4 and base[-1] == base[-2] and base[-1] not in "aeiou":
                base = base[:-1]
            return base
    return w


def _stem_variants(word: str) -> set[str]:
    """어간 후보들. 묵음 e 가 사라지는 경우까지 흡수한다 (thriving -> thriv -> thrive)."""
    base = _stem(word)
    return {base, base + "e"}


def _matches(token: str, target: str) -> bool:
    """토큰 하나가 대상 단어와 같은가. 어형 변화와 대소문자를 흡수한다."""
    t, g = token.lower(), target.lower()
    if t == g:
        return True
    # 어간 후보 집합이 겹치면 같은 단어로 본다.
    return bool(_stem_variants(t) & _stem_variants(g))


def _variants(english: str) -> list[str]:
    """표제어를 슬래시 대안별로 편다.

    슬래시는 두 가지로 쓰인다. 하나로 묶어 처리하면 한쪽이 깨진다.

    | 표기 | 뜻 | 예 |
    |---|---|---|
    | 공백 있는 ` / ` | **구 단위** 대안 | `pay back / pay off / clear` |
    | 공백 없는 `a/b` | **낱말 단위** 대안 | `set a good/bad example` |

    구 단위로만 자르면 `set a good` 이 되어 `example` 이 날아가고,
    낱말 단위로만 자르면 `pay back  pay off  clear` 가 되어 셋을 모두 요구한다.
    """
    phrases = [p for p in _PHRASE_SLASH.split(english) if p.strip()] or [english]
    out: list[str] = []
    for phrase in phrases:
        choices = [w.split("/") for w in phrase.split()]
        if not choices:
            continue
        for combo in itertools.product(*choices):
            out.append(" ".join(c for c in combo if c))
            if len(out) >= _MAX_VARIANTS:
                return out
    return out or [english]


def _is_letter_placeholder(token: str) -> bool:
    """`extend A into B` 의 `A`·`B`. 대문자 한 글자는 자리표시자다 (refs #32).

    예문에서는 실제 대상으로 바뀐다: "They extended the road into the valley."

    빼는 것 둘:

    - **`I`** — 1인칭 대명사다. 표제어에 자리표시자로 쓰일 일이 없다
    - **소문자 `a`/`b`** — 관사 `a` 와 구분이 안 된다. 대문자만 본다
    """
    return len(token) == 1 and token.isupper() and token != "I"


def _targets(english: str) -> list[list[str]]:
    """표제어를 「매칭 후보 토큰 목록」들로 편다. 하나라도 맞으면 통과다.

    말줄임(`…`)·괄호·슬래시·자리표시자를 걷어낸다. **판정을 느슨하게 만드는
    변경**이므로, 무관한 문장이 통과하지 않는지 테스트로 함께 못 박는다.

    정규화가 표제어를 통째로 지우는 경우(예: 표제어 자체가 `something`)에는
    **원문으로 되돌린다** — 후보가 비면 무조건 실패가 되어 재생성만 반복한다.
    """
    stripped = _PARENS.sub(" ", _ELLIPSIS.sub(" ", english))
    out: list[list[str]] = []
    for variant in _variants(stripped):
        parts = [
            t
            for t in _tokens(variant)
            if t.lower() not in _PLACEHOLDERS and not _is_letter_placeholder(t)
        ]
        if parts and parts not in out:
            out.append(parts)
    return out or [_tokens(english)]


def _part_matches(token: str, part: str, *, loose: bool) -> bool:
    """표제어 구성 요소 하나가 예문 토큰과 맞는가."""
    # 접두사 표제어(`pro-`)는 붙어 나온다: `pro-` -> `pro-recycling`.
    # ⚠️ 접미사(`-ish`)는 여기서 못 잡는다 - _WORD_RE 가 낱말 첫 글자를 요구해
    # 표제어 단계에서 하이픈이 이미 사라진다. decisions.md 에 기록.
    if len(part) > 1 and part.endswith("-"):
        return token.lower().startswith(part.lower())
    # 소유격은 자리표시자와 함께 바뀐다. 구·관용구일 때만 유연하게 본다 -
    # 표제어가 `their` 한 낱말이면 그건 그 단어를 배우는 것이므로 그대로 요구한다.
    if loose and part.lower() in _POSSESSIVES:
        return token.lower() in _POSSESSIVES
    return _matches(token, part)


def _in_order(tokens: list[str], parts: list[str]) -> bool:
    """구성 요소가 순서대로 등장하는가 (사이에 다른 낱말이 끼어도 된다)."""
    loose = len(parts) > 1
    idx = 0
    for part in parts:
        while idx < len(tokens) and not _part_matches(tokens[idx], part, loose=loose):
            idx += 1
        if idx == len(tokens):
            return False
        idx += 1
    return True


def contains_target(example: str, english: str) -> bool:
    """대상 단어가 예문에 쓰였는가.

    - 어형 변화 허용 (`run` -> `running`)
    - 대소문자 무시
    - **구동사는 흩어져도 인정** (`put off` -> `put the meeting off`)
    - **부분 문자열은 인정 안 함** (`cat` 이 `category` 안에 있다고 통과시키지 않는다)
    - **표기 장치는 걷어낸다** — 슬래시 대안은 하나만 맞으면 되고, 괄호·자리표시자·
      소유격은 예문에서 바뀌므로 글자 그대로 요구하지 않는다 (refs #27)
    """
    tokens = _tokens(example)
    return any(_in_order(tokens, parts) for parts in _targets(english))


def _normalize(sentence: str) -> str:
    """구두점·대소문자·공백 차이를 지운다.

    재생성했는데 마침표만 다른 같은 문장이 나오면 사용자는 고장으로 여긴다.
    """
    return " ".join(_tokens(sentence)).lower()


def duplicates_any(example: str, avoid: list[str]) -> bool:
    """`avoid` 중 하나와 사실상 같은 문장인가."""
    if not avoid:
        return False
    target = _normalize(example)
    return any(_normalize(a) == target for a in avoid)


def has_korean(text: str) -> bool:
    """한글이 하나라도 있는가. 한글이 목적의 절반이다."""
    return bool(_HANGUL_RE.search(text))


def violations(example: str, example_ko: str, payload: WordExampleIn) -> list[str]:
    """위반 사유 목록. 비어 있으면 통과.

    사유를 문자열로 돌려주는 이유: 어느 기준을 어겼는지 알아야
    run error 메시지가 쓸모 있고, 재생성이 잦으면 프롬프트를 고칠 근거가 된다.
    """
    out: list[str] = []
    if not contains_target(example, payload.english):
        out.append(f"대상 단어가 예문에 없다: {payload.english!r}")
    n = word_count(example)
    if not length_ok(example):
        out.append(f"길이 위반: {n} 단어 (허용 {LENGTH_LO}~{LENGTH_HI})")
    if duplicates_any(example, payload.avoid):
        out.append("avoid 의 문장과 사실상 동일하다 (중복)")
    if not has_korean(example_ko):
        out.append("example_ko 에 한글이 없다")
    return out
