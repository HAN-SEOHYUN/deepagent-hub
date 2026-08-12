"""정책 훅 - 인증·쿼터가 들어올 자리 (refs #10).

**지금은 통과만 한다.** 그럼에도 호출 지점을 지금 만드는 이유:
나중에 붙일 때 `hub/graph.py` 를 안 고치기 위해서다.

지금 안 만드는 이유는 근거가 없어서다. 호출자가 본인 하나이고 Tailscale 이 경계이며,
쿼터 상한은 호출 기록이 쌓인 뒤에야 근거를 갖고 정할 수 있다.
"""

from __future__ import annotations

from hub.envelope import Envelope


def check(env: Envelope) -> None:
    """정책 위반이면 예외를 던진다. 통과하면 아무것도 안 한다.

    붙을 것:
      - 인증: `env.service` 가 아는 호출자인가
      - 인가: 그 호출자가 `env.task` 를 부를 수 있는가
      - 쿼터: 최근 호출 수가 상한을 넘지 않았는가

    셋 다 `env.service` 를 축으로 삼는다. 봉투에 `service` 를 처음부터
    필수로 받는 이유가 이것이다 - 나중에 붙일 때 호출자를 재배포하지 않는다.
    """
    return None
