# 운영 이미지 - FastAPI 직접 서빙 (refs #25, D22·D23·D24)
#
# langgraph build 를 쓰지 않는 이유: 베이스 langchain/langgraph-api 는 self-hosted 에
# Enterprise 라이선스가 필요해 기동되지 않는다 (2026-08-20 실측 exit 3).
#
# 빌드는 반드시 amd64 로 - 홈서버(idwwt-k8s-node1)는 x86_64 다. 로컬 Apple Silicon 에서
# 그냥 빌드하면 서버에서 exec format error 로 죽는다:
#
#   docker buildx build --platform linux/amd64 -t docker.io/rosasian/deepagent-hub:<tag> .
#
# env 는 이미지에 없다 (.dockerignore 가 .env 제외). 런타임 주입:
#   로컬 검증:  docker run --env-file .env ...
#   운영(k8s):  envFrom secretRef/configMapRef (#4)

FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# 의존성 레이어 분리 - 코드만 바뀌면 재설치 없이 캐시를 탄다.
# extras(dev) 를 설치하지 않으므로 ELv2(langgraph-api)·pytest·ruff 가 이미지에 없다.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# 패키지는 명시 목록만 - pyproject packages.find 와 같은 명단 (D10)
COPY hub/ hub/
COPY word_example/ word_example/
COPY serve/ serve/
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# --workers 1 명시 - 다중이면 관측 파일 경합 + 메모리 목표(200MB) 배가 (D21 규모)
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
