# syntax=docker/dockerfile:1
# Paragon Vehicle Design Studio — 단일 서비스 배포용 이미지.
# FastAPI가 API와 빌드된 React SPA를 함께 서빙하므로 서비스 하나면 충분하다.

# ---------- Stage 1: 프론트엔드 빌드 ----------
FROM node:22-slim AS frontend

WORKDIR /app

# 의존성 레이어를 소스와 분리해 캐시 적중률을 높인다.
COPY src/package.json src/package-lock.json ./
RUN npm ci

COPY src/web/frontend ./web/frontend
# src/styles.css가 `@import "../../cfa_service/static/styles.css"`로 백엔드의
# 디자인 시스템 스타일시트를 가져오므로 빌드 단계에도 함께 있어야 한다.
COPY src/web/cfa_service/static ./web/cfa_service/static

RUN npm run build:web


# ---------- Stage 2: 파이썬 런타임 ----------
FROM python:3.12-slim

# 파이썬 로그가 버퍼링 없이 Render 로그로 바로 나가도록.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY src/web/requirements-web.txt ./web/requirements-web.txt
RUN pip install --no-cache-dir -r web/requirements-web.txt

# 애플리케이션 소스 (ParametricModels CSV, GLB 에셋 포함)
COPY src/ ./

# 스테이지 1에서 만든 SPA 번들을 FastAPI가 서빙하는 경로에 배치
COPY --from=frontend /app/web/frontend/dist ./web/frontend/dist

# 학습 산출물(.pkl)은 저장소에 커밋하지 않으므로 빌드 시점에 생성한다.
# 기본 모델이 RandomForest라 scikit-learn만 있으면 되고 3~5초면 끝난다.
RUN python web/models/train_parametric_baseline.py

# Render가 $PORT를 주입한다. 로컬 docker run 시엔 8000으로 뜬다.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn web.cfa_service.app:app --host 0.0.0.0 --port ${PORT}"]
