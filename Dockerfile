# syntax=docker/dockerfile:1
# Paragon — 단일 서비스 배포용 이미지.
# FastAPI(backend/)가 API와 빌드된 React SPA(frontend/)를 함께 서빙한다.

# ---------- Stage 1: 프론트엔드 빌드 ----------
FROM node:22-slim AS frontend

WORKDIR /app

# 의존성 레이어를 소스와 분리해 캐시 적중률을 높인다.
COPY frontend/package.json frontend/package-lock.json ./frontend/
WORKDIR /app/frontend
RUN npm ci

WORKDIR /app
COPY frontend ./frontend
# frontend/src/styles.css가 `@import "../../backend/cfa_service/static/styles.css"`로
# 백엔드의 디자인 시스템 스타일시트를 가져오므로, 빌드 단계에도 저장소와 동일한
# 상대 위치에 그 파일이 있어야 한다.
COPY backend/cfa_service/static ./backend/cfa_service/static

WORKDIR /app/frontend
RUN npm run build:web


# ---------- Stage 2: 파이썬 런타임 ----------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY backend/requirements-web.txt ./backend/requirements-web.txt
RUN pip install --no-cache-dir -r backend/requirements-web.txt

# 백엔드 소스 (ParametricModels CSV, cfa_service/static GLB 포함)
COPY backend ./backend

# PointNet 서빙 자산(ONNX + 데모 홀드아웃 점군). 서비스 트리 안에 둔다.
COPY ml/models ./backend/cfa_service/models

# 스테이지 1의 SPA 번들을 FastAPI가 서빙하는 경로(FRONTEND_DIST)에 배치
COPY --from=frontend /app/frontend/dist ./frontend/dist

# 학습 산출물(.pkl)은 커밋하지 않으므로 빌드 시점에 생성한다(RandomForest, 수 초).
RUN python backend/models/train_parametric_baseline.py

# Render가 $PORT를 주입한다. 로컬 docker run 시엔 8000으로 뜬다.
ENV PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn backend.cfa_service.app:app --host 0.0.0.0 --port ${PORT}"]
