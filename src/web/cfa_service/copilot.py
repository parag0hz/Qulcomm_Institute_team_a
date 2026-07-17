"""Grounded design copilot for Paragon."""

from __future__ import annotations

import json
import os
import re
from typing import Mapping, Sequence
from urllib import request

from .predictor import analyze_parameters, maybe_predict_parameters, optimize_parameters


SYSTEM_PROMPT = """You are Paragon Design Copilot for early vehicle aerodynamics studies.
Answer in the user's language. Use only the supplied Paragon calculations as numerical evidence.
Never invent CFD results or claim approximate mesh morphing is CAD-accurate.
Clearly distinguish surrogate prediction from CFD validation. Be concise and actionable.
When suggesting changes, explain the likely direction and mention domain or uncertainty warnings."""


def copilot_status() -> dict[str, object]:
    configured = bool(os.getenv("OPENAI_API_KEY"))
    return {
        "configured": configured,
        "provider": "OpenAI Responses API" if configured else "Grounded local explainer",
        "model": os.getenv("OPENAI_MODEL", "gpt-5-mini") if configured else None,
        "message": "Server-side API key detected." if configured else "Set OPENAI_API_KEY to enable LLM explanations.",
    }


def ask_copilot(
    message: str,
    parameters: Mapping[str, object],
    history: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    message = message.strip()
    if not message:
        raise ValueError("Copilot message is required.")
    if len(message) > 2000:
        raise ValueError("Copilot message is too long (maximum 2000 characters).")

    prediction = maybe_predict_parameters(parameters)
    analysis = analyze_parameters(parameters)
    goal = _extract_cd_goal(message)
    optimization = optimize_parameters(parameters, goal, []) if goal is not None else None
    evidence = {
        "prediction": {
            "cd": prediction["cd"],
            "percentile": prediction["percentile"],
            "provider": prediction["provider"],
            "domain_status": prediction["domain_status"],
            "nearest_sample_distance": prediction["nearest_sample_distance"],
            "uncertainty": prediction["uncertainty"],
            "warnings": prediction["warnings"],
        },
        "top_drivers": analysis["drivers"][:5],
        "goal_search": optimization,
    }

    if os.getenv("OPENAI_API_KEY"):
        try:
            answer = _call_openai(message, evidence, history or [])
            provider = "OpenAI Responses API"
            model = os.getenv("OPENAI_MODEL", "gpt-5-mini")
        except Exception as exc:
            answer = _local_answer(message, evidence, llm_error=str(exc))
            provider = "Grounded local fallback"
            model = None
    else:
        answer = _local_answer(message, evidence)
        provider = "Grounded local explainer"
        model = None

    return {
        "answer": answer,
        "provider": provider,
        "model": model,
        "evidence": evidence,
        "disclaimer": "Surrogate-model design guidance; validate shortlisted concepts with CFD.",
    }


def _call_openai(message: str, evidence: Mapping[str, object], history: Sequence[Mapping[str, object]]) -> str:
    payload = {
        "model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
        "instructions": SYSTEM_PROMPT,
        "input": [
            *[
                {"role": str(item.get("role", "user")), "content": str(item.get("content", ""))[:2000]}
                for item in history[-6:]
            ],
            {
                "role": "user",
                "content": f"Designer question:\n{message}\n\nParagon evidence JSON:\n{json.dumps(evidence, ensure_ascii=False)}",
            },
        ],
        "max_output_tokens": 700,
    }
    req = request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"LLM request failed: {exc}") from exc
    chunks = []
    for output in result.get("output", []):
        for content in output.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(content["text"])
    if not chunks:
        raise RuntimeError("LLM response contained no output text.")
    return "\n".join(chunks)


def _local_answer(message: str, evidence: Mapping[str, object], llm_error: str | None = None) -> str:
    prediction = evidence["prediction"]
    drivers = evidence["top_drivers"]
    lines = [
        f"현재 설계의 예측 Cd는 {prediction['cd']:.4f}, 데이터셋 위치는 P{prediction['percentile']:.0f}입니다.",
        f"학습 영역 상태는 {prediction['domain_status']}이고 가장 가까운 샘플 거리는 {prediction['nearest_sample_distance']:.3f}입니다.",
    ]
    if drivers:
        lines.append("현재 국소 민감도가 큰 항목은 " + ", ".join(item["label"] for item in drivers[:3]) + "입니다.")
    goal_search = evidence.get("goal_search")
    if goal_search and goal_search.get("recommendations"):
        best = goal_search["recommendations"][0]
        names = ", ".join(change["label"] for change in best["changes"][:3]) or "변경 없음"
        lines.append(f"요청한 목표에 가장 가까운 탐색안은 예상 Cd {best['cd']:.4f}이며 주요 변경은 {names}입니다.")
    warnings = prediction.get("warnings") or []
    if warnings:
        lines.append("주의: " + " ".join(warnings))
    lines.append("이 값은 surrogate 모델 기반 설계 가이드이며 최종 선택 전 CFD 검증이 필요합니다.")
    if llm_error:
        lines.append("LLM 연결에 실패해 계산 근거만으로 답변했습니다.")
    return "\n".join(lines)


def _extract_cd_goal(message: str) -> float | None:
    match = re.search(r"(?:cd\s*)?(0\.[12-3]\d{1,3})", message, flags=re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    return value if 0.18 <= value <= 0.36 else None
