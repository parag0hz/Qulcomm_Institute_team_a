# 평가 지표 정리 (METRICS.md)

이 프로젝트의 모든 학습·평가에 사용된 지표의 정의, 계산 방법, 사용 위치, 선택 이유. 수치 자체는 [RESULTS.md](RESULTS.md)(canonical)와 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)를 참조.

---

## 1. 핵심 3지표 (모든 모델 공통 — [eval_metrics.py](eval_metrics.py))

저장된 test 예측(`outputs/*_pred.npz`)을 세 가지 렌즈로 평가한다. 하나의 지표만 보고하는 것은 금지 (CLAUDE.md 평가 원칙).

### 1-1. R² (결정계수)

```
R² = 1 − Σ(ŷᵢ − yᵢ)² / Σ(yᵢ − ȳ)²
```

- **기준선이 "부분집합 자신의 평균"** — 전체 평균이 아니라, 평가 중인 그룹(예: Estate)의 평균 Cd를 찍는 것보다 나은가를 묻는다.
- 답하는 질문: *평균 찍기보다 나은가?* 음수면 그 그룹 평균을 예측하는 것만도 못하다는 뜻.
- 주의: 분산이 작은 클래스(Estate/Notchback σ≈0.019 vs Fastback 0.038)에서는 같은 절대 오차라도 R²가 훨씬 가혹하게 나온다. 질문은 옳지만 크기 비교는 오해 소지 → MAE와 함께 봐야 함.

### 1-2. MAE (drag counts)

```
MAE(counts) = mean(|ŷᵢ − yᵢ|) × 1000        (1 count = 0.001 Cd)
MAE(%)      = mean(|ŷᵢ − yᵢ| / yᵢ) × 100
```

- 답하는 질문: *물리 단위로 얼마나 틀리나?* 공기역학 도메인 원단위(drag count)라 클래스 간 직접 비교 가능.
- 분산 차이에 영향받지 않으므로 R²의 왜곡을 보정하는 역할.

### 1-3. 쌍별 순위 정확도 (pairwise ranking accuracy)

```
무작위 설계쌍 (i, j) 20만 개 추출 (seed 0, i≠j):
rank_acc = mean( (ŷᵢ < ŷⱼ) == (yᵢ < yⱼ) ) × 100
```

- 답하는 질문: *두 설계 중 저항 낮은 쪽을 맞히는가?* — 제품이 실제로 주장하는 기능(상대적 설계 피드백)과 일치하는 **헤드라인 지표**. 50% = 동전던지기.

### 보조: Spearman ρ

test 집합 전체의 예측-실측 순위 상관. 순위 정확도의 연속형 보완 지표로 같은 표에 출력.

### 보고 규칙 (지표만큼 중요)

- **차종별(Fastback/Estate/Notchback) 분해 필수** — Fastback이 68%라 전역 평균은 소수 클래스 실패를 숨긴다.
- **쌍둥이/고립(twin/lone) 분리** — train에 같은 인덱스 형제가 있는 test 설계(516)와 없는 설계(642)를 나눠 암기 여부를 점검.
- 열 구성: `n / R² / MAE% / MAE(counts) / Spearman / 순위 정확도`.

---

## 2. 학습 시 사용된 손실·선택 기준

### 딥러닝 트랙 ([train_r2.py](train_r2.py), holdout, crossmodal 파인튜닝 공통)

| 항목 | 내용 |
|---|---|
| 학습 손실 | **SmoothL1 (Huber, β=1.0)** — 표준화된 Cd 타깃에 적용. 이상치에 MSE보다 강건 |
| 타깃 전처리 | Cd를 train 평균/표준편차로 표준화 (평가 시 역변환) |
| 모델 선택 | **val R² 최대** 시점의 가중치 저장 (best checkpoint) |
| 조기 종료 | val R² 기준 patience 30 epoch |
| wandb 로깅 | `train_loss`, `val_r2`, `best_val_r2`, `lr` (매 epoch) |

### 머신러닝 트랙 ([scripts/automl_parametric.py](scripts/automl_parametric.py) — 논문 §5.1.2 프로토콜)

| 항목 | 내용 |
|---|---|
| 평가 지표 | **R²** (`sklearn.r2_score`; AutoGluon도 `eval_metric="r2"`) |
| 프로토콜 | 80/20 분할 × **20회 반복** (seed 42 계열), train 크기 스윕 |
| 집계 | mean R² ± std, **95% CI = t₀.₉₇₅(df=n−1) · std/√n** |
| AutoGluon 내부 | 8개 기본 모델 학습 후 val 성능 기반 greedy 가중 앙상블 (WeightedEnsemble_L2) |

---

## 3. ΔCd 상대 비교 지표 ([scripts/delta_cd_eval.py](scripts/delta_cd_eval.py))

기하적으로 유사한 설계쌍(voxel-NN 매칭)에서 Cd **차이**를 얼마나 맞히는가 — "이 수정이 저항을 줄였는가"라는 제품 질문의 직접 검증.

| 지표 | 정의 |
|---|---|
| ΔCd R² | Δŷ vs Δy의 R² |
| ΔMAE (counts) | mean\|Δŷ − Δy\| × 1000 |
| **부호 정확도** | sign(Δŷ) == sign(Δy) 비율 (Δy≠0 쌍만) — 개선/악화 방향을 맞히는가 |
| 구간 분해 | \|ΔCd\| < 5 / 5–15 / 15+ counts 구간별 부호 정확도·MAE·R² — **해상도 한계**(≈7 counts) 도출 |

측정 결과: 15+ counts 차이는 95% 정확, 5 counts 미만은 57%(≈우연) → 모델의 분해능 공표에 사용.

---

## 4. 일반화·누수 검증 지표

| 프로브 | 스크립트 | 지표 |
|---|---|---|
| 계열 홀드아웃 (unseen family) | [scripts/holdout_eval.py](scripts/holdout_eval.py) `--test-prefix` | §1의 3지표 동일 적용 |
| 차종 홀드아웃 (unseen body type) | 〃 `--test-body Estate` | 〃 |
| 암기/누수 프로브 | [check_leakage.py](check_leakage.py) | **zero-learning voxel 1-NN/5-NN 검색 R²** — 학습 없이 최근접 이웃의 Cd를 복사했을 때의 성능. 모델 성능의 "검색 가능 하한선" |
| 쌍둥이 이득 | eval_metrics.py twin/lone 분해 | 두 부분집합 간 MAE·순위 정확도 차 |

해석 규칙: 공식 split R²(0.968)만으로 일반화를 주장하지 않는다 — 1-NN만으로 0.865가 나오는 데이터셋이므로, 3단 프레임(공식 split 0.968 / unseen family 0.75–0.80 / unseen body 0.457)으로 보고.

---

## 5. 크로스모달 실험 지표 ([scripts/train_crossmodal.py](scripts/train_crossmodal.py), [scripts/crossmodal_phase2.py](scripts/crossmodal_phase2.py))

| 지표 | 정의 | 답하는 질문 |
|---|---|---|
| InfoNCE (CLIP) loss | 양방향 cross-entropy 평균, 학습 손실 | 잠재공간 정렬 학습 |
| **검색 top-1 / top-5** | test 쌍 N개 중 형상 잠재로 올바른 파라미터 행 검색 (무작위 기준 1/N) | 두 모달이 정렬되었는가 |
| 역추정 R² (per-param) | z_shape → 23개 설계변수 회귀, 변수별 R² | inverse design 실현성 |
| Cd 선형 프로브 R² | z_shape → Cd Ridge 회귀 | 대조학습 특징의 품질 |
| 루프 폐쇄 R² | 형상→예측 파라미터→XGB→Cd vs 실제 Cd | 파이프라인 연결 가능성 |
| 저라벨 파인튜닝 test R² | {5,10,25}% 라벨 × {pretrained, scratch} | 사전학습의 라벨 효율 이득 |

모델 선택은 **val top-1 검색 정확도** 최대 시점.

---

## 6. 지표별 대표 수치 (요약)

| 지표 | 대표값 (PointNet, 공식 test) |
|---|---|
| R² (전체 / Estate / Notchback) | +0.968 / +0.839 / +0.802 |
| MAE | 5.1 counts (전체) |
| 순위 정확도 | 94.6% (Estate 87.3%) |
| ΔCd 부호 정확도 (15+ counts) | 95% |
| 계열 홀드아웃 R² | 0.750–0.799 |
| 차종 홀드아웃 R² (Estate 제외 학습) | 0.457 |
| 1-NN 검색 하한선 R² | 0.865 |
| 크로스모달 검색 top-1 (559쌍) | 85.2% (무작위 0.18%) |
