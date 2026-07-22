# A100_DL_HANDOFF.md — 딥러닝 백본 비교를 A100에서 완결하기

이 문서는 **A100 서버에서 딥러닝 부분만 돌려 끝내기 위한 작업 지시서**다. 환경·데이터 준비는 [A100_BOOTSTRAP.md](A100_BOOTSTRAP.md)를 먼저 따르고, 이 문서는 **그 위에서 무엇을 실행할지**만 다룬다.

- **왜 A100인가**: 5080(16GB)에서는 RegDGCNN이 2048점에서 OOM이라 **4개 백본을 전부 1024점으로 낮춰야** 했다. A100 40/80GB면 **2048점에서 전 백본 실행이 가능**하므로, 메인 결과(PointNet 2048)와 **동일 조건**의 깨끗한 비교가 나온다. 또한 RegDGCNN 5-fold가 5080에선 ~4시간, DGCNN HPO는 ~3시간이라 사실상 불가능하다.
- **역할 분담**: **1024점은 5080이 담당**(현재 실행 중). **A100은 2048점과 4096점만** 맡는다.
- **목표**: 새 평가 프로토콜(교집합 3,704대 · K=5 rotating)에서
  1. **4개 백본(PointNet/DGCNN/RegDGCNN/Triplane) × 2048점** 5-fold — 최우선
  2. **4096점** 5-fold — 점 개수 포화 확인
  3. 가능하면 **DGCNN·RegDGCNN 하이퍼파라미터 튜닝**

---

## 0. ⛔ 16GB에서 막힌 것 — A100이 풀어야 할 목록 (2026-07-22 갱신)

RTX 5080(15.46 GiB 가용)에서 **실제로 실패한 것들**. 전부 재현 로그가 남아 있다.

| # | 막힌 작업 | 증상 | 영향 |
|---|---|---|---|
| **1** | **RegDGCNN 2048점 학습** | `Tried to allocate 1.25 GiB` OOM | **4백본 비교를 전부 1024점으로 강등**해야 했다 |
| **2** | RegDGCNN 4096점 | 미시도 (2048도 안 되므로) | 점 개수 포화 검증 불가 |
| **3** | **DGCNN 100k 입력** | bs=1에서 `1192.09 GiB` 요구 → OOM | 100k 실험에서 **"불가"로 기록** |
| **4** | **RegDGCNN 100k 입력** | 동일 (kNN N×N 행렬) | 동일 |
| **5** | PointNet 100k 입력 | bs 32/16/8 전부 OOM → **bs=4만 가능** | 배치 통계 붕괴로 R² 0.968→0.911 하락. **점 개수 효과와 배치 교란이 뒤섞임** |
| **6** | RegDGCNN 2048점 해석 실험 | OOM (위 #1과 동일) | Grad-CAM·가림실험을 1024점으로 재실행해야 했다 |

**시간 제약(OOM은 아니지만 실질적 차단)**

| # | 작업 | 5080 소요 | 상태 |
|---|---|---|---|
| 7 | RegDGCNN 5-fold (1024점) | **6.9시간** | 완료했으나 반복 불가 |
| 8 | DGCNN 하이퍼파라미터 탐색 | ~3시간 | **미실행** |
| 9 | RegDGCNN 하이퍼파라미터 탐색 | ~20시간 | **미실행** |

**공정성 문제 (A100에서 함께 해결할 것)**

`#10` **RegDGCNN fold별 조기종료 시점이 제각각이라 과소평가됐을 수 있다.** patience 30으로 인해:

| fold | 종료 epoch | test R² |
|---|---:|---:|
| 1 | **ep51** (조기) | **0.750** ← 최저 |
| 2 | ep116 (거의 완주) | 0.779 |
| 3 | ep70 | 0.784 |
| 4 | ep73 | 0.793 |
| 5 | ep73 | 0.790 |

cosine 스케줄이 120 epoch 기준이라 중간에 끊기면 LR 감쇠가 미완이다. PointNet·DGCNN은 대부분 ep107~119까지 갔다.
같은 현상으로 배포용 PointNet이 ep43 종료 시 0.814 → 완주 시 0.892로 **0.078 차이**가 났다.
**→ A100에서는 `--patience 200`(사실상 해제)으로 전 백본을 완주시켜 공정성을 확보할 것.**

### A100에서 할 일 (위 목록에 대응)

```bash
# ①②⑥ 4백본 × 2048점, 조기종료 없이 완주
python scripts/run_protocol_comparison.py --only dl --npoints 2048 \
  --backbones pointnet dgcnn regdgcnn --out outputs/protocol_dl2048_a100.json
#   ⚠ run_protocol_comparison.py 의 run_dl(patience=30)을 200으로 올릴 것 (#10 공정성)

# ② 4096점 (fps4096.npz 필요)
python scripts/run_protocol_comparison.py --only dl --npoints 4096 --cache data/fps4096.npz \
  --backbones pointnet dgcnn regdgcnn --out outputs/protocol_dl4096_a100.json

# ③④⑤ 100k 전체 입력 — A100 40/80GB면 DGCNN/RegDGCNN도 가능할 수 있다
python scripts/train_100k.py --build-cache          # 캐시 없을 때만
python scripts/train_100k.py --backbone pointnet --bs 32    # bs=4 → 32로 배치 교란 제거
python scripts/train_100k.py --backbone dgcnn --bs 2        # 5080에선 bs=1도 불가였음
python scripts/train_100k.py --backbone regdgcnn --bs 2

# ⑧⑨ HPO (tune_optuna.py의 tune_dl을 BACKBONES[backbone] 받도록 일반화 필요)
python scripts/tune_optuna.py --models pointnet --dl-trials 25

# ⑥ 해석 실험을 2048점으로 (RegDGCNN 포함)
python scripts/saliency_compare.py --backbones pointnet dgcnn regdgcnn --npoints 2048 --train
python scripts/gradcam_all.py    --backbones pointnet dgcnn regdgcnn --npoints 2048 --n 3
python scripts/occlusion_test.py --backbones pointnet dgcnn regdgcnn --npoints 2048 --n 5
```

> **Triplane은 제외한다.** TripNet 공식 코드가 비공개라 우리 `TriplaneCNN`은 표현 아이디어만 가져온
> 자체 경량 재구현이다. 성능(1024점 R² 0.339)은 **TripNet이 아니라 이 구현의 성능**이므로,
> 성능 표에 약한 baseline으로만 남기고 **해석 실험·논문 비교에는 쓰지 않는다.**


---

## 1. 배경 — 왜 이 프로토콜인가 (리뷰 피드백 반영본)

지도교수/리뷰어 피드백으로 평가 방식을 전면 교체했다. **이 규칙은 바꾸지 말 것.**

| # | 피드백 | 반영 |
|---|---|---|
| 1 | ML/DL 학습 데이터가 달라 정량 비교 불가 | **교집합 3,709대**만 사용 (파라미터 CSV ∩ 포인트클라우드) |
| 2 | 논문 프로토콜 대신 k-fold를 쓸 것 | **K=5 rotating**: 학습 3 fold / 검증 1 / 테스트 1, 5세트 회전 |
| 4 | R²·MAE·MSE 전부 정리 + 명시적 HPO | 6지표 산출 + **Optuna** 탐색 |
| 6 | 데모용 데이터를 학습에서 뺄 것 | **5대 영구 제외** → 3,704대 |

회전 방식:
```
세트1  train 1,2,3 | val 4 | test 5      세트4  train 4,5,1 | val 2 | test 3
세트2  train 2,3,4 | val 5 | test 1      세트5  train 5,1,2 | val 3 | test 4
세트3  train 3,4,5 | val 1 | test 2
```
차종 층화(stratified)로 각 fold의 Fastback/Estate/Notchback 비율이 전체와 동일하다.

**⚠ 데이터 성격이 기존과 다르다**: 3,704대 · Cd 0.256 ± 0.023 (기존 7,713대는 0.284 ± 0.037). 교집합이 신세대만 남아 **분산이 좁아져 R²가 구조적으로 낮게 나온다.** 기존 0.968과 직접 비교하지 말 것.

---

## 2. 전송해야 할 파일

[A100_BOOTSTRAP.md](A100_BOOTSTRAP.md)의 rsync로 저장소를 통째로 옮겼다면 대부분 따라온다. **부트스트랩 문서 작성 이후 새로 생긴 파일**이므로 반드시 포함 확인:

| 파일 | 역할 | 필수 |
|---|---|---|
| `scripts/protocol.py` | **공통 프로토콜** — 데이터·분할·지표. 모든 실행이 이걸 거친다 | ✅ |
| `scripts/run_protocol_comparison.py` | ML/DL 5-fold 실행기 (`--backbones`, `--npoints`) | ✅ |
| `scripts/tune_optuna.py` | Optuna HPO (탐색은 val로만, test 미사용) | ✅ |
| `scripts/make_demo_holdout.py` | 데모 홀드아웃 생성기 | ○ |
| `data/demo_holdout.json` | **제외할 5대 ID** — 없으면 protocol.py가 죽는다 | ✅ |
| `data/fps2048.npz` | FPS 캐시 (2048점) | ✅ |
| `data/fps4096.npz` | **FPS 캐시 (4096점, 363 MiB)** — 4096 실험에 필수 | ✅ |
| `data/DrivAerNet_ParametricData.csv` | 교집합 계산·ML 트랙에 필요 | ✅ |
| `outputs/protocol_comparison.json` | 기존 결과 (비교 기준) | ○ |
| `outputs/optuna_results.json` | 기존 HPO 결과 | ○ |

### 코드는 GitHub에서 (2026-07-21부터)

코드·문서는 **팀 저장소의 `dongwon` 브랜치 `ml/` 폴더**에 올라가 있다. rsync 대신 git으로 받는다:

```bash
git clone -b dongwon https://github.com/parag0hz/Qulcomm_Institute_team_a ~/team_repo
cd ~/team_repo/ml          # ← 여기가 작업 디렉토리 (기존 qi/ 와 동일 구조)
git pull                   # 이후 갱신은 이것만
```

**데이터는 저장소에 없다** (수십 GB). 데이터만 rsync로 받는다:
```bash
mkdir -p ~/team_repo/ml/data
rsync -avhP --exclude='DrivAer++_Points.tar' --exclude='pc100k_f32.dat' \
  kwy00@192.168.0.105:/home/kwy00/qi/data/  ~/team_repo/ml/data/
export QI_DATA=~/team_repo/ml/data
```
> `data/demo_holdout.json`은 git에 포함돼 있으니 rsync가 덮어써도 무방하다.

**결과 회수도 git으로**: A100에서 나온 `outputs/*.json`은 용량이 작으니 커밋해서 푸시하면 된다(그림 PNG는 .gitignore로 막혀 있으니 필요하면 `git add -f`).

**환경**: `automl` env 하나로 ML+DL 모두 실행 가능하다(torch+autogluon+lightgbm+optuna). A100에서는 [A100_BOOTSTRAP.md](A100_BOOTSTRAP.md) §2대로 표준 torch(cu124 등)를 쓰면 되고, **cu130 인덱스는 5080 전용이니 쓰지 말 것**. optuna는 `pip install optuna` 추가 필요.

**첫 확인**:
```bash
cd ~/qi && export QI_DATA=$PWD/data
python scripts/protocol.py        # 3,704대 / fold 균형 / 무결성 검증 통과해야 함
```

---

## 3. 실행할 것 (우선순위 순)

### ① 4개 백본 × 2048점 5-fold — **최우선, 이게 A100에 온 이유**

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python scripts/run_protocol_comparison.py --only dl --npoints 2048 \
  --backbones pointnet dgcnn regdgcnn triplane \
  --out outputs/protocol_dl2048_a100.json
```
- 배치 기본값은 `BS_DEFAULT`(pointnet 32 / dgcnn 16 / regdgcnn 8 / triplane 32). **A100에서는 RegDGCNN을 16 이상으로 올려도 되지만, 올릴 거면 4개 백본 모두 동일 배치로 맞추는 게 공정하다.** 배치가 결과에 영향을 주므로 비교 조건을 흔들지 말 것.
- 5080에서 RegDGCNN 1024점/bs8이 ~50초/epoch였다. A100 2048점이면 비슷하거나 빠를 것으로 예상.

### ② DGCNN·RegDGCNN 하이퍼파라미터 튜닝

```bash
python scripts/tune_optuna.py --models pointnet --dl-trials 25 --dl-search-epochs 60
```
> ⚠ `tune_optuna.py`의 `tune_dl`은 현재 **PointNet 고정**이다. DGCNN/RegDGCNN을 튜닝하려면 `train_pointnet()`이 `BACKBONES[backbone]`을 받도록 일반화해야 한다(수 줄 수정). 5080에선 시간이 없어 못 했다.
- 탐색 공간은 lr / weight_decay / batch / dropout / emb. **탐색은 세트1의 val로만, test는 최종 1회만** — 이 원칙을 절대 깨지 말 것.

### ③ 4096점 5-fold — 점 개수 포화 확인

```bash
python scripts/run_protocol_comparison.py --only dl --npoints 4096 \
  --cache data/fps4096.npz \
  --backbones pointnet dgcnn regdgcnn triplane \
  --out outputs/protocol_dl4096_a100.json
```

> ⚠️ **조용한 절단 함정 (해결됨)**: `--npoints 4096`인데 fps2048 캐시를 쓰면 슬라이싱이 에러 없이 **2048점만 반환**해 "4096 결과"라는 잘못된 라벨이 붙는다. 지금은 npoints>2048이면 fps4096.npz를 자동 선택하고, 캐시 점수가 모자라면 **즉시 종료하는 assert**가 들어 있다. 그래도 `--cache`를 명시하는 습관을 권장.

**기대 결과**: 5080에서 크로스모달 실험 결과 **4096은 2048 대비 이득 없음**(검색 88.4→87.8%, 역추정 0.667→0.658)이었다. Cd 회귀에서도 같은 포화가 확인되면 **"2048이 최적 예산"**이 두 과제에서 독립적으로 입증된다.

### ④ (5080에서 진행 중이라 A100은 불필요) 1024점
5080이 담당 중. A100에서 중복 실행하지 말 것.

---

## 4. 지금까지 나온 결과 (A100에서 비교 기준)

**모두 동일 프로토콜(3,704대 · K=5 rotating), 5-fold 평균 ± 표준편차.**

### ML 트랙 (설계 파라미터 23개)
| 모델 | R² | MAE | MSE | 튜닝 후 R² |
|---|---:|---:|---:|---:|
| AutoGluon | 0.573 ± 0.027 | 0.01166 | 2.24e-04 | — |
| LightGBM | 0.557 ± 0.033 | 0.01175 | 2.32e-04 | **0.563** |
| GradientBoosting | 0.554 ± 0.018 | 0.01238 | 2.34e-04 | — |
| XGBoost | 0.516 ± 0.028 | 0.01195 | 2.53e-04 | **0.561** |
| RandomForest | 0.486 ± 0.025 | 0.01278 | 2.69e-04 | — |

→ **튜닝해도 0.56대에서 수렴** = 설계 파라미터만으로는 R² ≈ 0.57이 천장.

### DL 트랙 — **1024점 5-fold 완료** (2026-07-22)
| 백본 | R² | MAE | MAPE | 순위acc | 비고 |
|---|---:|---:|---:|---:|---|
| **PointNet** | **0.8483 ± 0.0141** | 0.00701 | 2.74% | 87.8% | 최고 |
| DGCNN | 0.8177 ± 0.0077 | 0.00777 | 3.05% | 86.3% | |
| RegDGCNN | 0.7792 ± 0.0152 | 0.00864 | 3.40% | 86.4% | **조기종료로 과소평가 가능(§0 #10)** |
| Triplane | 0.3394 ± 0.0935 | 0.01496 | 5.78% | 73.6% | 자체 재구현, 참고용 |

**2048점 PointNet**(기본 0.853 / 튜닝 0.865)은 별도 실행. 2048점 4백본 동시 비교는 **A100 담당**.

차종별 R² — PointNet만 균일:
| 백본 | Fastback | Estate | Notchback |
|---|---:|---:|---:|
| PointNet | 0.778 | 0.780 | 0.777 |
| DGCNN | 0.750 | 0.727 | 0.724 |
| RegDGCNN | 0.705 | 0.658 | 0.671 |

### 모델 해석 (가림 실험, 2048점 · 대조군 대비 배율)
| 구역 | PointNet | DGCNN |
|---|---:|---:|
| **하부(언더바디)** | **26.1×** | 1.8× |
| 후면 | 6.6× | **2.3×** |

**PointNet은 하부(지면 기준선 → 절대 높이), DGCNN은 후면(후류)에 의존** — 아키텍처마다 전략이 다르다.
RegDGCNN은 2048점 OOM으로 미포함(§0 #6) → **A100에서 채울 것**.

### 확인된 결론 (A100 결과로 검증/반증할 것)
1. **동일 조건에서도 DL 압도**: PointNet 0.853 vs 최고 ML 0.573 → **격차 +0.28**. 튜닝 후에도 0.865 vs 0.563으로 **격차 유지(+0.30)** — 하이퍼파라미터로 설명되지 않는 모달리티 우위.
2. **ML은 Estate에서 붕괴**: AutoGluon Estate R² +0.038, RandomForest **−0.257**. PointNet은 세 차종 모두 0.78대로 균일.
3. **DL이라고 다 좋은 게 아니다**: Triplane 0.370은 테뷸러 ML보다 낮다. **구조 선택이 모달리티만큼 중요**하다.
4. **PointNet 최적 emb는 512** (기본 1024의 절반) — 더 작은 모델이 이겼다.

---

## 5. 반드시 지킬 것

1. **프로토콜을 바꾸지 말 것** — 데이터 범위(3,704), fold 배정(seed 42 층화), 회전 방식은 리뷰 피드백 반영본이다. `protocol.py`를 수정하면 기존 결과와 비교 불가.
2. **데모 5대는 영원히 제외** — `data/demo_holdout.json`. protocol.py가 자동 처리하니 우회하지 말 것.
3. **HPO는 val로만** — test를 목적함수에 넣는 순간 모든 수치가 무의미해진다. val R²(0.89)와 test R²(0.865)의 격차가 정상이며, 그 격차가 곧 정직함의 증거다.
4. **배치 크기를 백본마다 다르게 쓸 거면 명시할 것** — 배치는 결과에 영향을 준다. 공정 비교를 주장하려면 동일 배치가 이상적.
5. **미터 스케일 유지** — 학습 fold 기준 상수 평행이동만. unit-sphere 정규화 금지 ([A100_BOOTSTRAP.md](A100_BOOTSTRAP.md) §5-4).

---

## 6. 산출물 & 회수

A100에서 나오면 원 서버로 되가져올 것:
```
outputs/protocol_dl2048_a100.json      # 4백본 × 2048점 5-fold 전체 지표
outputs/protocol_dl4096_a100.json      # 4백본 × 4096점 5-fold
outputs/optuna_results.json            # DGCNN/RegDGCNN 튜닝 결과 (덮어쓰지 말고 별도 파일명 권장)
outputs/optuna_*.png                   # 최적화 히스토리 + 파라미터 중요도
```
```bash
rsync -av ~/qi/outputs/protocol_dl*_a100.json ~/qi/outputs/optuna_* \
  kwy00@192.168.0.105:/home/kwy00/qi/outputs/
```

**보고 형식**: 4개 백본 × (R²/MAE/MSE/RMSE/MAPE/순위정확도) × (전체 + 차종별), 5-fold 평균±표준편차. `protocol.py`의 `evaluate()`/`aggregate()`가 이미 그 형태로 만들어 준다.

---

## 7. 참고 문서
- [A100_BOOTSTRAP.md](A100_BOOTSTRAP.md) — 환경·데이터 준비 (먼저 읽을 것)
- [PROTOCOL_COMPARISON.md](PROTOCOL_COMPARISON.md) — 현재까지의 ML vs DL 비교 결과 전문
- [METRICS.md](METRICS.md) — 지표 정의
- [RESULTS.md](RESULTS.md) — 기존(공식 split) 결과 전체. **주의: 프로토콜이 달라 직접 비교 불가**
