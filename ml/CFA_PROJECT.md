# CFA (Car Fluid Analyzer) — MVP Project Brief

STL 3D 형상 → 공기저항계수(Cd) 예측 → 개선 제안 → PDF 리포트.
CFD 시뮬레이션(수일)을 학습된 surrogate 모델(수초)로 대체한다.

---

## 0. 핵심 설계 결정 (읽고 시작할 것)

| 항목 | 결정 | 이유 |
|---|---|---|
| 학습 입력 | **포인트클라우드** (STL 아님) | 이미 전처리된 데이터 존재, Globus 불필요 |
| 모델 | **PointNet 회귀** | 가볍고 RTX 5080에서 학습 가능 |
| 제안 방식 | **반사실 탐색 (FFD 변형 → 재예측)** | surrogate가 빠르니 수십 번 재예측 가능 |
| LLM 역할 | **번역가 (분석가 아님)** | 숫자를 문장으로만 바꿈. 환각 금지 |
| STL의 위치 | **추론 시점에만** | 유저 업로드 → 샘플링 → 모델 입력 |

**⚠️ Globus / Harvard Dataverse는 이 프로젝트에서 사용하지 않는다.**
STL 메시 서브셋(수백 GB)은 RegDGCNN(메시 직접 입력) 학습 시에만 필요하며, 현재 스코프 밖이다.

---

## 1. 데이터 다운로드 (Phase 0)

### 1-1. 포인트클라우드 (메인, ~8.6GB)

```bash
mkdir -p data/subset_dir
BASE="https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B"

# 포인트클라우드 tar (약 8.6GB) — 이어받기 지원
wget -c "$BASE/DrivAer%2B%2B_Points.tar" -O data/DrivAer++_Points.tar
# wget 없으면:
# curl -L --fail -C - -o data/DrivAer++_Points.tar "$BASE/DrivAer%2B%2B_Points.tar"

tar -xf data/DrivAer++_Points.tar -C ./data

# tar 내부 구조: workspace/gino_data/14_DrivAer++/paddle_tensor
mv ./data/workspace/gino_data/14_DrivAer++/paddle_tensor \
   ./data/point_clouds_100k
rm -rf ./data/workspace
rm -f data/DrivAer++_Points.tar   # 압축 해제 후 삭제
```

### 1-2. 라벨 + split (작은 파일)

```bash
# Cd 라벨
curl -L --fail -o "data/DrivAerNetPlusPlus_Drag_8k.csv" \
  "$BASE/DrivAerNetPlusPlus_Drag_8k.csv"

# 공식 train/val/test split
for f in train_design_ids.txt val_design_ids.txt test_design_ids.txt; do
  curl -L --fail -o "data/subset_dir/$f" "$BASE/$f"
done
```

### 1-3. 최종 디렉토리 구조

```
data/
├── point_clouds_100k/              # 7,713개 .paddle_tensor (각 10만 점)
│   └── DrivAer_F_D_WM_WW_0001.paddle_tensor ...
├── DrivAerNetPlusPlus_Drag_8k.csv  # 컬럼: Design, Average Cd
└── subset_dir/
    ├── train_design_ids.txt
    ├── val_design_ids.txt
    └── test_design_ids.txt
```

### 1-4. 다운로드 후 즉시 검증

```python
# scripts/verify_data.py — 반드시 먼저 실행
# 1. .paddle_tensor 파일 개수 확인 (기대: 7,713)
# 2. 파일 하나 로드해서 shape 확인 (기대: (100000, 3) 또는 유사)
# 3. Drag CSV 로드 → Design 컬럼과 파일명 매칭 규칙 확인
# 4. CSV에는 8,000개 라벨, 포인트클라우드는 7,713개 → inner join 필수
# 5. split 파일의 ID 포맷이 파일명/CSV와 일치하는지 확인
```

**⚠️ 알려진 함정:** CSV 라벨은 ~8,000개인데 포인트클라우드는 7,713개만 존재.
반드시 **포인트클라우드가 실제로 있는 Design만** inner join 할 것.

### 1-5. .paddle_tensor 로딩

PaddlePaddle 텐서 포맷이다. 로딩 방법을 먼저 확인할 것:
- `paddle.load()` 로 열리는지 확인 (paddlepaddle 설치 필요할 수 있음)
- 안 되면 pickle/numpy 등으로 시도
- **한 번 로드에 성공하면, 전체를 `.npy`로 일괄 변환해두는 것을 권장** (학습 시 I/O 병목 제거)

```python
# scripts/convert_to_npy.py (권장)
# .paddle_tensor → float32 .npy 로 일괄 변환
# 동시에 8192점으로 미리 서브샘플링해서 저장하면 학습이 훨씬 빨라짐
```

---

## 2. 환경 셋업

### ⚠️ RTX 5080 = Blackwell 아키텍처 (sm_120)

일반 `pip install torch` 는 동작하지 않을 수 있다. Blackwell은 **CUDA 12.8+ 빌드**가 필요하다.

```bash
python -m venv .venv && source .venv/bin/activate

# Blackwell(sm_120) 지원 빌드 — 최신 index 확인 필요
pip install torch --index-url https://download.pytorch.org/whl/cu128

# 검증 (반드시 실행)
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.zeros(1).cuda())"
```

`sm_120 is not compatible` 에러가 나면 PyTorch 빌드가 Blackwell을 지원 안 하는 것 →
https://pytorch.org 에서 현재 권장 설치 명령 확인할 것.

```bash
pip install numpy pandas scikit-learn trimesh open3d matplotlib tqdm
pip install fastapi uvicorn python-multipart   # API
```

---

## 3. 프로젝트 구조

```
cfa/
├── data/                     # (gitignore)
├── scripts/
│   ├── verify_data.py        # Phase 0 검증
│   └── convert_to_npy.py     # .paddle_tensor → .npy 변환
├── src/
│   ├── dataset.py            # PyTorch Dataset (샘플링/정규화)
│   ├── model.py              # PointNet 회귀
│   ├── train.py
│   ├── evaluate.py           # R², MAE, MSE
│   ├── inference.py          # STL → 포인트클라우드 → Cd
│   ├── deform.py             # FFD 반사실 변형 ★핵심
│   └── report.py             # LLM + PDF
├── checkpoints/
└── CFA_PROJECT.md
```

---

## 4. 전처리 규약 (학습/추론 반드시 동일)

이 규약이 어긋나면 추론이 조용히 망가진다. **한 곳에 상수로 정의하고 학습·추론이 공유할 것.**

```python
# src/normalize.py
N_POINTS = 8192          # 100k → 서브샘플링 (16GB VRAM 고려)
                         # 나중에 16384로 올려서 성능 비교

def normalize(points):   # (N, 3) float32
    points = points - points.mean(axis=0)        # 중심 정렬
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / scale                       # 유닛 스피어
    return points
```

**중요:**
- 100k 점을 그대로 학습에 쓰면 16GB VRAM으로는 배치가 거의 안 나온다 → 8192점 권장
- 학습 시엔 매 epoch 랜덤 서브샘플 (augmentation 효과)
- 추론 시엔 고정 시드로 서브샘플 (재현성)
- **좌표축 규약을 데이터에서 직접 확인할 것** (x=주행방향인지 등). 추론 시 STL 축이 다르면 정렬 필요.

---

## 5. 개발 단계

### Phase 0 — 데이터 (지금)
- [ ] 다운로드 (§1)
- [ ] `verify_data.py` 통과: 7,713개 매칭 확인
- [ ] `.npy` 변환 (8192점 서브샘플 저장)

### Phase 1 — 학습
- [ ] Dataset: 공식 split 사용, 정규화 규약 적용
- [ ] PointNet 회귀 (출력 1차원, Loss: MSE 또는 L1)
- [ ] 평가: **테스트 R² ≥ 0.6 이 1차 목표**
  - 참고: 문헌상 PointNet은 DrivAerNet++에서 R² ≈ 0.64
  - 이 숫자에 근접하면 파이프라인이 옳다는 신호
- [ ] 체크포인트 + 정규화 상수 함께 저장

### Phase 2 — 추론
- [ ] `inference.py`: STL → `trimesh.load().sample(100000)` → 정규화 → 예측
- [ ] 학습 데이터 STL 몇 개로 검증 (예측값이 CSV 라벨과 비슷한지)
  - **여기서 STL 샘플 몇 개만 있으면 됨. 전체 데이터셋 불필요.**

### Phase 3 — 반사실 제안 ★핵심 차별점
- [ ] `deform.py`: 포인트클라우드 변형 함수들
  - `roof_height(pc, ratio)` — 특정 높이 이상 점들의 z 스케일 (경계 스무딩)
  - `rear_taper(pc, ratio)` — x 위치에 따라 y를 점진적으로 좁힘
  - `length_ratio(pc, ratio)`, `width_ratio(pc, ratio)`
  - `rake(pc, angle)` — shear
- [ ] **변형 폭을 데이터셋 실측 범위로 클램프 (OOD 방지)**
  - 루프 높이: 1.35 ~ 1.73 m
  - 전장: 4.40 ~ 5.16 m
  - 이 범위 밖은 예측 신뢰 불가 → UI에 "within dataset range" 표시
- [ ] 20~50개 변형 → 배치 재예측 → ΔCd 랭킹
- [ ] ⚠️ 정규화가 변형을 지우지 않는지 검증: **비율을 바꾸는 변형만 사용**
      (전체 균일 스케일은 정규화로 상쇄되어 무의미)

### Phase 4 — 리포트
- [ ] LLM 입력 JSON 고정:
```json
{
  "baseline_cd": 0.312,
  "counterfactuals": [
    {"change": "roof height -3%", "new_cd": 0.298, "delta": -0.014}
  ]
}
```
- [ ] **LLM 규칙 (프롬프트에 명시):**
  - 주어진 숫자 외 어떤 수치도 생성 금지
  - 근거 없는 발견(finding) 생성 금지 ("C필러에 문제" 같은 것)
  - 메커니즘 설명은 허용 (교과서 물리)
- [ ] PDF: 형상 썸네일 + baseline Cd + 제안 표 + 문장
- [ ] **PDF 하단 필수 문구:**
      `AI surrogate estimate — not a substitute for CFD validation.`

### Phase 5 — 앱 (P1, MVP 이후)
로그인 / 히스토리 / 디자인 스타일 프로파일 / 팀 대시보드

---

## 6. MVP 완료 정의

```
STL 업로드 → Cd 예측 → 개선 제안 3개 → PDF 다운로드
```
이 루프 하나가 끝까지 도는 것. 나머지는 전부 그 다음.

---

## 7. 데이터 출처

- **포인트클라우드/라벨/split:** PaddleScience DNNFluid-Car (DrivAerNet++ 미러)
- **원본:** DrivAerNet++, Elrefaie et al., 2024 — CC BY-NC 4.0 (비상업)
- **모델 참고:** PointNet (Qi et al., 2017)
