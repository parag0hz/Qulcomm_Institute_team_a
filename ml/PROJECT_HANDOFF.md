# DrivAerNet++ 포인트클라우드 → 항력계수(Cd) 예측 프로젝트 — 핸드오프

> 이 문서 하나로 새 서버의 Claude가 프로젝트를 이어받을 수 있도록 정리한 자립형 핸드오프.
> 작성 기준일: 2026-07(이전 세션 요약). 이전 작업 환경: macOS Apple M1 Pro. 새 환경은 GPU 서버(Linux 가정).

---

## 0. 한 줄 요약
아이폰 LiDAR로 차량을 스캔 → 포인트클라우드 추출 → (GPU 추론 서버로 전송) → **항력계수 Cd를 예측**하는 앱. 해커톤 주제.
**앱 UI는 "돌아가는 데모"면 충분**하고, **진짜 승부처는 "실제(폰스캔 같은 부분·노이즈) 포인트클라우드 → Cd" 모델을 만들고 고도화**하는 것.

---

## 1. 프로젝트 방향 / 결정사항
- **목표 지표**: 클린 정확도가 아니라 **폰스캔(부분·노이즈·희소) 조건에서의 Cd 예측 정확도**.
- **앱**: 데모용. 서버 추론이라 온디바이스 제약 없음 → **무거운 point-cloud 딥러닝으로 정면승부 가능**.
- **핵심 원칙 2가지**
  1. 겪을 열화(한쪽만 스캔·노이즈·희소)를 **학습에 주입**(도메인 랜덤화)한다.
  2. **⚠️ 스케일을 죽이지 마라**: Cd는 무차원이지만 이 데이터셋은 **절대 치수(특히 높이)가 강한 신호**(높이 r=0.82). PointNet 관행대로 클라우드를 unit-scale 정규화하면 이 신호를 버림. LiDAR는 미터 스케일을 주므로 **미터 스케일 유지** + 전역 치수를 보조 입력으로.
- **알려진 한계(스코프)**: 학습셋은 DrivAer 세단 계열 변형뿐 → 실제 SUV/트럭은 **OOD(학습분포 밖)**. 실차 절대 Cd 측정은 불가. → "상대 스코어/설계 피드백 도구"로 포지셔닝 + **OOD 신뢰도 플래그**로 정직하게.

---

## 2. 환경 & 함정
- 이전 환경: macOS M1 Pro, Python(anaconda3), numpy 2.4, scipy 1.17, matplotlib 3.10, **torch 2.10 (MPS 사용가능)**. **sklearn/xgboost 미설치**, **wget 미설치**(macOS).
- **함정 1 — wget**: macOS엔 wget이 없었음. 원본 다운로드 스크립트가 전부 실패했던 원인. Linux 서버엔 보통 wget 있음. 아래 3장에 curl 버전도 제공.
- **함정 2 — pickle 보안**: `.paddle_tensor`는 pickle이라 `pickle.load`는 임의 코드 실행 위험으로 **자동 차단**됨. **`pickletools` + `np.frombuffer`로 코드 실행 없이 로드**하는 안전 로더를 쓴다(4장 코드).
- **함정 3 — numpy 2.x**: `ndarray.ptp()` 메서드 제거됨 → `np.ptp(arr, axis=...)` 사용. pickle 내부가 `numpy.core.multiarray`를 참조하지만 안전 로더는 무관.
- **함정 4 — 한글 폰트(matplotlib)**: macOS는 `Apple SD Gothic Neo`. Linux는 `NanumGothic` 등 설치 필요(`rcParams["font.family"]`, `axes.unicode_minus=False`).
- 새 서버: GPU면 torch device를 `cuda`로(이전엔 `mps`).

---

## 3. 데이터셋 다운로드
**데이터셋 이름: DrivAerNet++ (DrivAer++)** — DrivAerNet의 확장판. 독일 TU München의 DrivAer 표준 차체를 파라미터 변형해 생성한 자동차 공력 데이터셋. 출처 미러: PaddleScience `DNNFluid-Car` (Baidu `bcebos.com`).

```bash
mkdir -p data/subset_dir
BASE="https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B"

# (A) 100k 포인트클라우드 메인 tar — 약 8.6GB (9,264,496,640 B)
#   wget 있으면:
wget -c "$BASE/DrivAer%2B%2B_Points.tar" -O data/DrivAer++_Points.tar
#   wget 없으면 curl (이어받기 -C -):
#   curl -L --fail -C - -o data/DrivAer++_Points.tar "$BASE/DrivAer%2B%2B_Points.tar"

tar -xf data/DrivAer++_Points.tar -C ./data
# tar 내부 구조: workspace/gino_data/14_DrivAer++/paddle_tensor
mv ./data/workspace/gino_data/14_DrivAer++/paddle_tensor \
   ./data/DrivAerNetPlusPlus_Processed_Point_Clouds_100k_paddle
rm -rf ./data/workspace
rm -f data/DrivAer++_Points.tar   # 압축 해제 후 8.6GB tar 삭제

# (B) 라벨 + split (작은 파일들)
for f in DrivAerNetPlusPlus_Drag_8k.csv; do curl -L --fail -o "data/$f" "$BASE/$f"; done
for f in train_design_ids.txt val_design_ids.txt test_design_ids.txt; do
  curl -L --fail -o "data/subset_dir/$f" "$BASE/$f"; done
```

**최종 디렉토리 구조**
```
data/
├── DrivAerNetPlusPlus_Processed_Point_Clouds_100k_paddle/   # 7,713개 .paddle_tensor (총 9.26GB)
│   └── DrivAer_F_D_WM_WW_0001.paddle_tensor ...
├── DrivAerNetPlusPlus_Drag_8k.csv                            # Cd 라벨 (컬럼: Design, Average Cd)
└── subset_dir/{train,val,test}_design_ids.txt               # 공식 split
```

---

## 4. 데이터 포맷 & 안전 로더
- 각 `.paddle_tensor` = **Python pickle**. 내용은 튜플 `(name_str, numpy.ndarray)`, 배열 shape **`(100000, 3)` float32** = 10만 점 × (x,y,z).
- **좌표계(전 차량 공통)**: 지면 **z=0**, **y는 0 중심(좌우대칭축)**, x는 앞→뒤. 전역 범위 ≈ x∈[-1.15, 4.12], y∈[-1.19, 1.19], z∈[0, 1.76] (단위 m). y가 ±1.0 넘는 건 사이드미러.
- 파일명 규칙: `{variant}_{index}`. variant 예: `DrivAer_F_D_WM_WW`(패스트백 상세), `E_S_WWC_WM`(에스테이트), `N_S_WW_WM`(노치백). F/E/N = Fastback/Estate/Notchback.

**⚠️ CSV `Design` id ↔ 파일명 매칭 주의**: `DrivAer_F_D...`는 CSV도 zero-padded(`_0001`), 그러나 `E/F/N_S...`는 패딩 없음(`_1`). → **(prefix, int)로 정규화해 매칭**.

```python
# safe loader — pickle.load 금지, pickletools로 opcode만 읽어 raw 버퍼 복원 (코드 실행 없음)
import pickletools, numpy as np
def safe_load_ndarray(path):
    with open(path, "rb") as f: data = f.read()
    ops = [(op.name, arg) for op, arg, pos in pickletools.genops(data)]
    raw = max((a for n,a in ops if isinstance(a,(bytes,bytearray))), key=len)  # 최대 bytes = 배열버퍼
    typestr = None
    for i,(n,a) in enumerate(ops):
        if a == "dtype":
            for n2,a2 in ops[i+1:i+6]:
                if isinstance(a2,str):
                    try: np.dtype(a2); typestr=a2; break
                    except TypeError: pass
            if typestr: break
    if typestr is None:
        for its,ts in ((4,"<f4"),(8,"<f8")):
            if len(raw)%its==0 and (len(raw)//its)%3==0: typestr=ts; break
    dt = np.dtype(typestr); n = len(raw)//dt.itemsize
    return np.frombuffer(raw, dtype=dt).reshape(n//3, 3).astype(np.float64), typestr

import re, csv
def norm_id(name):
    m = re.match(r"^(.*)_(\d+)$", str(name).replace(".paddle_tensor",""))
    return (m.group(1), int(m.group(2))) if m else (name,-1)
def load_cd(csv_path="data/DrivAerNetPlusPlus_Drag_8k.csv"):
    return {norm_id(r["Design"]): float(r["Average Cd"]) for r in csv.DictReader(open(csv_path))}
```

---

## 5. EDA 결과 (검증 완료, 데이터 품질 매우 깨끗)
- **파일**: 7,713개, 전부 `.paddle_tensor`, 9.26GB. 파일크기 1,200,212~1,200,215B(3바이트 차이 = 텐서 이름 문자열 길이 차, 데이터는 전부 100k×3 동일). **라벨↔파일 누락 0**.
- **Cd 분포**: min **0.201** ~ max **0.383**, 평균 0.284, std 0.037, 중앙값 0.283, 5~95% 구간 0.226~0.349. **IQR 이상치 0개**. 최저 `F_S_WWC_WM_407`(0.201), 최고 `DrivAer_F_D_WM_WW_1587`(0.383).
- **차종별 Cd** (피치 소재):
  | 차종 | 대수 | Cd 평균±std |
  |---|---|---|
  | Notchback | 1,124 | **0.247 ± 0.019** (최저) |
  | Estate | 1,366 | 0.273 ± 0.019 |
  | Fastback | 5,223 | **0.296 ± 0.038** (최고·편차 큼) |
  → **클래스 불균형 심함(Fastback 68%)**. 모델링 시 샘플 가중치/오버샘플 + **차종별 평가 필수**.
- **기하**: 길이 4.32~5.10m, 폭 1.67~2.15m, 높이 1.30~1.71m (전부 현실적). 좌표계 z_min=0.000±0.000, y중심=0.000±0.000 → 완벽 정렬.
- **split**: train 5398 / val 1157 / test 1158 (70/15/15). 세 split의 Cd 분포 거의 동일(평균 0.284~0.285) → **평가 신뢰 가능**.

---

## 6. 지금까지의 실험 결과 (Tier-1: 거친 치수 회귀)
**형상 지표 vs Cd 상관(500대)**: 높이 **r=+0.82**(최강), 길이 +0.68, 전면적 +0.60, H/L +0.60, 후미하강 +0.58, 폭 −0.33.

**Tier-1 모델**: 로버스트 치수 6개 → 선형+2차항 회귀. **클린 test MAE 4.4%, R²=0.82** (높이 1개만도 R²=0.69). 저장됨: `data/cd_model.npz` (키: coef, mu, sd, cmean).

**폰스캔 열화 강건성 실험 (핵심 발견)** — 같은 모델, 입력만 열화:
| 열화 | v1(max/hull 피처) R² | **v2(퍼센타일 피처) R²** |
|---|---|---|
| clean | 0.82 | 0.81 |
| noise(1.5cm) | 0.40 | **0.77** |
| sparse(4k) | 0.79 | 0.79 |
| glass dropout | 0.82 | 0.81 |
| **one-side**(한쪽만) | −5.5 | **0.08** |
| **phone**(전부합침) | −3.9 | **0.33** |
| **phone+mirror**(좌우대칭복원) | 0.51 | **0.80 (MAE 4.7%)** |

**결론**:
1. **퍼센타일(로버스트) 피처** → 노이즈 문제 해결.
2. **좌우대칭 복원(mirror)** → 한쪽 스캔 붕괴 해결. 물리적으로도 타당(한쪽만 보면 차가 작아 보여 Cd 과소평가 → 대칭복원이 되돌림).
3. **phone+mirror에서 클린과 거의 동등(MAE 4.7% vs 4.4%)**. → "로버스트 치수 + 대칭복원"이 폰스캔에 강건한 baseline. 단, 일부 형상은 베이스 모델 자체가 약함(예 `F_S_WWS_WM_050` 클린부터 12% 오차) → 여기가 point-cloud 딥러닝으로 넘어갈 지점.

```python
# 재사용 핵심 함수 (Tier-1 파이프라인)
from scipy.spatial import ConvexHull
def robust_feats(p):  # 노이즈 강건 거친 치수 6개 [H,L,W,FA,H/L,drop]
    x,y,z = p[:,0],p[:,1],p[:,2]
    H = np.percentile(z,99.5); L = np.percentile(x,99.5)-np.percentile(x,0.5)
    W = np.percentile(y,99.5)-np.percentile(y,0.5)
    ylo,yhi=np.percentile(y,[0.5,99.5]); zlo,zhi=np.percentile(z,[0.5,99.5])
    m=(y>=ylo)&(y<=yhi)&(z>=zlo)&(z<=zhi)
    try: FA=ConvexHull(p[:,1:3][m]).volume
    except Exception: FA=np.nan
    HL=H/L if L>1e-6 else np.nan
    xlo,xhi=np.percentile(x,[0.5,99.5])
    mid=p[(x>xlo+0.4*(xhi-xlo))&(x<xlo+0.6*(xhi-xlo))]; rear=p[x>xhi-0.10*(xhi-xlo)]
    drop=np.percentile(mid[:,2],99)-np.percentile(rear[:,2],99) if len(mid)>5 and len(rear)>5 else np.nan
    return np.array([H,L,W,FA,HL,drop])

def degrade_phone(p, rng):  # 아이폰 스캔 모사: 노이즈+한쪽+유리소실+희소
    q = p + rng.normal(0,0.015,p.shape)
    q = q[q[:,1]>-0.15]
    high=q[:,2]>1.0; q=q[~high|(rng.random(len(q))<0.4)]
    if len(q)>5000: q=q[rng.choice(len(q),5000,replace=False)]
    return q
def mirror_complete(q):  # 좌우대칭 복원
    m=q.copy(); m[:,1]*=-1; return np.concatenate([q,m],0)
```

---

## 7. 모델 전략 / 고도화 로드맵 (다음 작업의 핵심)
**베이스라인 사다리** (각 단계는 동일 평가 격자로 비교, 항상 돌아가는 모델 유지):
| 단계 | 모델 | 목적 | 상태 |
|---|---|---|---|
| R0 | 로버스트 치수→선형/2차 | 바닥선 | ✅ MAE 4.4% |
| R1 | 치수→GBM/MLP | 테이블 상한 | 예정 |
| R2 | **PointNet**(raw점, 증강X) | raw점 클린 상한 | **다음 진입점** |
| R3 | **PointNet + 도메인랜덤화** | **앱 실사용 모델** | 진짜 목표 |
| R4 | PointNet++/DGCNN/PointNeXt | 정확도 상한 | 여유시 |
| R5 | 불확실성·멀티태스크·앙상블·TTA | 고도화 프론티어 | 차별화 |

**입력·전처리**: 정준화(지면z=0·대칭축y=0·PCA 주축x) → **FPS 2048점** → **미터 스케일 유지**(+전역치수 late-fusion) → 타깃 Cd 표준화 → Huber loss.

**강건성 엔진(고도화 본체)**:
- ① **현실적 부분뷰 합성**: 조잡한 `y>-0.15` 대신 **Hidden Point Removal(HPR)** 로 시점별 실제 가림 생성(차 주위 걷는 궤적 시뮬).
- ② **온-더-플라이 도메인 랜덤화**: 매 배치 랜덤으로 HPR가림·노이즈(0~2cm)·희소(2k~20k)·상단dropout·소회전(±5°)·스케일지터(±3%)·부분크롭.
- ③ 대칭복원을 **학습·추론 양쪽 일관** 적용.

**sim-to-real 갭 정면돌파(가장 중요)**: 실스캔 정답 Cd가 없음 → 제조사 공개 Cd 아는 차 **5~10대만 실제 폰스캔해 검증셋**(학습 아님)으로. "실차 오차 X%"라는 단 하나의 실측 근거가 발표에 결정적.

**고도화 프론티어**: 불확실성/OOD(MC-dropout·앙상블 → "학습분포 밖" 자동감지 = 앱 신뢰도 플래그), 멀티태스크(Cd+전면적+차종 동시), 물리결합(Cd·A 구조), TTA(미러·회전 평균), 앙상블(치수-GBM ⊕ PointNet: 실패모드 상보).

**평가 하네스(발표 핵심표)**: `[clean/noise/sparse/HPR가림/phone/phone+mirror] × [Fastback/Estate/Notchback]` 격자로 MAE·R²(+불확실성 캘리브레이션). 클래스 불균형 → 차종별 필수 + 학습 샘플 가중치.

**권장 실행 순서**: ① 평가 하네스 먼저(R0/R1 꽂아 기준선 고정) → ② R2 PointNet 클린 베이스라인("raw점이 치수모델을 이기나" 확인) → ③ R3 도메인랜덤화(HPR) = 앱 실사용 모델 → ④ R4/R5 임팩트순.

---

## 8. 이전 세션에서 만든 스크립트 (참고용, 새 서버엔 없음 — 위 코드로 재생성)
- `viz_pointcloud.py` — 안전 로더 + 단일 차량 3D 시각화(측면/등각)
- `viz_grid.py` — 여러 차량 격자 시각화
- `viz_compare.py` — 옆/정면 실루엣 오버레이(같은 스케일)
- `viz_analyze.py` — 뒷부분확대(F/E/N) / 극단샘플 / Δ히트맵(KDTree)
- `viz_cd_corr.py` — 형상지표 vs Cd 상관 산점도
- `train_and_degrade.py`(v1) / `train_and_degrade_v2.py`(v2 로버스트) — 학습+열화 강건성 테스트
- `cd_common.py` — 공유 모듈(로더/피처/열화/미러/예측)
- `train_model.py` — Tier-1 모델 학습 → `data/cd_model.npz`
- `demo.py` — 킬러 데모(원본→폰스캔→대칭복원→Cd, 4패널)
- `eda.py` — 종합 EDA

---

## 9. 즉시 다음 할 일 (새 서버 진입점)
1. 3장으로 데이터 재다운로드 + 4장 안전 로더 검증(파일 하나 로드해 shape `(100000,3)` 확인).
2. **평가 하네스** 구축: 공식 split 로드, `[열화 조건] × [차종]` 격자로 MAE/R² 출력하는 함수. R0(치수 회귀) 꽂아 기준선 재현(클린 MAE 4.4% 나오는지).
3. **R2 PointNet(torch, device=cuda)** 베이스라인: FPS 2048, 미터스케일 유지, Huber, val 조기종료. 클린에서 치수모델(R²0.82) 넘는지 확인.
4. **R3**: HPR 기반 도메인 랜덤화 증강을 학습 루프에 주입 → 평가 격자로 phone/phone+mirror 성능 측정.

**주의**: 무엇을 하든 항상 평가 격자로 수치화해서 개선을 증명할 것. 클린 R²만 보지 말고 **phone+mirror·차종별**을 봐야 함.
