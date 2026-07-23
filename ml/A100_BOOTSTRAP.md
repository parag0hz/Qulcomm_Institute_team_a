# A100_BOOTSTRAP.md — 새 A100 서버에서 이 프로젝트 시작하기

이 파일 하나만 읽고 **빈 A100 서버**(코드·데이터·conda 없음)에서 이 프로젝트(포인트클라우드 → 자동차 Cd 예측, DrivAerNet++)를 처음부터 돌릴 수 있게 만든 부트스트랩 문서다. 새 서버에서 Claude Code를 켜고 이 파일을 주면 된다.

- **원본 서버**: `kwy00@192.168.0.105:/home/kwy00/qi` (RTX 5080 16GB, Blackwell sm_120)
- **대상 서버**: A100 (Ampere **sm_80**, 40 또는 80GB VRAM) — VRAM이 커서 5080에서 OOM 나던 실험이 여기서 열린다 (§6, 이 서버로 옮기는 진짜 이유)
- 근거 문서: 원본 저장소의 [CLAUDE.md](CLAUDE.md), [RESULTS.md](RESULTS.md), [DATA_SUMMARY.md](DATA_SUMMARY.md), [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md), [METRICS.md](METRICS.md). 이 문서와 그것들이 충돌하면 **이 문서가 이긴다**(A100 기준으로 갱신·검증됨).

---

## TL;DR — 딱 3단계

```bash
# ① 코드 + 데이터를 원본 서버에서 통째로 복사 (전처리까지 끝난 상태로 옴)
#    tar(재압축해제 불필요)와 100k 캐시(13초면 재생성)는 제외 → 약 9GB만 전송
rsync -avhP --exclude='DrivAer++_Points.tar' --exclude='pc100k_f32.dat' \
  --exclude='.git' --exclude='wandb/' --exclude='__pycache__' \
  kwy00@192.168.0.105:/home/kwy00/qi/  ~/qi/

# ② 환경 — A100(sm_80)은 특수 torch 빌드 불필요. CLAUDE.md의 cu130 지시는 5080 전용이니 무시.
conda create -y -n qi python=3.12 && conda activate qi     # ⚠ 3.13 금지(open3d wheel 없음)
pip install torch==2.13.0                                   # 표준 wheel이 sm_80 지원
pip install numpy==2.5.1 scipy scikit-learn xgboost pandas matplotlib open3d wandb tqdm

# ③ repo를 ~/qi가 아닌 다른 경로에 뒀으면 데이터 루트만 알려주고, 확인 후 학습
cd ~/qi && export QI_DATA=$PWD/data                         # 데이터가 여기 없으면 실제 경로로
python -c "import torch,cd_common as C; C.check_integrity(); print(torch.cuda.get_device_name(0))"
python train_r2.py --backbone pointnet --dims 0 --npoints 2048    # 재현: clean test R² ≈ 0.968
```

이게 끝이다. 아래는 각 단계 상세와 **A100에서 먼저 돌릴 실험(§6)**.

---

## 1. 코드 + 데이터 확보

### 경로 A — 원본 서버에서 rsync (권장, 전처리 스킵)

원본 서버 `data/`에는 **압축 해제 + 전처리(FPS 캐시)까지 끝나** 있다. 그대로 복사하면 다운로드·압축해제·전처리·ID매칭함정을 전부 건너뛴다. TL;DR ①이 그것이다. 전송량 **약 9GB**(압축 해제된 포인트클라우드 8.7G + `fps2048.npz` 182M + CSV들). `fps2048.npz`가 딸려오므로 `precompute_fps.py`도 안 돌려도 된다.

- `192.168.0.105`는 원본의 **LAN IP**. A100이 다른 네트워크면 실제 접속 호스트/포트로 (`rsync -e "ssh -p PORT" user@host:...`).
- **100k 원본 학습(§6)** 을 하려면 `pc100k_f32.dat`가 필요한데, 복사(8.7G) 대신 도착 후 한 줄로 재생성(측정 13초): `python scripts/train_100k.py --build-cache`. (tar까지 포함하려면 `--exclude='DrivAer++_Points.tar'`만 빼면 되지만, 압축 해제본이 이미 오므로 불필요.)

### 경로 B — 원본에서 재다운로드 (rsync 불가할 때)

**코드** — 이 저장소는 자체 git 리포(branch `main`)지만 **커밋도 원격도 없다**(§4.2). 그래서 코드 이전은 (선호) 코드만 rsync `rsync -av --exclude='data/' --exclude='.git' --exclude='wandb/' kwy00@192.168.0.105:/home/kwy00/qi/ ~/qi/`, 또는 원본에서 커밋 후 새 원격에 push→clone.

**데이터** — 두 출처에서 받는다(⚠ 포인트클라우드는 bcebos, 파라메트릭 CSV는 GitHub):

```bash
cd ~/qi && mkdir -p data/subset_dir
BASE="https://dataset.bj.bcebos.com/PaddleScience/DNNFluid-Car/DrivAer%2B%2B"

# (1) 100k 포인트클라우드 tar (8.63 GiB, 이어받기 -c)
wget -c "$BASE/DrivAer%2B%2B_Points.tar" -O data/DrivAer++_Points.tar
tar -xf data/DrivAer++_Points.tar -C ./data
# ⚠ 반드시 이 디렉토리명으로 rename (cd_common.py가 이 이름을 기대. CFA_PROJECT.md의 point_clouds_100k는 틀림)
mv ./data/workspace/gino_data/14_DrivAer++/paddle_tensor \
   ./data/DrivAerNetPlusPlus_Processed_Point_Clouds_100k_paddle
rm -rf ./data/workspace

# (2) Cd 라벨 + 공식 split (bcebos)
curl -L --fail -o data/DrivAerNetPlusPlus_Drag_8k.csv "$BASE/DrivAerNetPlusPlus_Drag_8k.csv"
for f in train_design_ids.txt val_design_ids.txt test_design_ids.txt; do
  curl -L --fail -o "data/subset_dir/$f" "$BASE/$f"; done

# (3) 파라메트릭 CSV (ML 트랙용) — ⚠ bcebos에 없음. DrivAerNet GitHub에서.
curl -L --fail -o data/DrivAerNet_ParametricData.csv \
  "https://raw.githubusercontent.com/Mohamedelrefaie/DrivAerNet/main/ParametricModels/DrivAerNet_ParametricData.csv"

# (4) FPS 캐시 재생성 (GPU, 수 분). 포인트클라우드 트랙의 전제.
python precompute_fps.py
```

### 확보 후 기대 구조 & 검증

```
data/   (rsync 시 27G, tar·pc100k 제외 시 ~9G)
├── DrivAerNetPlusPlus_Processed_Point_Clouds_100k_paddle/  # 7,713 × .paddle_tensor (8.7 GiB)
├── DrivAerNetPlusPlus_Drag_8k.csv                          # Cd 라벨 (헤더+7,713행)
├── DrivAerNet_ParametricData.csv                           # ML 트랙 (헤더+4,165행, 32열)  ← GitHub
├── subset_dir/{train,val,test}_design_ids.txt              # 5398 / 1157 / 1158
├── fps2048.npz                                             # FPS-2048 캐시 (train_r2 필수, 182 MiB)
└── pc100k_f32.dat + pc100k_meta.json                       # 100k 캐시 (§6, 재생성 13초)
```
```bash
python scripts/verify_data.py    # 또는:
python -c "import cd_common as C; C.check_integrity(); fi=C.file_index(); k=sorted(fi)[0]; \
  p=C.safe_load_ndarray(fi[k]); print('OK', k, p.shape, p.dtype)"   # OK (...,) (100000, 3) float32
```

---

## 2. 환경 (A100 맞춤 — 5080과 다른 점)

> **핵심 차이(단 하나).** 원본(RTX 5080)은 Blackwell **sm_120**이라 **cu130** 전용 torch가 필수였다 — [CLAUDE.md](CLAUDE.md)와 [requirements.txt](requirements.txt)의 `--extra-index-url .../cu130`, `torch==2.13.0+cu130`는 **5080 전용**이다. A100은 Ampere **sm_80**이고 이건 **모든 표준 PyTorch CUDA wheel(cu118+)이 지원**한다. 그냥 `pip install torch`로 된다 (증거: 5080의 cu130 빌드조차 `get_arch_list()`에 `sm_80` 포함).

```bash
conda create -y -n qi python=3.12       # ⚠ 3.13 금지 — open3d가 3.13 wheel을 안 냄 (Python 핀 이유)
conda activate qi
pip install torch==2.13.0               # 표준 wheel. cu124/cu128 인덱스도 sm_80 OK. cu130 인덱스는 쓰지 말 것
pip install numpy==2.5.1 scipy==1.18.0 scikit-learn==1.9.0 xgboost==3.3.0 \
            pandas matplotlib open3d==0.19.0 wandb tqdm
# 선택: torch-geometric==2.8.0 (설치돼 있으나 코드가 import 안 함 — PyG 계열 확장 때만)
# 선택(ML/크로스모달 스크립트): autogluon lightgbm seaborn (핵심 R0~R3엔 불필요)
```

정상 확인:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"
#   기대: 'A100-SXM4-80GB'(또는 40GB), arch_list에 'sm_80' 포함
```

[requirements.txt](requirements.txt)로 재현하려면 cu130 줄만 걷어내고 써라: `grep -vE '^(--extra-index-url|torch==|triton==|nvidia-)' requirements.txt > req-a100.txt && pip install -r req-a100.txt` (torch는 위에서 먼저 설치).

---

## 3. 파이프라인 (재현 순서)

```bash
conda activate qi && cd ~/qi
python scripts/verify_data.py       # (선택) 데이터 무결성 먼저
python precompute_fps.py            # rsync로 fps2048.npz 왔으면 스킵. 100k→FPS-2048 캐시, GPU
python train_r0.py                  # R0: 치수6 → 회귀. 차종×열화 격자 출력 (fps 캐시 불필요)
python train_r2.py --backbone pointnet --dims 0 --npoints 2048    # R2 메인. clean R² ≈ 0.968
                                    # 옵션: --backbone dgcnn|regdgcnn|triplane|mlp  --dims 1  --scale unit  --attn se|cbam|pool|sa
python eval_metrics.py outputs/pointnet_n2048_dims0_meter_pred.npz   # 3지표 + 차종/쌍둥이 분해
python summarize_r2.py              # outputs/*.json → R0 대비 비교표
python check_leakage.py            # voxel 1-NN 누수 프로브 (CPU)
python scripts/holdout_eval.py --test-body Estate       # 일반화 홀드아웃
```

- **wandb**: 프로젝트 `cfa`. 처음 `wandb login`(또는 각 스크립트 `--wandb 0`으로 끔). `scripts/wandb_backfill.py`는 **한 번만**(재실행 시 런 중복).
- **tmux**: 긴 학습은 tmux 안에서 (SSH 끊겨도 유지). `./run_rerun.sh`는 train_r2+평가를 `outputs/rerun.log`로.
- 지표 정의는 [METRICS.md](METRICS.md).

---

## 4. 반드시 지켜야 할 함정 & 이식 주의점

### 4.1 데이터·코드 함정 (다시 밟지 말 것)

1. **`.paddle_tensor`를 `pickle.load`/`paddle.load` 금지** — 서드파티 미러의 raw pickle이라 임의 코드 실행(RCE). `cd_common.safe_load_ndarray`만 사용(opcode 실행 없이 버퍼만 복원, `(100000,3)` float32). CFA_PROJECT.md §1-5의 `paddle.load()` 제안은 따르지 말 것.
2. **ID 매칭은 `norm_id`/`file_index()`만** — 순진한 CSV↔파일 조인은 **정확히 609개 설계를 조용히** 버린다(`E/F/N_S_*`의 1–99 슬라이스). 파일명을 ID로 재구성하지 말고 `file_index()`로 조회, 파이프라인 첫 줄에서 `check_integrity()`.
3. **numpy 2.x** — `ndarray.ptp()` 제거됨 → `np.ptp(arr, axis=...)`.
4. **미터 스케일 유지 (unit-sphere 정규화 금지)** — 절대 차높이가 최강 예측자(r≈+0.827). 재스케일하면 이 신호가 날아가 R² 0.968 → 문헌 ~0.64. 전역 치수는 late-fusion, T-Net도 끔. (CFA_PROJECT.md §4와 정면 충돌 — 이 저장소가 이긴다.)
5. **평가는 차종별 분해 필수** — Fastback 68%라 전역 평균이 실패를 숨긴다. 3지표(R²/MAE drag counts/순위 정확도) 동시 보고. R0는 사실상 Fastback 전용(Estate R²−0.519, Notchback −0.151) — point-cloud 모델이 이겨야 할 표적.
6. **미러/`|y|>1.0` 고정 임계 금지** — 넓은 차 12%에서 차체 절단. 차별로 `p99.5(|y|)`로 세그먼트.
7. **좌표계는 이미 canonical**(바닥 z=0, 대칭축 y=0, 미터). 재정렬 금지.

### 4.2 새 머신 이식 주의점 (경로 하드코딩)

- **git에 커밋이 0개다.** 실측: `git log` → "does not have any commits yet", 원격 없음, 전 파일 untracked. 방법 B(push→clone)를 쓰려면 **먼저 `git add -A && git commit`** 해야 한다. [.gitignore](.gitignore)가 `data/ outputs/ external/ *.npz *.pt *.tar`를 제외하므로 git으론 **코드만** 가고 데이터·캐시는 안 따라온다 → 그래서 rsync(경로 A)가 대개 낫다.
- **`QI_DATA` 환경변수로 데이터 루트를 잡는다.** [cd_common.py](cd_common.py)는 `DATA_DIR = os.environ.get("QI_DATA", "/home/kwy00/qi/data")`. repo를 다른 경로에 두면 `export QI_DATA=/abs/path/to/data` 먼저.
- **일부 스크립트는 `/home/kwy00/qi`를 하드코딩**한다([precompute_fps.py](precompute_fps.py), [scripts/train_100k.py](scripts/train_100k.py)의 npz 경로, `run_rerun.sh`의 `PY=/home/kwy00/anaconda3/envs/qi/bin/python`). 새 홈이 다르면 이 상수들을 실제 경로로 치환하거나, repo를 `~/qi` + conda env `qi`로 맞춰라. (`QI_DATA`는 `cd_common` 경유 접근만 바꾸고, 스크립트의 하드코딩 npz 경로는 안 바꾼다.)
- **`external/DrivAerNet`** 은 `regdgcnn` 백본·`scripts/train_regdgcnn.py`에서만 지연 import된다(핵심 파이프라인은 없어도 됨). 필요 시 `git clone https://github.com/Mohamedelrefaie/DrivAerNet ~/qi/external/DrivAerNet`. (rsync 경로 A는 `.git`만 제외하므로 `external/`도 함께 온다.)

---

## 5. 파일 인벤토리 (코드)

| 파일 | 역할 |
|---|---|
| [cd_common.py](cd_common.py) | 안전 로더·ID 정규화·`file_index`·`check_integrity`·로버스트 피처 6종·모든 열화 연산자. **모든 데이터 접근은 여기 경유** |
| [models_pc.py](models_pc.py) | 백본 `BACKBONES={pointnet,dgcnn,mlp,regdgcnn,triplane}`. `regdgcnn`만 `external/` 지연 import |
| [precompute_fps.py](precompute_fps.py) | 100k → FPS-2048 캐시(`data/fps2048.npz`). 대부분 DL 스크립트의 전제 |
| [train_r0.py](train_r0.py) | R0 치수 회귀 + 열화8×차종3 격자. `data/cd_model.npz` 저장 (fps 캐시 불필요) |
| [train_r2.py](train_r2.py) | R2 메인 학습. `--backbone/--dims/--scale/--npoints/--attn/--seed` |
| [eval_metrics.py](eval_metrics.py) | 3지표 평가 + 차종/쌍둥이 분해 (`--r0`) |
| [summarize_r2.py](summarize_r2.py) · [check_leakage.py](check_leakage.py) | 비교표 / voxel 1-NN 누수 프로브 |
| [scripts/train_100k.py](scripts/train_100k.py) | 100k 원본 점군 학습(§6). `--build-cache`, OOM 자동 기록 |
| [scripts/holdout_eval.py](scripts/holdout_eval.py) · [scripts/delta_cd_eval.py](scripts/delta_cd_eval.py) | 홀드아웃 / ΔCd 정확도 |
| [scripts/automl_parametric.py](scripts/automl_parametric.py) · [scripts/train_regdgcnn.py](scripts/train_regdgcnn.py) | ML 트랙(§5.1.2) / 저자 RegDGCNN 재현 |
| [scripts/train_crossmodal.py](scripts/train_crossmodal.py) · [scripts/crossmodal_phase2.py](scripts/crossmodal_phase2.py) | 크로스모달(future work) |
| [scripts/verify_data.py](scripts/verify_data.py) | Phase 0 데이터 검증 — 새 머신에서 먼저 돌릴 것 |

핵심 재현(R0→R2→평가)에 필요한 건 루트 파일뿐. `scripts/`는 검증·확장 실험용.

---

## 6. A100에서 새로 열리는 것 (이 서버로 옮기는 진짜 이유 — 먼저 돌릴 것)

5080(16GB)의 병목은 순전히 VRAM. A100 40GB(=2.4×)/80GB(=4.8×)에서 열리는 것들. 숫자는 실제 로그에서 검증.

### (1) 100k 원본 점군 학습 — 최우선

5080에서는 100k PointNet이 **bs 8/16/32 전부 OOM**, `bs=4`에서만 통과 → **~102초/epoch**, 현재 best val R² **0.891**(진행 중). 반면 같은 PointNet이 FPS-2048에선 val 0.965/test 0.968 — 즉 100k가 **bs4의 열악한 배치 통계 때문에 오히려 뒤처지는 중**이고, 이게 A100 재실행의 핵심 동기다. DGCNN/RegDGCNN은 100k에서 bs=1도 OOM 예상(스크립트가 자동으로 "불가" JSON 기록).

```bash
python scripts/train_100k.py --build-cache            # 캐시 없을 때만, ~13초
python scripts/train_100k.py --backbone pointnet --bs 32   # A100 40GB: bs16-32 / 80GB: bs32-64
```
→ 큰 배치로 정상 통계 + 훨씬 빠름. "FPS-2048 ≈ 100k" 주장을 제대로 검증/반증. **조기종료까지 ~1시간 이내 추정.**

### (2) 2,048점 공정 벤치마크 — 4개 백본 전부

5080에서는 RegDGCNN이 2048점에서 OOM → 4모델 비교를 **1,024점/bs8**로 낮춰야 했다(덱의 0.962=공유 1024설정 vs 0.968=서비스 2048설정이 갈린 이유). A100이면 **2,048점에서 4개 전부** 동일 조건 재실행 가능.

```bash
for bb in pointnet dgcnn regdgcnn triplane; do
  python train_r2.py --backbone $bb --dims 0 --npoints 2048 --bs 16
done ; python summarize_r2.py
```
5080 실측: PointNet 2048/bs32 = 1.9초/ep, 120ep ~4분; RegDGCNN 1024/bs8 = 143초/ep, ~1.75h. A100 2048점 RegDGCNN은 epoch ~1–2분 추정.

### (3) 아직 안 잰 것 (RESULTS.md §7)

1. **PointNet 열화 성능(R3)**: noise/sparse/one-side/phone — 앱의 실제 조건, 완전 공백. 가장 값진 미측정치. 연산자는 [cd_common.py](cd_common.py)에 이미 있음(신규 R3 스크립트 필요).
2. **RegDGCNN/DGCNN 정식 레시피(SGD lr 0.1, 250ep)** — ⚠ **플래그로 안 됨.** [scripts/train_regdgcnn.py](scripts/train_regdgcnn.py)·[train_r2.py](train_r2.py)가 Adam/AdamW를 하드코딩하고 `--optimizer` 플래그가 없다. 옵티마이저·스케줄러를 **코드에서 교체**해야 진짜 레시피 실행. A100에서만 250ep이 현실적.
3. **시드 분산**: `for s in 0 1 2 3 4; do python train_r2.py --backbone pointnet --npoints 2048 --seed $s; done` — PointNet 0.968 vs DGCNN 0.951 격차가 시드 노이즈보다 큰지 결론.
4. **계열 홀드아웃**: `scripts/holdout_eval.py --test-prefix F_S_WWC_WM` (일부 실측: F/E 계열 0.750/0.799, Estate 차종 0.457).

**우선순위**: (1) 100k PointNet @ 큰 배치 → (2) 2048점 4-백본 매트릭스 → (3) PointNet 열화 R3 → (4) RegDGCNN 정식 SGD(코드 수정) → (5) 시드 스윕.
