# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Predict a car's **drag coefficient (Cd)** from a **3D point cloud**, specifically under the degradation of an iPhone LiDAR scan (partial view, noise, sparsity). Hackathon project: the phone app is a demo; the real work is the model.

Dataset is **DrivAerNet++** (7,713 sedan-family cars, 100k points each, Cd labels).

[PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) is a self-contained handoff from a prior session on a different machine. It is the source of truth for project intent and roadmap — but several of its *numbers* have since been contradicted by a full-dataset scan. Where this file disagrees with it, this file wins.

What exists here now:

- [cd_common.py](cd_common.py) — safe pickle loader, ID normalization, filesystem index, `check_integrity()`, the 6 robust features (`robust_feats`), and every degradation operator (`degrade_noise/sparse/glass/oneside/phone`, `mirror_complete`, `estimate_symmetry_plane`). **Every data access should go through this.**
- [DATA_SUMMARY.md](DATA_SUMMARY.md) — full-dataset statistics (all 7,713 clouds, 771 M points), and the three places the handoff turned out to be wrong.
- [RESULTS.md](RESULTS.md) — **the canonical results document** (Korean): the full clean baseline ladder, all-metric tables, R0 degradation grid, twin-leakage analysis, and §7's list of what has *not* been measured. Update it when new numbers land; the tables in this file are excerpts.
- [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) — consolidated experiment report (Korean, 2026-07-16): both tracks (tabular AutoML + point-cloud DL) with inputs/outputs, configs, fair-comparison matrix, attention ablation, holdouts, ΔCd resolution, and the paper-verification findings. RESULTS.md remains the canonical running document; this is the presentation-ready snapshot.
- [PRD.md](PRD.md) — product requirements (Korean, 2026-07-14): user-facing scope (STL upload → Cd web tool), staged targets with achievement status, out-of-scope list, risks.
- [CFA_PROJECT.md](CFA_PROJECT.md) — an MVP build plan authored outside this repo (FFD counterfactual suggestions, LLM report, FastAPI serving). Useful as a product roadmap, but **several of its technical instructions contradict measured findings here** — see the warning below.
- Presentation materials: `CFA_HackathonFinal_v9.pdf` (final 22-page deck) and `해커톤 ppt (1,2합친버전).pptx` (draft).
- Training and eval scripts — see the workflow below. `outputs/` holds per-run metrics (`<tag>.json`) and test predictions (`*_pred.npz`).

R0 and R2 have been reproduced/built on this machine. Only handoff §8's `viz_*.py` / `train_model.py` were never ported.

### ⚠ Where CFA_PROJECT.md conflicts with this repo, this file wins

- Its §4 mandates mean-centering + **unit-sphere normalization** as the shared train/inference convention. That erases the meter scale invariant 1 preserves. Its own expectation (§5: "literature PointNet R² ≈ 0.64") is roughly what that costs; this repo's meter-scale PointNet measures **0.968** on the official test split.
- Its §1-5 suggests `paddle.load()` / pickle to open `.paddle_tensor` — unpickling executes arbitrary code. Use `cd_common.safe_load_ndarray`.
- Its §1-4 prescribes a naive CSV↔file inner join — that silently drops 609 designs (ID zero-padding mismatch). Use `cd_common.norm_id` / `file_index()`.
- Its §1 download and §2 venv/torch setup are already done here (`data/` verified, conda env `qi`).
- One correction to its §3: with meter scale kept, uniform-scale deformations are *not* cancelled by normalization — that caveat only applies to its unit-sphere pipeline.

Still sound and worth keeping: clamping counterfactual deformations to dataset ranges (exact ranges in [DATA_SUMMARY.md](DATA_SUMMARY.md) §4), the LLM-as-translator rules (no invented numbers or findings), and the PDF disclaimer line.

## Workflow

```bash
python precompute_fps.py    # one-time: 100k clouds → data/fps2048.npz (FPS-2048 + 6 dims, GPU); required before train_r2
python train_r0.py          # R0: fit dims model → data/cd_model.npz, print full degradation × body-type grid
python train_r2.py --backbone pointnet --dims 0 --npoints 2048
                            # R2; other options: --backbone dgcnn|mlp, --dims 1, --scale unit
python eval_metrics.py outputs/pointnet_n2048_dims0_meter_pred.npz
                            # 3-metric eval + twin/lone split; `--r0` evaluates the dims model instead
python summarize_r2.py      # outputs/*.json → comparison table vs R0
python check_leakage.py     # zero-learning voxel 1-NN retrieval baseline (leakage probe)
python scripts/holdout_eval.py --test-body Estate      # generalization holdout (or --test-prefix <family>)
./run_rerun.sh              # train_r2 + both evals, logged to outputs/rerun.log — run inside tmux
```

Experiment tracking: wandb project **`cfa`** — `train_r2.py` and `scripts/holdout_eval.py` log automatically (`--wandb 0` to disable; without login they warn and continue). `scripts/wandb_backfill.py` backfills the pre-wandb history (11 runs) — run **once**, after `wandb login`; re-running duplicates runs.

There is no test suite; verification is `check_integrity()` passing and the metric tables matching [RESULTS.md](RESULTS.md).

## Git

`qi/` is its own repository (branch `main`, initialized 2026-07-10, no remote yet). It sits inside a larger repo rooted at `$HOME` — `DiffKD_ECG`, an unrelated ECG paper project — but is decoupled from it two ways: the nested `.git` here, and a `/qi/` line in `$HOME/.git/info/exclude`. Commands run from this directory hit *this* repo, not the paper repo.

One residual caution: if you ever operate on the `$HOME` repo itself, `$HOME` still has ~150 untracked paths and the exclude entry is local-only (not pushed) — don't stage broadly there.

The [.gitignore](.gitignore) keeps `data/`, checkpoints, and `*.tar` out of this repo.

## Environment

```bash
conda activate qi     # 이 프로젝트 전용 env
```

Linux, **NVIDIA RTX 5080 (16.6 GB)**, `/home/kwy00/anaconda3/envs/qi`.

| | Version | Note |
|---|---|---|
| Python | 3.12.13 | **not 3.13** — open3d ships no 3.13 wheel |
| torch | 2.13.0+cu130 | `sm_120` present in `get_arch_list()`; GPU matmul verified |
| open3d | 0.19.0 | `hidden_point_removal` smoke-tested (needed for R3) |
| xgboost / sklearn / scipy / numpy | 3.3.0 / 1.9.0 / 1.18.0 / 2.5.1 | |
| torch_geometric | not installed | only needed if R4 goes to PointNet++/DGCNN |

Use `device="cuda"`. The RTX 5080 is Blackwell (**sm_120**); a torch built for CUDA < 12.8 will fail at kernel launch, so install torch from the `cu130` index, not default PyPI.

Korean matplotlib labels: `Noto Sans CJK KR` is installed (no NanumGothic). Set `rcParams["font.family"]` and `axes.unicode_minus=False`.

The handoff's §2 "환경 & 함정" describes the *old* macOS box. Stale on four counts: MPS→CUDA, `wget` missing→present, sklearn missing→present, torch 2.10→2.13. Its numpy-2 and pickle traps still apply.

## Data

Downloaded and verified in `data/` (2026-07-10). Handoff §5's EDA reproduces exactly: 7,713 clouds, Cd 0.201–0.383 (0.284 ± 0.037), class counts 1124/1366/5223, height↔Cd r = +0.818.

```
data/
├── DrivAerNetPlusPlus_Processed_Point_Clouds_100k_paddle/   # 7,713 × .paddle_tensor
├── DrivAerNetPlusPlus_Drag_8k.csv                           # Design, Average Cd  (7,713 rows)
├── subset_dir/{train,val,test}_design_ids.txt               # 5398 / 1157 / 1158
└── DrivAer++_Points.tar                                     # 8.6 GB source, kept (re-download ≈ 9 min)
```

Three traps, all still live:

- **Loading the point clouds.** Each `.paddle_tensor` is a raw Python pickle from a third-party mirror. Do not `pickle.load` it — unpickling executes arbitrary opcodes. Use the `safe_load_ndarray` reader in handoff §4: it walks `pickletools.genops` and reconstructs the buffer with `np.frombuffer`, executing nothing. Verified to yield exactly `(100000, 3)` float32 / 1,200,000 bytes.

- **ID matching — silently drops 609 of 7,713 designs. Fixed; use `cd_common`.** The CSV leaves `E/F/N_S_*` indices unpadded (`E_S_WWC_WM_1`) while both the split files and the point-cloud filenames zero-pad to three digits (`E_S_WWC_WM_001`). Joining on the raw string yields 7,104 rows and raises nothing. The 609 lost designs are exactly indices 1–99 of each of the seven `E/F/N_S_*` variants — a systematic slice, not a random one.

  Root cause: DrivAerNet++ merges the original DrivAerNet fastback set (`DrivAer_F_D_WM_WW_XXXX`, 4-digit, 3,966 designs) with newly generated parametric families (`{E,F,N}_S_*`, 3-digit). The two came from different ID pipelines, and *neither* the official nor the PaddleScience dataloader normalizes — both assume exact string equality ([issue #21](https://github.com/Mohamedelrefaie/DrivAerNet/issues/21) acknowledges the sibling `DrivAer_` prefix problem). Upstream has since stripped that prefix; our mirror is a pre-fix snapshot, so `cd_common.norm_id` strips it too and stays compatible with both.

  Never reconstruct a filename from an ID — look it up in `cd_common.file_index()`, which is built by scanning the directory. Call `check_integrity()` at the top of any pipeline.

- **numpy 2.x.** `ndarray.ptp()` was removed; call `np.ptp(arr, axis=...)`.

Coordinate frame is already canonical for every car: ground at `z=0` (verified 0.0000 ± 0.0000), lateral symmetry axis at `y=0` (0.0001 ± 0.0008), `x` front→back, meters.

Note when comparing geometry against handoff §5: those ranges (length 4.32–5.10, width 1.67–2.15 m) come from **percentile** extents. Raw `min`/`max` runs wider — raw width reaches 2.39 m, robust width 2.15 m.

Handoff §4 says points with `|y| > 1.0` are side mirrors. **True for a typical car, false for 929 of them (12%)**, whose robust half-width alone exceeds 1.0 m (max 1.077 m). A fixed `|y| > 1.0` threshold will amputate body panels on wide cars — the affected point fraction reaches 27% on the worst one. Segment mirrors per-car (e.g. against `p99.5(|y|)`), not with a constant.

## Modeling invariants

Two principles drive every design decision here; violating either silently destroys the result.

**1. Do not normalize away scale.** Cd is dimensionless, but in this dataset absolute vehicle height is the single strongest predictor (r = +0.82). The standard PointNet habit of rescaling each cloud to a unit sphere throws that signal away. Keep meter scale, and feed global dimensions as an auxiliary late-fusion input. LiDAR supplies true metric scale, so this is legitimate at inference.

**2. Inject the degradation you will face at test time.** Train against synthesized phone-scan corruption (one-sided occlusion, noise, sparsity, glass dropout), not clean clouds.

**3. R0 is a Fastback-only model. This is the central problem, and it is measured, not conjectured.** Reproduced R0 on the test split (`python train_r0.py`), clean, v2 robust features:

| | All | Fastback | Estate | Notchback |
|---|---:|---:|---:|---:|
| R² | +0.814 | +0.881 | **−0.519** | **−0.151** |
| MAE | 4.6% | 3.6% | 6.6% | 6.7% |

Negative R² means the model is **worse than predicting that class's own mean Cd**. On 2,490 of 7,713 cars, R0 has negative skill. It only looks good globally because Fastback is 68% of the data and carries most of the Cd variance (σ = 0.0375 vs 0.019).

For scale: predicting nothing but the body-type mean already gives global R² = 0.226. Height's global r = +0.827 collapses to +0.004 within Estate and −0.040 within Notchback ([DATA_SUMMARY.md](DATA_SUMMARY.md) §7).

Class weighting and oversampling do **not** fix this — the features carry no within-class signal to reweight. What separates Estate and Notchback designs is rear-end shape and curvature detail, which is exactly what the point-cloud models (R2/R3) exist to capture. **This is the target R2 must beat**, and the only honest way to see it is per-class R².

Two further findings from the Tier-1 (R0) baseline:

- **Percentile features beat max/hull features** under noise (R² 0.77 vs 0.40 at 1.5 cm noise) — robust statistics, not extrema.
- **Mirror completion rescues one-sided scans — but only with an oracle symmetry plane.** Mirroring the phone-degraded cloud across the dataset's ground-truth `y=0` restores R² 0.478 → 0.783. A real scan has no such axis, and the plane must be estimated. Measured sensitivity of `phone+mirror` (v2, test):

  | plane error δ | 0 | 5 cm | 10 cm | 15 cm | 20 cm | 30 cm | 50 cm |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | global R² | +0.783 | +0.746 | +0.675 | +0.594 | +0.527 | −0.081 | −1.280 |

  Budget: **δ < 5 cm** to keep the headline result; beyond **δ ≈ 25 cm** mirroring is worse than not mirroring at all (`phone` alone = +0.478). Three cheap estimators all miss the budget — centroid/PCA origin 56 cm (R² −0.655), roof-ridge 18.8 cm (R² +0.332, i.e. *worse than no mirroring*), a chamfer self-symmetry fit 8.3 cm but degenerate (it drifts to the search-grid boundary).

  Root reason: a strictly one-sided scan never observes the far outer surface, so the plane is only weakly identifiable. Do not treat the 0.783 number as deployable. Either estimate the plane properly, or get real coverage — the multi-view HPR trajectory planned for R3 is the principled fix. Either way, **plane error must be an axis of the evaluation grid**, not an assumption.

Do not trust handoff §6's `drop` (rear-end drop) correlation of +0.58. Swept across 96 plausible definitions, r ranges +0.159 to +0.799, and every definition reaching 0.55 relies on raw `min`/`max` extrema and a 5% rear window — precisely the choices that collapse under noise. The robust definition §6 itself publishes yields **+0.191**. Treat `drop` as near-useless and use the other five features.

When computing frontal area via `scipy.spatial.ConvexHull` on the 2D `(y, z)` projection, the area is `.volume`, not `.area` (which is the perimeter in 2D).

## R2 results (clean test, 1158 designs)

PointNet on FPS-2048 raw points, no T-Net, no augmentation, 0.81 M params. `python train_r2.py --backbone pointnet --dims 0`

| | R² | MAE (drag counts) | pairwise rank acc |
|---|---:|---:|---:|
| R0 dims, All | +0.814 | 12.5 | 85.6% |
| **PointNet, All** | **+0.968** | **5.1** | **94.6%** |
| R0 dims, Estate | −0.519 | 18.3 | **51.9%** (coin flip) |
| **PointNet, Estate** | **+0.839** | **5.5** | **87.3%** |
| R0 dims, Notchback | −0.151 | 16.0 | 54.1% |
| **PointNet, Notchback** | **+0.802** | **6.6** | **86.2%** |

Raw points close the per-class gap that no dimension model could: physical error is now near-uniform across body types (4.7 / 5.5 / 6.6 counts) where R0 was 10.2 / 18.3 / 16.0.

Backbone findings, all at matched budget (AdamW, lr 1e-3, 120 ep):

- **PointNet beats DGCNN** (+0.968 vs +0.951), with half the parameters and 8× faster epochs. Caveat: DGCNN was given PointNet's recipe, not its canonical SGD/lr=0.1/250-epoch one. Local neighbourhood aggregation did not help — Cd here appears to be driven by global silhouette, not fine curvature.
- **Late-fusing the 6 global dims adds nothing** (+0.969 vs +0.967). The point cloud already contains them. May still matter under degradation, where percentile dims are robust and raw points are not.
- **1024 points ≈ 2048 points** (+0.964 vs +0.967). Cheap enough for the phone.
- **Drop the T-Net.** The data is already canonically aligned, and a learned 3×3 transform can undo metric scale.
- **Unit-sphere ablation ties meter scale on clean data** (`--scale unit`: +0.969 vs +0.968) — a full clean cloud evidently carries the height signal even after rescaling. This does *not* retire invariant 1: its real motivation is degraded/partial clouds, where that implicit signal is corrupted, and that case is unmeasured. Meter scale stays the default.

Two things the headline number does *not* mean:

- **A zero-learning voxel 1-NN retrieval baseline already scores R² 0.865** (5-NN: 0.903), because the dataset is a dense parametric sweep. On Fastback it reaches 0.924 — most of PointNet's 0.974 there is retrievable. But on Estate retrieval collapses to **−0.632** while PointNet gets **+0.839**, so the within-class gain is genuinely learned. Verified further by splitting test into designs that do (516) and do not (642) have a same-index sibling in train: stratified by body type, the twin advantage is ≤0.8 counts and ≤3 pp of rank accuracy.
- **All 8 morph families appear in both train and test.** "Lone" is not out-of-distribution. A family-holdout split is the honest generalization test and has not been run.

## Evaluation discipline

Never report clean accuracy alone. The presentation metric is a grid:

`[clean / noise / sparse / HPR-occluded / phone / phone+mirror] × [Fastback / Estate / Notchback]`, reporting MAE and R².

Report three metrics, not one ([eval_metrics.py](eval_metrics.py)) — they answer different questions:

- **R²** (subgroup's own mean as baseline) — does the model beat predicting that class's mean? Harsh on Estate/Notchback because σ = 0.018 vs Fastback's 0.038, so a 1.8× worse absolute error reads as a sign flip. Right question, misleading magnitude.
- **MAE in drag counts** (1 count = 0.001 Cd) — the domain-native unit, comparable across classes.
- **Pairwise ranking accuracy** — given two designs, is the lower-drag one identified? This is what the product actually claims to do (handoff §1 positions it as a relative design-feedback tool), and it is the honest headline. 50% is a coin flip.

Per-class breakdown is mandatory because the dataset is badly imbalanced — Fastback is 68% of designs (5,223) and also has the widest Cd spread (0.296 ± 0.038), so a global average hides failures on Notchback (0.247 ± 0.019) and Estate. Use sample weighting or oversampling in training.

Official splits (`subset_dir/{train,val,test}_design_ids.txt`, 5398/1157/1158) have near-identical Cd distributions, so held-out numbers are trustworthy. Use them; don't re-split.

The known scope limit: training data is DrivAer sedan-family variants only, so real SUVs and trucks are out of distribution. Position outputs as a relative design-feedback score with an OOD confidence flag rather than an absolute Cd measurement.

## Roadmap

The ladder (handoff §7), with a working model kept at every rung:

| | Model | Status |
|---|---|---|
| R0 | Robust dimensions → linear/quadratic regression | **Reproduced** ([train_r0.py](train_r0.py)): clean test R² 0.814, MAE 4.6% (handoff: 0.81 / 4.4%). Model at `data/cd_model.npz`. Eval grid = 8 degradations × 3 body types. |
| R1 | Dimensions → MLP | **Done** (`train_r2.py --backbone mlp`): global R² +0.861 but Estate −0.109 / Notchback +0.034 — more capacity on the same 6 dims does not fix the per-class failure |
| R2 | PointNet on raw points, no augmentation | **Done** ([train_r2.py](train_r2.py), [models_pc.py](models_pc.py)). Clean test R² 0.968, 5.1 drag counts. Beats DGCNN. See below. |
| R3 | PointNet + HPR domain randomization | **next entry point** — R2 is clean-only; nothing is known about degraded performance |
| R4 | PointNet++ / DGCNN / PointNeXt | if time permits |
| R5 | Uncertainty, multi-task, ensemble, TTA | differentiator |

The evaluation grid, R0, R1, and R2 are done. The single most valuable missing number is **PointNet under degradation** (noise / sparse / one-side / phone) — the app's actual operating condition, and R0's grid shows degradation is where models fall apart. R3 exists to close that gap; see [RESULTS.md](RESULTS.md) §7 for the full list of open measurements (family-holdout split, seed variance, canonical DGCNN recipe).

R2 input pipeline: canonicalize → FPS to 2048 points (cached in `data/fps2048.npz`) → **keep meter scale** → late-fuse global dimensions → standardize the Cd target → Huber loss → early stopping on val.

The highest-value experiment for the presentation is closing the sim-to-real gap: no real scan has a ground-truth Cd, so physically scanning 5–10 cars with manufacturer-published Cd values gives a real-world error number. Use it as a validation set only, never for training.
