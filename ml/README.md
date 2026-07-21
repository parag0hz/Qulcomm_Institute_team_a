# ml — 자동차 항력계수(Cd) 예측

DrivAerNet++ 데이터로 **설계 파라미터(ML)** 와 **3D 포인트클라우드(DL)** 두 경로에서 Cd를 예측한다.

## 빠른 시작
```bash
conda create -y -n qi python=3.12 && conda activate qi
pip install torch numpy scipy scikit-learn xgboost pandas matplotlib open3d wandb tqdm optuna
export QI_DATA=$PWD/data          # 데이터 위치 (데이터는 저장소에 없음, 별도 전송)
python scripts/protocol.py        # 분할·지표 프로토콜 검증
```
> **데이터는 저장소에 포함돼 있지 않다** (수십 GB). 준비 방법은 [A100_BOOTSTRAP.md](A100_BOOTSTRAP.md) 참조.

## 문서
| 파일 | 내용 |
|---|---|
| [CLAUDE.md](CLAUDE.md) | 프로젝트 전체 가이드·주의사항 (먼저 읽을 것) |
| [PROTOCOL_COMPARISON.md](PROTOCOL_COMPARISON.md) | **ML vs DL 정량 비교** (동일 데이터·K=5 fold) |
| [METRICS.md](METRICS.md) | 평가 지표 정의 |
| [RESULTS.md](RESULTS.md) | 실험 결과 전체 |
| [A100_BOOTSTRAP.md](A100_BOOTSTRAP.md) | 새 서버 환경·데이터 구축 |
| [A100_DL_HANDOFF.md](A100_DL_HANDOFF.md) | A100에서 돌릴 딥러닝 작업 지시서 |

## 핵심 결과 (교집합 3,704대 · K=5 rotating fold)
| 모델 | 입력 | R² | MAE |
|---|---|---:|---:|
| **PointNet** | 포인트클라우드 | **0.853** | 0.0069 |
| AutoGluon | 설계 파라미터 23개 | 0.573 | 0.0117 |

동일 조건에서도 형상 기반 딥러닝이 파라미터 기반 ML을 크게 앞선다.
