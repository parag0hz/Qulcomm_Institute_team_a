#!/usr/bin/env bash
# PointNet 재실행 (위생 수정 반영 + 예측값 저장) 후 세 지표 평가.
# tmux 안에서 돌아가므로 SSH가 끊겨도 살아남는다.
set -uo pipefail
PY=/home/kwy00/anaconda3/envs/qi/bin/python
cd /home/kwy00/qi
LOG=outputs/rerun.log
: > "$LOG"

echo "### [1/3] PointNet 재실행 (좌표 중심을 train에서만 계산하도록 수정한 뒤)" | tee -a "$LOG"
$PY train_r2.py --backbone pointnet --dims 0 --npoints 2048 2>&1 | tee -a "$LOG"

echo -e "\n### [2/3] R0(치수 모델)을 세 지표로" | tee -a "$LOG"
$PY eval_metrics.py --r0 2>&1 | tee -a "$LOG"

echo -e "\n### [3/3] PointNet을 세 지표로 + 쌍둥이/고립 분리" | tee -a "$LOG"
$PY eval_metrics.py outputs/pointnet_n2048_dims0_meter_pred.npz 2>&1 | tee -a "$LOG"

echo -e "\nRERUN DONE" | tee -a "$LOG"
