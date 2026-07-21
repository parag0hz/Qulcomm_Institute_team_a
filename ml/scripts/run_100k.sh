#!/bin/bash
# 100k 전체 점군 실험: 캐시 -> 4개 백본 순차 (OOM은 각자 JSON에 불가 기록) -> 평가
PY=/home/kwy00/anaconda3/envs/qi/bin/python
cd /home/kwy00/qi
LOG=outputs/n100k.log
{
echo "=== $(date) 100k 실험 시작 ==="
if [ ! -f data/pc100k_meta.json ]; then
  echo "--- 캐시 생성 ---"
  $PY scripts/train_100k.py --build-cache || { echo "CACHE_FAILED"; exit 1; }
fi
for bb in pointnet dgcnn regdgcnn triplane; do
  echo "--- $bb ---"
  $PY scripts/train_100k.py --backbone $bb || echo "RUN_FAILED: $bb"
done
echo "--- eval_metrics (성공 런만) ---"
for f in outputs/*_n100000_dims0_meter_pred.npz; do
  [ -f "$f" ] && $PY eval_metrics.py "$f"
done
echo "=== $(date) 완료 ==="
} >> $LOG 2>&1
touch outputs/n100k.DONE
