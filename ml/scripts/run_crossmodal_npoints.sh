#!/bin/bash
# 크로스모달 점 개수 ablation: 1024 / 2048 / 4096 (FPS-4096 캐시에서 nested)
PY=/home/kwy00/anaconda3/envs/qi/bin/python
cd /home/kwy00/qi
CACHE=data/fps4096.npz
LOG=outputs/crossmodal_npoints.log
: > $LOG
{
echo "=== $(date) 크로스모달 점개수 실험 시작 ==="
if [ ! -f $CACHE ]; then
  echo "--- FPS-4096 캐시 생성 ---"
  $PY scripts/precompute_fps_k.py --k 4096 --out $CACHE || { echo "CACHE_FAILED"; exit 1; }
fi
for n in 1024 2048 4096; do
  echo ""
  echo "############## npoints=$n ##############"
  $PY scripts/train_crossmodal.py --npoints $n --cache $CACHE 2>&1 | grep -vE "^wandb:|UserWarning|warnings.warn" || echo "RUN_FAILED: $n"
done
echo ""
echo "=== $(date) 완료 ==="
} >> $LOG 2>&1
touch outputs/crossmodal_npoints.DONE
