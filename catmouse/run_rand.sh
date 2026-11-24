#!/usr/bin/env bash
# 随机分配功劳对照：3 个种子 × 300 轮，训练后交叉评估（可续跑）
cd "$(dirname "$0")"; export KMP_DUPLICATE_LIB_OK=TRUE; PY=D:/conda/envs/graph312/python.exe
log() { echo "[$(date +%m-%d_%H:%M)] $*" >> results/run_rand.log; }
run() { local tag=$1; shift; for k in 1 2 3 4 5; do $PY -X utf8 campus_ppo.py --tag $tag "$@" >> "results/campus_ppo_${tag}.log" 2>&1 && return 0; log "$tag crashed, retry $k"; sleep 20; done; log "$tag FAILED"; return 1; }
for s in 0 1 2; do run rand_s$s --reward rand --seed $((70+s)) --iters 300 & done
wait; log training done
for s in 0 1 2; do
  [ -f results/eval/rand_s$s.pt ] || $PY -X utf8 campus_eval.py results/eval/rand_s$s.pt --no-baseline rand_s$s > results/eval/rand_s$s.log 2>&1 &
done
wait; log all done; cat results/eval/rand_s*.log
