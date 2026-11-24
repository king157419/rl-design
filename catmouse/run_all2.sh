#!/usr/bin/env bash
# 第二次审查后的完整重跑（可续跑：中断后重新运行本脚本，已完成的部分自动跳过）
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
PY=D:/conda/envs/graph312/python.exe
log() { echo "[$(date +%m-%d_%H:%M)] $*" >> results/run_all2.log; }
# 1) 脚本实验
grep -q "^saved" results/campus_imitate.log 2>/dev/null || $PY -X utf8 campus_imitate.py >> results/campus_imitate.log 2>&1
grep -q "^saved" results/campus_imitate.log && log imitate done || { log imitate FAILED; exit 1; }
grep -q "^saved" results/campus_probe.log 2>/dev/null || $PY -X utf8 campus_probe.py >> results/campus_probe.log 2>&1
grep -q "^saved" results/campus_probe.log && log probe done || { log probe FAILED; exit 1; }
# 2) 训练：最多 3 个并行；显卡驱动重置导致的崩溃自动续跑，最多 5 次
run() { local tag=$1; shift; for k in 1 2 3 4 5; do $PY -X utf8 campus_ppo.py --tag $tag "$@" >> "results/campus_ppo_${tag}.log" 2>&1 && return 0; log "$tag crashed, retry $k"; sleep 20; done; log "$tag FAILED after 5 retries"; return 1; }
JOBS=(
 "pool_s0 --reward indiv --seed 50 --iters 600 --pool"
 "pool_s1 --reward indiv --seed 51 --iters 600 --pool"
 "pool_s2 --reward indiv --seed 52 --iters 600 --pool"
 "poollat_s0 --reward indiv --seed 60 --iters 600 --pool --pool_max 1"
 "poollat_s1 --reward indiv --seed 61 --iters 600 --pool --pool_max 1"
 "poollat_s2 --reward indiv --seed 62 --iters 600 --pool --pool_max 1"
 "indiv_s0 --reward indiv --seed 0 --iters 300"
 "indiv_s1 --reward indiv --seed 1 --iters 300"
 "indiv_s2 --reward indiv --seed 2 --iters 300"
 "team_s0 --reward team --seed 10 --iters 300"
 "team_s1 --reward team --seed 11 --iters 300"
 "team_s2 --reward team --seed 12 --iters 300"
 "near_s0 --reward near --seed 30 --iters 300"
 "near_s1 --reward near --seed 31 --iters 300"
 "near_s2 --reward near --seed 32 --iters 300"
 "mix_s0 --reward mix --seed 40 --iters 300"
 "mix_s1 --reward mix --seed 41 --iters 300"
 "mix_s2 --reward mix --seed 42 --iters 300"
 "noconceal_s0 --reward indiv --seed 20 --iters 300 --no_conceal"
 "noconceal_s1 --reward indiv --seed 21 --iters 300 --no_conceal"
 "noconceal_s2 --reward indiv --seed 22 --iters 300 --no_conceal"
)
for j in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  run $j &
done
wait; log training done
# 3) 交叉评估（已有结果的跳过），3 个并行
mkdir -p results/eval
[ -f results/eval/baseline.pt ] || $PY -X utf8 -c "import campus_eval as E, torch; r=E.match('script','mixed'); E.summarize('脚本猫 vs 脚本老鼠', r); torch.save({'script_vs_script': r}, 'results/eval/baseline.pt')" > results/eval/baseline.log 2>&1
for j in "${JOBS[@]}"; do
  t=${j%% *}
  [ -f "results/eval/$t.pt" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  $PY -X utf8 campus_eval.py "results/eval/$t.pt" --no-baseline $t > "results/eval/$t.log" 2>&1 &
done
wait; log all done
