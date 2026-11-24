#!/usr/bin/env bash
# 共同训练：个人奖励、团队奖励各 2 个种子，“隐蔽无优势”反事实 2 个种子；每批 3 个并行，每个 300 轮。
# 先等脚本实验（探针 + 两轮转化）跑完，避免抢显卡。
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
PY=D:/conda/envs/graph312/python.exe
until grep -q "saved" results/campus_imitate.log 2>/dev/null; do sleep 30; done
run() { $PY -X utf8 campus_ppo.py --iters 300 "$@" > "results/campus_ppo_${2}.log" 2>&1; }
run --tag indiv_s0 --reward indiv --seed 0 &
run --tag indiv_s1 --reward indiv --seed 1 &
run --tag team_s0 --reward team --seed 10 &
wait
run --tag team_s1 --reward team --seed 11 &
run --tag noconceal_s0 --reward indiv --seed 20 --no_conceal &
run --tag noconceal_s1 --reward indiv --seed 21 --no_conceal &
wait
echo "all training done"
