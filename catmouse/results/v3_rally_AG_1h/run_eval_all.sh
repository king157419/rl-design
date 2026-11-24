#!/usr/bin/env bash
# 训练全部结束后做交叉评估：每个训练种子 64 局；脚本对脚本的基线单独一份
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
PY=D:/conda/envs/graph312/python.exe
until grep -q "all training done" results/train_all.log 2>/dev/null; do sleep 60; done
ev() { $PY -X utf8 campus_eval.py "results/eval/$1.pt" ${2:-} $1 > "results/eval/$1.log" 2>&1; }
mkdir -p results/eval
$PY -X utf8 -c "import campus_eval as E, torch; r=E.match('script','mixed'); E.summarize('脚本猫 vs 脚本老鼠', r); torch.save({'script_vs_script': r}, 'results/eval/baseline.pt')" > results/eval/baseline.log 2>&1 &
ev indiv_s0 --no-baseline &
ev indiv_s1 --no-baseline &
wait
ev team_s0 --no-baseline &
ev team_s1 --no-baseline &
ev noconceal_s0 --no-baseline &
wait
ev noconceal_s1 --no-baseline
echo "all eval done"
