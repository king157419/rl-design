#!/usr/bin/env bash
# 全部实验的队列，每一步都可续跑：中途暂停或中断后重新运行本脚本即可，已完成的部分会跳过。
cd "$(dirname "$0")"
export KMP_DUPLICATE_LIB_OK=TRUE
PY=D:/conda/envs/graph312/python.exe
log() { echo "[$(date +%m-%d_%H:%M)] $*" >> results/run_all.log; }
alive() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | ?{\$_.CommandLine -match '$1'}).Count" | tr -d '
'; }
# 1) 两轮转化实验（可能已有一个被暂停的进程在跑，先等它）
until grep -q "saved" results/campus_imitate.log 2>/dev/null; do
  if [ "$(alive campus_imitate)" = 0 ] && ! grep -q "saved" results/campus_imitate.log; then
    $PY -X utf8 campus_imitate.py >> results/campus_imitate.log 2>&1
  fi
  sleep 30
done; log imitate done
# 2) 脚本探针
grep -q "^saved" results/campus_probe.log 2>/dev/null || $PY -X utf8 campus_probe.py >> results/campus_probe.log 2>&1
grep -q "^saved" results/campus_probe.log && log probe done || { log probe FAILED; exit 1; }
# 3) 训练（3 个一批）
# 显卡驱动偶尔超时重置（Windows TDR）会让训练报 CUDA 错误退出；检查点每 10 轮一存，失败就自动续跑，最多 5 次
run() { for k in 1 2 3 4 5; do $PY -X utf8 campus_ppo.py --iters 300 "$@" >> "results/campus_ppo_${2}.log" 2>&1 && break; log "$2 crashed, retry $k"; sleep 20; done; }
run --tag indiv_s0 --reward indiv --seed 0 &
run --tag indiv_s1 --reward indiv --seed 1 &
run --tag team_s0 --reward team --seed 10 &
wait
run --tag team_s1 --reward team --seed 11 &
run --tag noconceal_s0 --reward indiv --seed 20 --no_conceal &
run --tag noconceal_s1 --reward indiv --seed 21 --no_conceal &
wait; log training done
# 4) 交叉评估（已有结果文件的跳过）
mkdir -p results/eval
ev() { [ -f "results/eval/$1.pt" ] || $PY -X utf8 campus_eval.py "results/eval/$1.pt" --no-baseline $1 > "results/eval/$1.log" 2>&1; }
[ -f results/eval/baseline.pt ] || $PY -X utf8 -c "import campus_eval as E, torch; r=E.match('script','mixed'); E.summarize('脚本猫 vs 脚本老鼠', r); torch.save({'script_vs_script': r}, 'results/eval/baseline.pt')" > results/eval/baseline.log 2>&1 &
ev indiv_s0 & ev indiv_s1 & wait
ev team_s0 & ev team_s1 & ev noconceal_s0 & wait
ev noconceal_s1
log all done
