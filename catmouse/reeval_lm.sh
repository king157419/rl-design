#!/usr/bin/env bash
cd "$(dirname "$0")"; export KMP_DUPLICATE_LIB_OK=TRUE; PY=D:/conda/envs/graph312/python.exe
for g in indiv team mix near noconceal pool poollat; do
  while [ "$(jobs -rp | wc -l)" -ge 3 ]; do wait -n; done
  $PY -X utf8 reeval_lm.py ${g}_s0 ${g}_s1 ${g}_s2 > results/eval/reeval_lm_$g.log 2>&1 &
done
wait; echo "reeval done"; grep -h "learned_cat_vs_mixed" results/eval/reeval_lm_*.log
