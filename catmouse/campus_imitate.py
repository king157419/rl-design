"""上下两轮之间会不会“改策略”：演化博弈实验。

第一步（模拟）：一轮 30 分钟，20 只老鼠里躲的比例 x 取 0、0.25、0.5、0.75、1，其余跑；每个 x 跑 256 局，
记下每只老鼠的策略和存活时间。
第二步（分析，见 stats_imitate.py）：
  - 躲和跑的收益（平均存活）随 x 怎么变：如果躲在任何 x 下都更好，模仿会把所有人推向躲；
    如果躲多了以后跑反而更好，就会出现反方向的转化和一个混合均衡。
  - 个体层面的模仿：第一轮结束后，每只老鼠随机看一个同伴，按费米规则
    p = 1 / (1 + exp(-(对方存活 - 自己存活) / K)) 改成对方的策略，统计“跑→躲”和“躲→跑”各有多少。
"""
import time

import torch

from campus import CCfg, Campus
from campus_scripted import HIDE, RUN, play

B = 256
OUT = "results/campus_imitate.pt"
import os
rows = torch.load(OUT, weights_only=False) if os.path.exists(OUT) else []  # 续跑
t0 = time.time()
for i, x in enumerate((0.0, 0.25, 0.5, 0.75, 1.0)):
    if any(r["x"] == x for r in rows):
        continue
    n_hide = round(20 * x)
    s = torch.full((B, 30), RUN, dtype=torch.long, device="cuda")
    # 每局随机决定哪些老鼠躲，避免编号和出生位置相关
    perm = torch.argsort(torch.rand(B, 20, device="cuda"), dim=1) + 10
    s.scatter_(1, perm[:, :n_hide], HIDE)
    env = Campus(CCfg(), B=B, seed=100 + i)
    st = play(env, s)
    rows.append({"x": x, "strat": s.cpu(), "surv": st.cpu()})
    torch.save(rows, OUT)
    m = st[:, 10:] / 60
    ss = s[:, 10:]
    msg = " | ".join(f"{n} {m[ss == k].mean():.1f} min" for k, n in ((RUN, "跑"), (HIDE, "躲")) if (ss == k).any())
    print(f"[{time.time() - t0:5.0f}s] 躲的比例 {x:.2f}: {msg}", flush=True)
print("saved", OUT)
