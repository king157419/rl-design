"""探针实验 1：脚本策略"躲 vs 跑"的因子扫描。

A. 混合种群（同一局里 10 只躲、10 只跑），全因子：
   掩体数 {0,4,8,14,20} × 体力 {无限, 20 s} × 静止显眼度 s {1, 0.5} × 隐藏身份 {否, 是}
B. 同质种群对照（20 只全躲 或 全跑），体力 20 s、隐藏身份=否：检验"躲的人是不是靠跑的人当诱饵"。
每个条件 2 张随机障碍布局 × 512 局 = 1024 局。
"""
import itertools
import json
import time

import torch

from env import Cfg, CatMouse
from scripted import HIDE, RUN, play

B, LAYOUTS = 512, (0, 1)
OUT = "results/probe1.pt"
rows = []


def run(cfg_kw, strat_fn, tag):
    sts = []
    for lay in LAYOUTS:
        c = Cfg(obs_seed=lay, **cfg_kw)
        e = CatMouse(c, B=B, seed=100 + lay)
        sts.append(play(e, strat_fn()).cpu())
    st = torch.cat(sts)  # (1024, 30) 存活时间
    rows.append({"tag": tag, **cfg_kw, "surv": st})
    return st


def mixed():
    s = torch.full((B, 30), RUN, device="cuda")
    s[:, 20:] = HIDE
    return s


t0 = time.time()
grid = list(itertools.product([0, 4, 8, 14, 20], [1e9, 20.0], [1.0, 0.5], [False, True]))
for n_obs, stam, sal, hid in grid:
    st = run(dict(n_obs=n_obs, stamina=stam, s=sal, hidden_info=hid), mixed, "mixed")
    d = (st[:, 20:].mean(1) - st[:, 10:20].mean(1))
    print(f"[{time.time()-t0:5.0f}s] mixed K={n_obs:2d} stam={stam:g} s={sal} hid={hid}: "
          f"run {st[:,10:20].mean():.1f} hide {st[:,20:].mean():.1f} diff {d.mean():+.1f}", flush=True)

for n_obs, sal in itertools.product([0, 4, 8, 14, 20], [1.0, 0.5]):
    for strat in (RUN, HIDE):
        st = run(dict(n_obs=n_obs, stamina=20.0, s=sal, hidden_info=False),
                 lambda: torch.full((B, 30), strat, device="cuda"), "homo_" + ("hide" if strat == HIDE else "run"))
        print(f"[{time.time()-t0:5.0f}s] homo {'hide' if strat==HIDE else 'run '} K={n_obs:2d} s={sal}: {st[:,10:].mean():.1f}", flush=True)
    torch.save(rows, OUT)

# C. 视野角：120° 对比 360°（体力 20 s、知道谁是猫）
for n_obs, sal in itertools.product([0, 8, 20], [1.0, 0.5]):
    st = run(dict(n_obs=n_obs, stamina=20.0, s=sal, hidden_info=False, fov=120.0), mixed, "mixed_fov120")
    d = (st[:, 20:].mean(1) - st[:, 10:20].mean(1))
    print(f"[{time.time()-t0:5.0f}s] fov120 K={n_obs:2d} s={sal}: run {st[:,10:20].mean():.1f} hide {st[:,20:].mean():.1f} diff {d.mean():+.1f}", flush=True)
    torch.save(rows, OUT)

torch.save(rows, OUT)
print("saved", OUT)
