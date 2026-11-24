"""上下两轮之间的策略转化（演化博弈分析）。
1) 收益曲线：躲的比例为 x 时，躲者和跑者各自的平均存活（按局自举）。
2) 外部性：别人跑得越多，躲者自己活得越短吗？（看躲者收益随 x 的变化）
3) 个体模仿：第一轮后每只老鼠随机看同局一个同伴，按费米规则
   p = 1 / (1 + exp(-(π_对方 - π_自己) / K)) 改用对方策略。统计“跑→躲”“躲→跑”的比例和第二轮躲的比例。
"""
import numpy as np
import torch

from campus_scripted import HIDE, RUN
from plot_probe import boot_ci

rows = torch.load("results/campus_imitate.pt", weights_only=False)
rng = np.random.default_rng(0)
res = []
for r in rows:
    x = r["x"]
    s = r["strat"][:, 10:].numpy()
    v = r["surv"][:, 10:].numpy() / 60
    rec = {"x": x}
    for k, n in ((HIDE, "hide"), (RUN, "run")):
        m = s == k
        if m.any():
            g = (v * m).sum(1) / np.maximum(m.sum(1), 1)
            g = g[m.sum(1) > 0]
            rec[n] = boot_ci(g)
    rec["all"] = boot_ci(v.mean(1))
    if 0 < x < 1:
        mh, mr = s == HIDE, s == RUN
        d = (v * mh).sum(1) / mh.sum(1) - (v * mr).sum(1) / mr.sum(1)
        rec["diff"] = boot_ci(d)
        # 个体模仿：每只老鼠看同局随机一个同伴
        for K in (1.0, 3.0):
            sw_rh, sw_hr = [], []
            for _ in range(20):
                other = (np.arange(20)[None] + rng.integers(1, 20, size=s.shape)) % 20
                so = np.take_along_axis(s, other, 1)
                vo = np.take_along_axis(v, other, 1)
                p = 1 / (1 + np.exp(-(vo - v) / K))
                sw = (rng.random(s.shape) < p) & (so != s)
                sw_rh.append((sw & (s == RUN)).sum() / (s == RUN).sum())
                sw_hr.append((sw & (s == HIDE)).sum() / (s == HIDE).sum())
            a, b = float(np.mean(sw_rh)), float(np.mean(sw_hr))
            x2 = x + (1 - x) * a - x * b
            rec[f"K{K:g}"] = (a, b, x2)
    res.append(rec)
    fmt = lambda t: f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]"
    line = f"躲的比例 {x:.2f}: 全体 {fmt(rec['all'])}"
    if "hide" in rec:
        line += f" | 躲者 {fmt(rec['hide'])}"
    if "run" in rec:
        line += f" | 跑者 {fmt(rec['run'])}"
    if "diff" in rec:
        line += f" | 躲−跑（同局） {fmt(rec['diff'])}"
    print(line)
    for K in (1.0, 3.0):
        if f"K{K:g}" in rec:
            a, b, x2 = rec[f"K{K:g}"]
            print(f"    费米模仿 K={K:g} min：跑→躲 {a:.1%}，躲→跑 {b:.1%}，第二轮躲的比例 {x:.2f} → {x2:.2f}")
torch.save(res, "results/campus_imitate_stats.pt")
