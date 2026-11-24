"""报告里引用的数字（一轮 30 分钟）。
存活时间是 30 分钟内的“截断平均存活”：没被抓的人记 30 分钟。
置信区间一律按“局”重采样（同一局里的老鼠不独立）。"""
import numpy as np
import torch

from campus import GAP, BUSH, CCfg, Campus
from campus_scripted import FREEZE, HIDE, HIDE_BUSH, HIDE_GAP, RUN
from plot_probe import boot_ci

T = CCfg().T / 60
rows = torch.load("results/campus_probe.pt", weights_only=False)
names = {RUN: "跑", FREEZE: "不动", HIDE_BUSH: "灌木", HIDE: "躲", HIDE_GAP: "夹缝"}


def game_mean(v, m):
    return ((v * m).sum(1) / m.sum(1).clamp(min=1)).numpy()


for r in rows:
    s, v = r["strat"][:, 10:], r["surv"][:, 10:].float() / 60
    if r["tag"] == "random_spot":
        continue
    if not r["tag"].startswith("mixed"):
        mu, lo, hi = boot_ci(v.mean(1).numpy())
        print(f"{r['tag']}: 每局人均 {mu:.2f} [{lo:.2f}, {hi:.2f}] min，撑满 {T:.0f} 分钟的比例 {(v >= T).float().mean():.3f}")
        continue
    print(f"== {r['tag']}（{s.shape[0]} 局）")
    run = game_mean(v, s == RUN)
    for k, n in names.items():
        g = game_mean(v, s == k)
        mu, lo, hi = boot_ci(g)
        dm, dl, dh = boot_ci(g - run)
        print(f"  {n}: {mu:.2f} [{lo:.2f}, {hi:.2f}] | 比跑 {dm:+.2f} [{dl:+.2f}, {dh:+.2f}] | 撑满比例 {(v[s == k] >= T).float().mean():.2f}")

# 随机藏身点：按“开局被分到的目的地类型”分组（老鼠逃跑后可能换点，所以不是“藏好后”的安全性）
rs = [r for r in rows if r["tag"] == "random_spot"]
if rs:
    env = Campus(CCfg(), B=1)
    for r in rs:
        sp = r["first_spot"][:, 10:]
        ty = env.cell_type(sp.to("cuda")).cpu()
        v = r["surv"][:, 10:].float() / 60
        r["_ty"], r["_v"] = ty, v
    ty = torch.cat([r["_ty"] for r in rs]); v = torch.cat([r["_v"] for r in rs])
    out = {}
    for t, n in ((BUSH, "灌木"), (GAP, "夹缝")):
        out[n] = game_mean(v, ty == t)
        mu, lo, hi = boot_ci(out[n])
        print(f"随机目的地={n}: {mu:.2f} [{lo:.2f}, {hi:.2f}] min（{int((ty == t).sum())} 只）")
    d = out["夹缝"] - out["灌木"]
    ok = ~np.isnan(d)
    dm, dl, dh = boot_ci(d[ok])
    print(f"  夹缝 − 灌木（同局配对）: {dm:+.2f} [{dl:+.2f}, {dh:+.2f}]")
