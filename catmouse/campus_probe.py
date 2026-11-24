"""校园地图探针（脚本策略）。
E1 混合：同一局 20 只老鼠，跑 / 躲(灌木或夹缝) / 只躲灌木 / 只躲夹缝 / 原地不动 各 4 只，配对比较；
   扫"发现概率"参数和夜跑路人数量，看结论对参数有多敏感。
E2 哪里最好躲：20 只老鼠开局各自被分到全图随机一个藏身点，记录藏身点位置和存活时间。
E3 同质对照：全跑 / 全躲，看跑的人有没有连累别人。
"""
import time

import torch

from campus import CCfg, Campus
from campus_scripted import FREEZE, HIDE, HIDE_BUSH, HIDE_GAP, HIDE_RANDOM, RUN, play

B = 256
OUT = "results/campus_probe.pt"
import os
rows = torch.load(OUT, weights_only=False) if os.path.exists(OUT) else []  # 续跑：已完成的条件直接跳过
done = {(r["tag"], r.get("seed", 0)) for r in rows}
t0 = time.time()


def run(tag, cfg_kw, strat, seed=0):
    if (tag, seed) in done:
        print("skip", tag, seed, flush=True)
        return
    env = Campus(CCfg(**cfg_kw), B=B, seed=seed)
    st = play(env, strat)
    row = {"tag": tag, "seed": seed, **cfg_kw, "strat": strat.cpu(), "surv": st.cpu(), "catches": env.catches.cpu(),
           "catcher": env.catcher.cpu(), "first_spot": getattr(env, "first_spot", None)}
    if row["first_spot"] is not None:
        row["first_spot"] = row["first_spot"].cpu()
    rows.append(row)
    torch.save(rows, OUT)
    m = st[:, 10:]
    names = {RUN: "跑", HIDE: "躲", HIDE_BUSH: "灌木", HIDE_GAP: "夹缝", FREEZE: "不动", HIDE_RANDOM: "随机点"}
    s = strat[:, 10:]
    parts = [f"{names[k]} {m[s == k].mean():.0f}" for k in names if (s == k).any()]
    print(f"[{time.time()-t0:5.0f}s] {tag} {cfg_kw}: " + " | ".join(parts) + f" | 1h 存活率 {(m >= 3600).float().mean():.2f}", flush=True)


def mixed():
    s = torch.empty(B, 30, dtype=torch.long, device="cuda")
    s[:, :10] = RUN
    for i, k in enumerate((RUN, HIDE, HIDE_BUSH, HIDE_GAP, FREEZE)):
        s[:, 10 + 4 * i: 14 + 4 * i] = k
    return s


def homo(k):
    return torch.full((B, 30), k, dtype=torch.long, device="cuda")


CONDS = [
    ("base", {}),
    ("still_same", dict(f_still=1.0)),  # 站着不动不比走动更难被发现
    ("terrain_same", dict(f_bush=1.0, f_gap=1.0)),  # 灌木、夹缝不起作用
    ("conceal_same", dict(f_still=1.0, f_bush=1.0, f_gap=1.0)),  # 两者都取消：怎么藏、动不动都一样显眼
    ("no_npc", dict(n_npc=0)),
    ("many_npc", dict(n_npc=20)),
]
for tag, kw in CONDS:
    run("mixed_" + tag, kw, mixed())
run("random_spot", {}, homo(HIDE_RANDOM), seed=1)
run("random_spot", {}, homo(HIDE_RANDOM), seed=2)
run("homo_run", {}, homo(RUN))
run("homo_hide", {}, homo(HIDE))
print("saved", OUT)
