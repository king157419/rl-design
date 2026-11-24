"""录一局（第 0 局）的轨迹，画几张快照，或导出 GIF。"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from scripted import HIDE, LEARNED, RUN, cat_actions, mouse_actions

COL = {"cat": "#C0504D", RUN: "#E8A33D", HIDE: "#484878", LEARNED: "#3A8F6B"}


@torch.no_grad()
def record(env, strategy, learned_fn=None, every=4):
    env.reset()
    frames = []
    while not env.done:
        a = mouse_actions(env, strategy)
        if learned_fn is not None:
            a = torch.where((strategy == LEARNED)[..., None], learned_fn(env), a)
        a = torch.where(env.is_cat[..., None], cat_actions(env), a)
        env.step(a)
        if round(env.t / env.c.dt) % every == 0:
            frames.append((env.t, env.pos[0].cpu().clone(), env.is_cat[0].cpu().clone()))
    return frames


def draw(ax, env, frame, strategy0, trail=None):
    t, pos, cat = frame
    ax.clear()
    for x0, y0, x1, y1 in env.obs_rects.cpu().tolist():
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color="#BBBBBB", lw=0))
    if trail is not None:
        for i in range(pos.shape[0]):
            ax.plot(trail[:, i, 0], trail[:, i, 1], lw=0.4, alpha=0.4,
                    color=COL["cat"] if cat[i] else COL[int(strategy0[i])])
    cols = [COL["cat"] if cat[i] else COL[int(strategy0[i])] for i in range(pos.shape[0])]
    ax.scatter(pos[:, 0], pos[:, 1], s=14, c=cols, zorder=3, edgecolors="white", linewidths=0.3)
    ax.set(xlim=(0, env.c.W), ylim=(0, env.c.H), aspect="equal", xticks=[], yticks=[])
    ax.set_title(f"t = {t:.0f} s   猫 {int(cat.sum())}  老鼠 {int((~cat).sum())}", fontsize=7)


def gif(env, strategy, path, learned_fn=None):
    frames = record(env, strategy, learned_fn)
    s0 = strategy[0].cpu()
    with plt.rc_context({"font.family": ["Times New Roman", "SimSun"]}):
        fig, ax = plt.subplots(figsize=(3.2, 3.4), dpi=110)
        anim = FuncAnimation(fig, lambda k: draw(ax, env, frames[k], s0), frames=len(frames))
        anim.save(path, writer=PillowWriter(fps=12))
        plt.close(fig)
    return frames
