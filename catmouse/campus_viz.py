"""校园地图可视化：底图 + 一局里每个人的轨迹 / 动画。"""
import json

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

C_CAT, C_RUN, C_HIDE, C_PPO = "#C0504D", "#E8A33D", "#484878", "#3A8F6B"
MAP_CMAP = ListedColormap(["#F4F1EA", "#B8B8B8", "#8FBF8F", "#D98880", "#3A3A3A"])


def draw_map(ax, env, labels=True, crop=None):
    g = env.m["grid"].cpu().numpy()
    r = env.m["res"]
    ax.imshow(g, cmap=MAP_CMAP, vmin=0, vmax=4, extent=[0, g.shape[1] * r, g.shape[0] * r, 0], interpolation="nearest")
    if labels:
        for k, (x, y) in env.m["labels"].items():
            ax.text(x, y, k.replace("楼", ""), ha="center", va="center", fontsize=6.5, color="#303030", zorder=5, path_effects=[__import__("matplotlib.patheffects", fromlist=["x"]).withStroke(linewidth=1.5, foreground="white")])
    if crop:
        ax.set_xlim(crop[0], crop[2]); ax.set_ylim(crop[3], crop[1])
    ax.set_xticks([]); ax.set_yticks([])


class Recorder:
    def __init__(self, b=0, every=2):
        self.b, self.every, self.k = b, every, 0
        self.P, self.C, self.NPC, self.T = [], [], [], []

    def __call__(self, env):
        self.k += 1
        if self.k % self.every:
            return
        self.P.append(env.pos[self.b].cpu().numpy().copy())
        self.C.append(env.is_cat[self.b].cpu().numpy().copy())
        self.NPC.append(env.npc[self.b].cpu().numpy().copy() if env.M else np.zeros((0, 2)))
        self.T.append(env.t)


def gif(env, rec, strat, path, crop=None, fps=12):
    from matplotlib.animation import FuncAnimation, PillowWriter
    fig, ax = plt.subplots(figsize=(4, 5), dpi=110)
    draw_map(ax, env, crop=crop)
    P = np.array(rec.P); C = np.array(rec.C)
    col0 = np.array([C_RUN if s == 0 else C_PPO if s == 9 else C_HIDE for s in strat])
    sc = ax.scatter(P[0, :, 0], P[0, :, 1], s=8, c=col0, zorder=3, edgecolors="white", linewidths=0.3)
    npc = ax.scatter([], [], s=6, c="#909090", marker="x", zorder=3)
    tt = ax.set_title("", fontsize=7, fontfamily=["Times New Roman", "SimSun"])

    def upd(i):
        sc.set_offsets(P[i]); sc.set_color(np.where(C[i], C_CAT, col0))
        if len(rec.NPC[i]):
            npc.set_offsets(rec.NPC[i])
        tt.set_text(f"t = {rec.T[i]:.0f} s   猫 {C[i].sum()} / 老鼠 {(~C[i]).sum()}")
        return sc, npc, tt
    an = FuncAnimation(fig, upd, frames=len(P), blit=False)
    an.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
