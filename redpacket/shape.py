"""红包形状随人数变化的总览图（Nature 风格）。
a：热图，纵轴红包个数 n，横轴相对顺序，颜色为"手气最佳概率 ÷ 1/n"。
b：第一位与最后一位的相对优势随 n 的变化，标出交叉点。
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from sim import simulate

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
NS = np.arange(2, 31)
TRIALS = 1_000_000

ratios = {int(n): simulate(n=int(n), trials=TRIALS)[2] * n for n in NS}
first = np.array([ratios[n][0] for n in NS])
last = np.array([ratios[n][-1] for n in NS])
cross = int(NS[np.argmax(last > first)])

NATURE = {
    "font.family": ["Arial", "SimHei"],
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.unicode_minus": False,
}
# 蓝（低于均等）— 白 — 暖红（高于均等），柔和色
CMAP = LinearSegmentedColormap.from_list("nmi_div", ["#484878", "#B4C0E4", "#F7F7F7", "#F2B8A2", "#C0504D"])
BLUE, RED = "#7884B4", "#C0504D"

with mpl.rc_context(NATURE):
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(183 / 25.4, 66 / 25.4), layout="constrained", gridspec_kw={"width_ratios": [1.25, 1]}
    )
    fig.get_layout_engine().set(w_pad=0.12)
    norm = TwoSlopeNorm(vmin=0.6, vcenter=1.0, vmax=2.0)
    for n in NS:
        r = ratios[int(n)]
        edges = np.arange(n + 1) / n
        mesh = ax_a.pcolormesh(edges, [n - 0.5, n + 0.5], np.clip(r, 0.6, 2.0)[None, :], cmap=CMAP, norm=norm, shading="flat")
    ax_a.axhline(cross - 0.5, color="#404040", lw=0.6, ls="--")
    ax_a.text(0.5, cross - 0.5, f"n = {cross}：最后一位开始超过第一位", ha="center", va="center", fontsize=6, color="#303030",
             bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
    ax_a.set(xlim=(0, 1), ylim=(1.5, 30.5), xlabel="抢红包的相对顺序（第 k 个 / 共 n 个）", ylabel="红包个数 n")
    ax_a.invert_yaxis()
    cb = fig.colorbar(mesh, ax=ax_a, shrink=0.9, pad=0.02)
    cb.set_label("手气最佳概率 ÷ (1/n)", labelpad=3)
    cb.outline.set_linewidth(0.6)
    cb.set_ticks([0.6, 0.8, 1.0, 1.5, 2.0])

    ax_b.plot(NS, first, "-o", color=BLUE, ms=2.2, lw=1.2, label="第一个抢")
    ax_b.plot(NS, last, "-o", color=RED, ms=2.2, lw=1.2, label="最后一个抢")
    ax_b.axhline(1, color="#A8A8A8", lw=0.8, ls="--")
    ax_b.axvline(cross, color="#404040", lw=0.6, ls=":")
    ax_b.fill_between(NS, first, last, where=first >= last, color=BLUE, alpha=0.15, lw=0, interpolate=True)
    ax_b.fill_between(NS, first, last, where=last >= first, color=RED, alpha=0.15, lw=0, interpolate=True)
    ax_b.text(2.0, 1.16, "人少：\n先抢占便宜", color=BLUE, fontsize=6, va="bottom")
    ax_b.text(14, 2.2, "人多：后抢占便宜", color=RED, fontsize=6)
    ax_b.text(29.5, 2.62, "最后一个抢", color=RED, fontsize=6, ha="right", va="bottom")
    ax_b.text(29.5, 0.84, "第一个抢", color=BLUE, fontsize=6, ha="right", va="top")
    ax_b.set(xlabel="红包个数 n", ylabel="手气最佳概率 ÷ (1/n)", xlim=(1.5, 30.5), ylim=(0.75, 2.8))

    for ax, tag in ((ax_a, "a"), (ax_b, "b")):
        ax.text(-0.13, 1.04, tag, transform=ax.transAxes, fontsize=8, fontweight="bold")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"redpacket_shape_vs_n.{ext}", dpi=300, bbox_inches="tight")
print("crossover n =", cross)
