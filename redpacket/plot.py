"""红包结果图：同一数据画两版——Nature 风格（按 nature-figure 规范）与 GraphPad Prism 风格仿制。"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sim import simulate

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
NS = (5, 10, 20)
TRIALS = 1_000_000

results = {n: simulate(n=n, trials=TRIALS) for n in NS}

# ---------- Nature 风格 ----------
NATURE = {
    "font.family": ["Arial", "SimHei"],  # 西文 Arial，中文回退黑体
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 7,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.unicode_minus": False,
}
COLORS = {5: "#B4C0E4", 10: "#7884B4", 20: "#484878"}  # NMI pastel 同色系，由浅到深

with mpl.rc_context(NATURE):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(183 / 25.4, 62 / 25.4), constrained_layout=True)
    for n in NS:
        mean, std, p = results[n]
        x = np.arange(1, n + 1) / n  # 横轴用相对顺序，三种人数可比
        ax_a.plot(x, p * n, "-o", color=COLORS[n], ms=2.5, lw=1.2, label=f"{n} 个红包")
        ax_b.plot(x, std / mean, "-o", color=COLORS[n], ms=2.5, lw=1.2)
    ax_a.axhline(1, color="#A8A8A8", lw=0.8, ls="--")
    ax_a.text(0.02, 1.01, "人人均等 (=1/n)", color="#606060", va="bottom", fontsize=6)
    ax_a.set(xlabel="抢红包的相对顺序（第 k 个 / 共 n 个）", ylabel="手气最佳概率 ÷ (1/n)")
    ax_a.legend(loc="upper center", ncol=3, handlelength=1.5)
    ax_b.set(xlabel="抢红包的相对顺序（第 k 个 / 共 n 个）", ylabel="金额变异系数（标准差 / 均值）")
    ax_b.text(0.03, 0.97, "每个位置的期望都相同，\n但越往后金额波动越大", transform=ax_b.transAxes, va="top", fontsize=6, color="#606060")
    for ax, tag in ((ax_a, "a"), (ax_b, "b")):
        ax.text(-0.14, 1.04, tag, transform=ax.transAxes, fontsize=8, fontweight="bold")
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"redpacket_nature.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ---------- GraphPad Prism 风格仿制（粗轴线、外向刻度、粗体、实心标记） ----------
PRISM = {
    "font.family": ["Arial", "SimHei"],
    "font.size": 11,
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.linewidth": 2,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "xtick.major.width": 2,
    "ytick.major.width": 2,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
}
with mpl.rc_context(PRISM):
    fig, ax = plt.subplots(figsize=(5, 3.8), constrained_layout=True)
    mean, std, p = results[10]
    k = np.arange(1, 11)
    se = np.sqrt(p * (1 - p) / TRIALS)
    ax.bar(k, p * 100, color="#4A6FE3", edgecolor="black", lw=1.5, width=0.7, yerr=1.96 * se * 100, capsize=4)
    ax.axhline(10, color="black", lw=1.5, ls="--")
    ax.set(xlabel="第几个抢", ylabel="成为手气最佳的概率 (%)", xticks=k, ylim=(0, 13.5))
    ax.set_title("100 元 / 10 个红包（10⁶ 次仿真）", fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"redpacket_prism_style.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
print("saved to", OUT)
