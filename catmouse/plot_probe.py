"""探针实验作图（Nature 风格）。
图 1：躲 vs 跑 的机制扫描（probe1）。
图 2：PPO 老鼠学到了什么（eval_ppo）。
"""
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = Path("figures")
OUT.mkdir(exist_ok=True)
NATURE = {
    "font.family": ["Times New Roman", "SimSun"],  # 与正式报告一致：西文 Times New Roman，中文宋体
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8.5,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 0.8,
    "legend.frameon": False,
    "axes.unicode_minus": False,
}
C_RUN, C_HIDE, C_PPO, C_CAT = "#E8A33D", "#484878", "#3A8F6B", "#C0504D"
COMBO = {  # (体力, 显眼度) -> 颜色/标签
    (1e9, 1.0): ("#BBBBBB", "体力无限 · 动静一样显眼"),
    (1e9, 0.5): ("#B4C0E4", "体力无限 · 静止不显眼"),
    (20.0, 1.0): ("#7884B4", "体力 20 s · 动静一样显眼"),
    (20.0, 0.5): ("#484878", "体力 20 s · 静止不显眼"),
}
KS = [0, 4, 8, 14, 20]


def boot_ci(x, n=2000, seed=0):
    g = np.random.default_rng(seed)
    x = np.asarray(x)
    m = g.choice(x, (n, len(x))).mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def tag(ax, s):
    ax.text(-0.16, 1.05, s, transform=ax.transAxes, fontsize=10, fontweight="bold")


def fig1():
    rows = torch.load("results/probe1.pt", weights_only=False)
    mixed = [r for r in rows if r["tag"] == "mixed"]
    homo = [r for r in rows if r["tag"].startswith("homo")]
    with mpl.rc_context(NATURE):
        fig, axs = plt.subplots(1, 3, figsize=(146 / 25.4, 68 / 25.4), gridspec_kw={"wspace": 0.38})
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.93)
        # a 混合种群：躲 − 跑（同一局配对差），隐藏身份=否
        ax = axs[0]
        for (st, sal), (col, lab) in COMBO.items():
            ms, lo, hi = [], [], []
            for K in KS:
                r = next(r for r in mixed if r["n_obs"] == K and r["stamina"] == st and r["s"] == sal and not r["hidden_info"])
                d = (r["surv"][:, 20:].mean(1) - r["surv"][:, 10:20].mean(1)).numpy()
                m, l, h = boot_ci(d)
                ms.append(m); lo.append(l); hi.append(h)
            ax.fill_between(KS, lo, hi, color=col, alpha=0.25, lw=0)
            ax.plot(KS, ms, "-o", color=col, ms=2.5, lw=1.2, label=lab)
        ax.axhline(0, color="#A8A8A8", lw=0.8, ls="--")
        ax.set(xlabel="掩体数量（50 m × 50 m 场地）", ylabel="躲 − 跑 的平均存活时间差 (s)")
        ax.set_ylim(-13, 28)
        ax.legend(loc="upper right", fontsize=7, handlelength=1.2)
        tag(ax, "a")
        # b 隐藏身份的影响：体力 20 s、静止不显眼
        ax = axs[1]
        for hid, ls in ((False, "-"), (True, "--")):
            for strat, col, sl in (("跑", C_RUN, slice(10, 20)), ("躲", C_HIDE, slice(20, 30))):
                ys = []
                for K in KS:
                    r = next(r for r in mixed if r["n_obs"] == K and r["stamina"] == 20.0 and r["s"] == 0.5 and r["hidden_info"] == hid)
                    ys.append(r["surv"][:, sl].mean().item())
                ax.plot(KS, ys, ls, marker="o", color=col, ms=2.5, lw=1.2,
                        label=f"{strat}（{'不知道谁变了猫' if hid else '知道谁是猫'}）")
        ax.set(xlabel="掩体数量", ylabel="平均存活时间 (s)", ylim=(18, 57))
        ax.legend(loc="lower left", fontsize=7, handlelength=1.8)
        tag(ax, "b")
        # c 同质种群对照：全躲 / 全跑 vs 混合里的躲 / 跑（体力 20 s，动静一样显眼 s=1）
        ax = axs[2]
        w = 0.2
        x = np.arange(len(KS))
        series = []
        for name, col, hatch, getter in (
            ("全跑", C_RUN, "", lambda K: next(r for r in homo if r["tag"] == "homo_run" and r["n_obs"] == K and r["s"] == 1.0)["surv"][:, 10:]),
            ("混合中的跑", C_RUN, "////", lambda K: next(r for r in mixed if r["n_obs"] == K and r["stamina"] == 20.0 and r["s"] == 1.0 and not r["hidden_info"])["surv"][:, 10:20]),
            ("混合中的躲", C_HIDE, "////", lambda K: next(r for r in mixed if r["n_obs"] == K and r["stamina"] == 20.0 and r["s"] == 1.0 and not r["hidden_info"])["surv"][:, 20:]),
            ("全躲", C_HIDE, "", lambda K: next(r for r in homo if r["tag"] == "homo_hide" and r["n_obs"] == K and r["s"] == 1.0)["surv"][:, 10:]),
        ):
            ys, es = [], []
            for K in KS:
                m, l, h = boot_ci(getter(K).mean(1).numpy())
                ys.append(m); es.append((m - l, h - m))
            es = np.array(es).T
            i = len(series)
            ax.bar(x + (i - 1.5) * w, ys, w, color=col, alpha=1.0 if not hatch else 0.4, edgecolor=col, lw=0.8,
                   yerr=es, error_kw=dict(lw=0.6, capsize=1.2), label=name)
            series.append(name)
        ax.set(xticks=x, xticklabels=KS, xlabel="掩体数量", ylabel="平均存活时间 (s)", ylim=(0, 85))
        ax.legend(loc="upper left", fontsize=7, ncol=2, handlelength=1.2, columnspacing=0.8)
        tag(ax, "c")
        import nfig
        nfig.save(fig, OUT / "fig1_hide_vs_run.pdf")
        plt.close(fig)


CONDS = [("real", "真实条件"), ("real_fov", "真实 + 120° 视野"), ("open", "空旷条件")]


def _ppo(ev, cond):
    return [ev[f"{cond}/ppo_s{s}"] for s in (0, 1, 2) if f"{cond}/ppo_s{s}" in ev]


def fig2():
    ev = torch.load("results/eval_ppo.pt", weights_only=False)
    with mpl.rc_context(NATURE):
        fig, axs = plt.subplots(1, 3, figsize=(146 / 25.4, 70 / 25.4),
                                gridspec_kw={"width_ratios": [1, 1.1, 1.35], "wspace": 0.38})
        fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.93)
        # a 训练曲线
        ax = axs[0]
        cc = {"real": C_PPO, "real_fov": "#1F5C44", "open": "#9BC9B3"}
        for cond, lab in CONDS:
            for i, r in enumerate(_ppo(ev, cond)):
                lg = [x for x in r["log"] if x["surv"] is not None]
                ax.plot([x["it"] for x in lg], [x["surv"] for x in lg], color=cc[cond], lw=0.8, alpha=0.85,
                        label=lab if i == 0 else None)
        ax.set(xlabel="训练轮数", ylabel="老鼠平均存活时间 (s)", ylim=(10, 125))
        ax.legend(loc="upper left", fontsize=7)
        tag(ax, "a")
        # b 存活时间
        ax = axs[1]
        x = np.arange(len(CONDS))
        for i, (name, col, lab) in enumerate((("run", C_RUN, "脚本·跑"), ("hide", C_HIDE, "脚本·躲"), ("ppo", C_PPO, "PPO（3 个种子）"))):
            ys, es = [], []
            for xi, (cond, _) in enumerate(CONDS):
                if name == "ppo":
                    vals = [r["surv"].mean().item() for r in _ppo(ev, cond)]
                    m = float(np.mean(vals)); e = (m - min(vals), max(vals) - m)
                    ax.scatter([xi + (i - 1) * 0.26] * len(vals), vals, s=4, color="black", zorder=4)
                else:
                    m, l, h = boot_ci(ev[f"{cond}/{name}"]["surv"].mean(1).numpy()); e = (m - l, h - m)
                ys.append(m); es.append(e)
            ax.bar(x + (i - 1) * 0.26, ys, 0.26, color=col, yerr=np.array(es).T, error_kw=dict(lw=0.6, capsize=1.2), label=lab)
        ax.set(xticks=x, xticklabels=[c[1].replace(" + ", "\n+ ") for c in CONDS], ylabel="平均存活时间 (s)", ylim=(0, 210))
        ax.legend(loc="upper left", fontsize=7)
        tag(ax, "b")
        # c 行为构成 + 贴掩体/贴墙
        ax = axs[2]
        cats = ["still", "walk", "jog", "sprint"]
        labs = {"still": "站着不动", "walk": "走", "jog": "慢跑", "sprint": "冲刺"}
        cols = {"still": C_HIDE, "walk": "#7884B4", "jog": "#F2C48D", "sprint": C_RUN}
        bars = [("脚本\n跑", "real/run"), ("脚本\n躲", "real/hide")] + [(f"PPO\n{lab.replace('真实 + 120° 视野', '120°').replace('条件', '')}", f"{cond}/ppo") for cond, lab in CONDS]
        for j, (lab, key) in enumerate(bars):
            if key.endswith("ppo"):
                bs = [r["beh"] for r in _ppo(ev, key.split("/")[0])]
                beh = {c: float(np.mean([b[c] for b in bs])) for c in cats + ["near_cover", "near_wall"]}
            else:
                beh = ev[key]["beh"]
            bot = 0
            for c in cats:
                ax.bar(j, beh[c], 0.62, bottom=bot, color=cols[c], label=labs[c] if j == 0 else None)
                bot += beh[c]
            ax.text(j, 1.02, f"{beh['near_cover']*100:.0f}%\n{beh['near_wall']*100:.0f}%", ha="center", va="bottom", fontsize=6.5, color="#404040")
        ax.set(xticks=range(len(bars)), xticklabels=[b[0] for b in bars], ylabel="老鼠时间占比", ylim=(0, 1.22), yticks=[0, 0.25, 0.5, 0.75, 1])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=7, handlelength=1)
        tag(ax, "c")
        import nfig
        nfig.save(fig, OUT / "fig2_ppo_behavior.pdf", exemptions=[{"panels": ["b", "c"], "checks": ["panel-width"], "reason": "c 是 6 根堆叠柱，故意加宽"}])
        plt.close(fig)


if __name__ == "__main__":
    import sys
    if "1" in sys.argv[1:]:
        fig1()
    if "2" in sys.argv[1:]:
        fig2()
