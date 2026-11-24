"""校园地图实验作图（Nature 风格）。
图 3：真实地图上的躲 vs 跑（campus_probe）。
图 4：哪里最好躲（随机藏身点实验，画在地图上）。
图 5：猫鼠一起学，学出了什么（campus_eval）。
用法：python plot_campus.py 3 4 5
"""
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from campus import BUSH, GAP, CCfg, Campus
from campus_scripted import FREEZE, HIDE, HIDE_BUSH, HIDE_GAP, RUN
from campus_viz import draw_map
from plot_probe import NATURE, boot_ci, tag
import nfig

OUT = Path("figures")
C_RUN, C_HIDE, C_PPO, C_CAT = "#E8A33D", "#484878", "#3A8F6B", "#C0504D"
STRATS = [(RUN, "跑", C_RUN), (FREEZE, "原地不动", "#B4A0C8"), (HIDE_BUSH, "只躲灌木", "#7FA97F"),
          (HIDE, "躲(灌木或夹缝)", C_HIDE), (HIDE_GAP, "只躲夹缝", "#C0504D")]
CONDS = [("base", "基准"), ("still_same", "静止\n无优势"), ("terrain_same", "地形\n无优势"), ("conceal_same", "都\n无优势"),
         ("no_npc", "无\n路人"), ("many_npc", "20个\n路人")]


def fig3():
    rows = torch.load("results/campus_probe.pt", weights_only=False)
    R = {r["tag"]: r for r in rows if not r["tag"].startswith("random")}
    with mpl.rc_context(NATURE):
        fig, axs = plt.subplots(1, 3, figsize=(146 / 25.4, 70 / 25.4), gridspec_kw={"width_ratios": [1, 1.75, 0.75], "wspace": 0.4})
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.2, top=0.93)
        ax = axs[0]
        r = R["mixed_base"]
        T = 1800
        ts = np.linspace(0, T, 200)
        for k, lab, col in STRATS:
            s = r["surv"][:, 10:][r["strat"][:, 10:] == k].numpy()
            ax.plot(ts / 60, [(s >= t).mean() for t in ts], color=col, lw=1.2, label=lab)
        ax.set(xlabel="时间 (min)", ylabel="还没被抓的比例", xlim=(0, 30), ylim=(0, 1.02))
        ax.legend(loc="upper right", fontsize=7, handlelength=1.2)
        tag(ax, "a")
        ax = axs[1]
        w = 0.16
        for i, (k, lab, col) in enumerate(STRATS):
            ys, es = [], []
            for c, _ in CONDS:
                rr = R["mixed_" + c]
                m = rr["strat"][:, 10:] == k
                per_game = (rr["surv"][:, 10:] * m).sum(1) / m.sum(1)
                mu, lo, hi = boot_ci(per_game.numpy() / 60)
                ys.append(mu); es.append((mu - lo, hi - mu))
            ax.bar(np.arange(len(CONDS)) + (i - 2) * w, ys, w, color=col, yerr=np.array(es).T, error_kw=dict(lw=0.5, capsize=1), label=lab)
        ax.set(xticks=range(len(CONDS)), ylabel="平均存活时间 (min)")
        ax.set_xticklabels([c[1] for c in CONDS], fontsize=7)
        tag(ax, "b")
        ax = axs[2]
        vals = []
        for name, key, sl, col, hatch in (("全跑", "homo_run", slice(10, 30), C_RUN, ""),
                                          ("混合·跑", "mixed_base", None, C_RUN, "////"),
                                          ("混合·躲", "mixed_base", None, C_HIDE, "////"),
                                          ("全躲", "homo_hide", slice(10, 30), C_HIDE, "")):
            rr = R[key]
            if sl is None:
                k = RUN if "跑" in name else HIDE
                m = rr["strat"][:, 10:] == k
                pg = (rr["surv"][:, 10:] * m).sum(1) / m.sum(1)
            else:
                pg = rr["surv"][:, sl].mean(1)
            mu, lo, hi = boot_ci(pg.numpy() / 60)
            vals.append((name, mu, lo, hi, col, hatch))
        for i, (name, mu, lo, hi, col, hatch) in enumerate(vals):
            ax.bar(i, mu, 0.7, color=col, alpha=0.4 if hatch else 1.0, edgecolor=col, lw=0.8,
                   yerr=[[mu - lo], [hi - mu]], error_kw=dict(lw=0.6, capsize=1.2))
        ax.set(xticks=range(4), ylabel="平均存活时间 (min)")
        ax.set_xticklabels([v[0].replace("·", "\n") for v in vals], fontsize=7)
        tag(ax, "c")
        nfig.save(fig, OUT / "fig3_campus_hide_vs_run.pdf", exemptions=[{"panels": ["b", "c"], "checks": ["panel-width"], "reason": "b 有 5 组条件，故意加宽；c 只有 4 根柱"}])
        plt.close(fig)


def fig4():
    rows = [r for r in torch.load("results/campus_probe.pt", weights_only=False) if r["tag"] == "random_spot"]
    env = Campus(CCfg(), B=1)
    sp = torch.cat([r["first_spot"][:, 10:].reshape(-1, 2) for r in rows]).numpy()
    sv = torch.cat([r["surv"][:, 10:].reshape(-1) for r in rows]).numpy() / 60
    ty = env.cell_type(torch.as_tensor(sp, device="cuda")).cpu().numpy()
    rally = env.m["rally"].cpu().numpy()
    with mpl.rc_context(NATURE):
        fig = plt.figure(figsize=(146 / 25.4, 108 / 25.4))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], hspace=0.35, wspace=0.32, left=0.03, right=0.99, bottom=0.1, top=0.95)
        ax = fig.add_subplot(gs[:, 0])
        draw_map(ax, env)
        # 按 12 m 网格聚合
        cell = 12.0
        key = np.floor(sp / cell).astype(int)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        mean = np.bincount(inv, sv) / np.bincount(inv)
        cnt = np.bincount(inv)
        ok = cnt >= 15
        xy = (uk[ok] + 0.5) * cell
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=mean[ok], s=10 + cnt[ok] / cnt.max() * 30, cmap="viridis", linewidths=0, zorder=3)
        ax.scatter(*rally, marker="*", s=60, color=C_CAT, zorder=4, linewidths=0)
        ax.text(rally[0] - 6, rally[1] + 4, "集合点", fontsize=7, color=C_CAT, ha="right", va="top")
        cb = fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.01)
        cb.set_label("开局被分到这里的老鼠平均存活 (min)")
        tag(ax, "a")
        ax = fig.add_subplot(gs[0, 1])
        for i, (t, lab, col) in enumerate(((BUSH, "灌木丛", "#7FA97F"), (GAP, "空调外机夹缝", "#C0504D"))):
            mu, lo, hi = boot_ci(sv[ty == t])
            ax.bar(i, mu, 0.6, color=col, yerr=[[mu - lo], [hi - mu]], error_kw=dict(lw=0.6, capsize=1.5))
            ax.text(i, hi + 0.4, f"n = {int((ty == t).sum())}", ha="center", va="bottom", fontsize=7)
        ax.set(xticks=[0, 1], xticklabels=["灌木丛", "空调外机夹缝"], ylabel="平均存活时间 (min)", ylim=(0, 14))
        tag(ax, "b")
        ax = fig.add_subplot(gs[1, 1])
        d = np.linalg.norm(sp - rally, axis=1)
        bins = np.arange(0, d.max() + 40, 40)
        for t, lab, col in ((BUSH, "灌木丛", "#7FA97F"), (GAP, "夹缝", "#C0504D")):
            m = ty == t
            idx = np.digitize(d[m], bins)
            xs, ys, los, his = [], [], [], []
            for b in np.unique(idx):
                if (idx == b).sum() >= 100:
                    mu, lo, hi = boot_ci(sv[m][idx == b])
                    xs.append((bins[b - 1] + 20)); ys.append(mu); los.append(lo); his.append(hi)
            ax.fill_between(xs, los, his, color=col, alpha=0.2, lw=0)
            ax.plot(xs, ys, "-o", ms=2.5, lw=1.2, color=col, label=lab)
        ax.set(xlabel="藏身点离集合点的距离 (m)", ylabel="平均存活时间 (min)")
        ax.legend(fontsize=7, loc="lower left", ncol=2)
        tag(ax, "c")
        nfig.save(fig, OUT / "fig4_where_to_hide.pdf", exclude=[a for a in fig.axes if a.get_label() == "<colorbar>"])
        plt.close(fig)


def _hbar(ax, x, seeds, key, col, w=0.6, dots=True, **kw):
    """柱高 = 各种子均值的平均；误差棒 = 分层自举 95% 区间；黑点 = 各种子均值。"""
    from stats_coevo import hier
    mu, lo, hi, ms = hier(seeds, key)
    ax.bar(x, mu, w, color=col, yerr=[[mu - lo], [hi - mu]], error_kw=dict(lw=0.6, capsize=1.5), **kw)
    if dots and len(seeds) > 1:
        ax.scatter([x] * len(ms), ms, s=5, color="#202020", zorder=3, linewidths=0)
    return mu


LL, LM, SL = "learned_cat_vs_learned_mouse", "learned_cat_vs_mixed_mouse", "script_cat_vs_learned_mouse"
C_SCR = "#A8A8A8"


def fig5():
    """老鼠学到了什么：同训对手 vs 脚本对手；隐蔽无优势；历史对手池。每组 3 个种子 × 64 局。"""
    S = torch.load("results/campus_coevo_stats.pt", weights_only=False)
    G = [("indiv", "个人奖励", "#C0504D"), ("noconceal", "隐蔽无优势", "#7F7F9F"),
         ("pool", "历史对手池", "#3A8F6B"), ("poollat", "只用最新对手", "#9FCBB5")]
    base = [S["baseline"]]
    with mpl.rc_context(NATURE):
        fig = plt.figure(figsize=(146 / 25.4, 118 / 25.4))
        gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], wspace=0.3, hspace=0.6, left=0.08, right=0.99, bottom=0.1, top=0.95)
        # a 老鼠存活：对同训的猫（深） vs 对脚本猫（浅）
        ax = fig.add_subplot(gs[0, 0])
        w = 0.36
        for i, (gk, lab, col) in enumerate(G):
            _hbar(ax, i - w / 2, S[(gk, LL)], "surv", col, w=w)
            _hbar(ax, i + w / 2, S[(gk, SL)], "surv", col, w=w, alpha=0.4)
        ax.axhline(base[0]["surv"].mean(), color=C_SCR, lw=0.8, ls=":")
        ax.set(xticks=range(len(G)), ylabel="AI 老鼠的平均存活 (min)", ylim=(0, 31))
        ax.set_xticklabels([g[1].replace("无", "\n无").replace("对手", "\n对手") for g in G], fontsize=7)
        ax.set_title("深：对同训的猫　浅：对脚本猫", fontsize=6.5, color="#606060", loc="right")
        tag(ax, "a")
        # b 猫：对上脚本老鼠（越低越强）
        ax = fig.add_subplot(gs[0, 1])
        _hbar(ax, 0, base, "surv", C_SCR)
        for i, (gk, lab, col) in enumerate([G[0], G[2], G[3]], 1):
            _hbar(ax, i, S[(gk, LM)], "surv", col)
        ax.set(xticks=range(4), ylabel="脚本老鼠的平均存活 (min)", ylim=(0, 31))
        ax.set_xticklabels(["脚本猫", "个人\n奖励", "历史\n对手池", "只用最\n新对手"], fontsize=7)
        ax.set_title("对手：脚本老鼠（越低猫越强）", fontsize=6.5, color="#606060")
        tag(ax, "b")
        # c 老鼠行为（对上一起训练的猫）
        ax = fig.add_subplot(gs[1, :])
        keys = [("still", "静止"), ("bush", "在灌木丛"), ("gap", "在夹缝"), ("sprint", "冲刺"), ("near_bld", "贴楼 2 m 内")]
        w = 0.18
        for i, (gk, lab, col) in enumerate(G):
            for k, (key, _) in enumerate(keys):
                _hbar(ax, k + (i - 1.5) * w, S[(gk, LL)], key, col, w=w, label=lab if k == 0 else None)
        ax.set(xticks=range(len(keys)), ylabel="老鼠的时间占比", ylim=(0, 0.95))
        ax.set_xticklabels([k[1] for k in keys], fontsize=7)
        ax.legend(loc="upper left", fontsize=6.5, handlelength=1, ncol=4, frameon=False)
        ra = fig.axes[0].get_position().width / ax.get_position().width
        ax.text(-0.16 * ra, 1.05, "c", transform=ax.transAxes, fontsize=10, fontweight="bold")
        nfig.save(fig, OUT / "fig5_coevolution.pdf")
        plt.close(fig)


def fig7():
    """猫的奖励方式：训练曲线、对上脚本老鼠的强弱、按场上猫数分层的包围度。"""
    S = torch.load("results/campus_coevo_stats.pt", weights_only=False)
    G = [("indiv", "个人", "#C0504D"), ("near", "按参与分配", "#D98C3F"), ("rand", "随机分配", "#5B8DB8"), ("mix", "混合", "#8C6BB1"), ("team", "团队", "#E6A5A3")]
    with mpl.rc_context(NATURE):
        fig, axs = plt.subplots(1, 3, figsize=(146 / 25.4, 62 / 25.4), gridspec_kw={"wspace": 0.45})
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.24, top=0.9)
        # a 训练曲线：三个种子的平均（粗）与各种子（细）
        ax = axs[0]
        for gk, lab, col in G:
            ys = []
            for sd in (0, 1, 2):
                lg = [x for x in S[(gk, sd, "log")] if "surv" in x]
                it = [x["it"] for x in lg]; y = [x["surv"] / 60 for x in lg]
                ax.plot(it, y, color=col, lw=0.4, alpha=0.6)
                ys.append(y)
            n = min(map(len, ys))
            ax.plot(it[:n], np.mean([y[:n] for y in ys], 0), color=col, lw=1.4, label=lab)
        ax.set(xlabel="训练轮数", ylabel="训练中老鼠平均存活 (min)", ylim=(0, 31))
        tag(ax, "a")
        # b 对上脚本老鼠
        ax = axs[1]
        _hbar(ax, 0, [S["baseline"]], "surv", C_SCR)
        for i, (gk, lab, col) in enumerate(G, 1):
            _hbar(ax, i, S[(gk, LM)], "surv", col)
        ax.set(xticks=range(6), ylabel="脚本老鼠的平均存活 (min)", ylim=(0, 25))
        ax.set_xticklabels(["脚本\n猫", "个人", "参与", "随机", "混合", "团队"], fontsize=7)
        ax.set_title("越低猫越强", fontsize=6.5, color="#606060")
        for t, (_, _, col) in zip(ax.get_xticklabels()[1:], G):
            t.set_color(col)
        tag(ax, "b")
        # c 包围度按场上猫数分层
        ax = axs[2]
        st = S["enc_by_ncat"]
        names = [("脚本猫", C_SCR)] + [({"indiv": "个人奖励", "near": "按参与分配", "rand": "随机分配", "mix": "混合奖励", "team": "团队奖励"}[g], c) for g, _, c in G]
        xl = ["10–12", "13–16", "17–21", "22–30"]
        for nm, col in names:
            ax.plot(range(4), [v[2] for v in st[nm]], "-o", ms=2.5, lw=1.1, color=col)
        ax.set(xticks=range(4), xlabel="抓捕时场上猫数", ylabel="抓捕瞬间的包围度", ylim=(0.1, 0.45))
        ax.set_xticklabels(xl, fontsize=7)
        tag(ax, "c")
        nfig.save(fig, OUT / "fig7_cat_reward.pdf")
        plt.close(fig)


def fig0():
    """环境：真实地图 + 发现概率随距离的变化。"""
    from matplotlib.patches import Patch
    from campus_viz import MAP_CMAP
    env = Campus(CCfg(), B=1)
    c = env.c
    rally = env.m["rally"].cpu().numpy()
    with mpl.rc_context(NATURE):
        fig = plt.figure(figsize=(146 / 25.4, 90 / 25.4))
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.15], wspace=0.25, left=0.01, right=0.98, bottom=0.14, top=0.93)
        ax = fig.add_subplot(gs[0, 0])
        draw_map(ax, env)
        ax.scatter(*rally, marker="*", s=40, color=C_CAT, zorder=4, linewidths=0)
        ax.text(rally[0] - 6, rally[1] + 4, "集合点", fontsize=7, color=C_CAT, ha="right", va="top")
        hs =[Patch(color=MAP_CMAP(i / 4), label=l) for i, l in ((0, "平地"), (1, "教学楼（不能进）"), (2, "灌木丛"), (3, "空调外机夹缝"))]
        ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=7, handlelength=1, columnspacing=1)
        sb = 100
        ax.plot([20, 20 + sb], [env.m["H"] * env.m["res"] - 12] * 2, color="#303030", lw=1)
        ax.text(20 + sb / 2, env.m["H"] * env.m["res"] - 16, "100 m", ha="center", va="bottom", fontsize=7)
        tag(ax, "a")
        ax = fig.add_subplot(gs[0, 1])
        d = np.linspace(0, 22, 400)
        g = np.clip(2 * (1 - d / c.R), 0, 1)
        for lab, f, col, ls in (("空地上走动", 1, C_RUN, "-"), ("空地上站着不动", c.f_still, "#B4A0C8", "-"),
                                ("灌木丛里走动", c.f_bush, "#7FA97F", "--"), ("灌木丛里不动", c.f_still * c.f_bush, "#7FA97F", "-"),
                                ("夹缝里不动", c.f_still * c.f_gap, "#C0504D", "-")):
            p = 1 - np.exp(-c.rate * g * f * 1.0)
            p[d < 1] = 1
            ax.plot(d, p, ls, color=col, lw=1.2, label=lab)
            p5 = p[np.searchsorted(d, 5.0)]
            ax.text(2.0, p5 + 0.015, f"{lab}  {p5 * 100:.0f}%", fontsize=7, color=col, va="bottom")
        ax.axvline(c.R / 2, color="#A8A8A8", lw=0.6, ls=":")
        ax.axvline(c.R, color="#A8A8A8", lw=0.6, ls=":")
        ax.text(c.R / 2 + 0.3, 1.03, "10 m", fontsize=7, color="#606060")
        ax.text(c.R - 4.0, 1.03, "视野 20 m", fontsize=7, color="#606060")
        ax.set(xlabel="猫和老鼠的距离 (m)", ylabel="1 秒内被发现的概率", xlim=(0, 22), ylim=(0, 1.1), yticks=[0, 0.25, 0.5, 0.75, 1])
        tag(ax, "b")
        nfig.save(fig, OUT / "fig0_environment.pdf")
        plt.close(fig)


def fig6():
    """上下两轮之间的策略转化：收益曲线 + 费米模仿下的转化比例。"""
    res = torch.load("results/campus_imitate_stats.pt", weights_only=False)
    with mpl.rc_context(NATURE):
        fig, axs = plt.subplots(1, 2, figsize=(146 / 25.4, 62 / 25.4), gridspec_kw={"wspace": 0.38})
        fig.subplots_adjust(left=0.08, right=0.98, bottom=0.2, top=0.92)
        ax = axs[0]
        for key, lab, col in (("hide", "躲者", C_HIDE), ("run", "跑者", C_RUN), ("all", "全体平均", "#909090")):
            rr = [(r["x"], *r[key]) for r in res if key in r]
            x, mu, lo, hi = map(np.array, zip(*rr))
            ax.fill_between(x, lo, hi, color=col, alpha=0.2, lw=0)
            ax.plot(x, mu, marker="o", ms=3, lw=1.2, color=col, ls="--" if key == "all" else "-")
            i = int(np.argmin(np.abs(x - 0.5)))
            ax.text(x[i], hi[i] + 0.4, lab, color=col, fontsize=7, ha="center", va="bottom")
        ax.set(xlabel="第一轮中躲的老鼠比例", ylabel="平均存活时间 (min)", xlim=(-0.05, 1.05), ylim=(0, 13), xticks=[0, 0.25, 0.5, 0.75, 1])
        tag(ax, "a")
        ax = axs[1]
        rr = [r for r in res if "K1" in r]
        x = np.array([r["x"] for r in rr])
        a = np.array([r["K1"][0] for r in rr]); b = np.array([r["K1"][1] for r in rr])
        w = 0.08
        ax.bar(x - w / 2, a * 100, w, color=C_HIDE, label="跑→躲")
        ax.bar(x + w / 2, b * 100, w, color=C_RUN, label="躲→跑")
        for xi, ai, bi, r in zip(x, a, b, rr):
            ax.text(xi, max(ai, bi) * 100 + 2, f"{xi:.2f}→{r['K1'][2]:.2f}", ha="center", fontsize=6.5, color="#404040")
        ax.set(xlabel="第一轮中躲的老鼠比例", ylabel="第二轮改策略的比例 (%)", xticks=[0.25, 0.5, 0.75], ylim=(0, 55), xlim=(0.1, 0.9))
        ax.legend(fontsize=7, loc="upper left", frameon=False)
        tag(ax, "b")
        nfig.save(fig, OUT / "fig6_between_rounds.pdf")
        plt.close(fig)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "0" in a:
        fig0()
    if "3" in a:
        fig3()
    if "4" in a:
        fig4()
    if "5" in a:
        fig5()
    if "6" in a:
        fig6()
    if "7" in a:
        fig7()
