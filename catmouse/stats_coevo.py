"""猫鼠共同训练的汇总统计。
每组 3 个训练种子 × 64 局。组均值 = 各种子均值的平均；95% 区间用分层自举（先抽训练种子，再在种子内抽局），
这样既包含对局的随机性，也包含训练种子之间的差异。同时列出每个种子自己的均值。
"""
import numpy as np
import torch

GROUPS = {"indiv": "个人奖励", "team": "团队奖励", "mix": "混合奖励", "near": "按参与分配", "rand": "随机分配",
          "noconceal": "隐蔽无优势", "pool": "历史对手池", "poollat": "只用最新对手"}
SEEDS = (0, 1, 2)
MATCHES = {"learned_cat_vs_learned_mouse": "AI 猫 vs AI 老鼠", "learned_cat_vs_mixed_mouse": "AI 猫 vs 脚本老鼠",
           "script_cat_vs_learned_mouse": "脚本猫 vs AI 老鼠"}
BEH = ("still", "bush", "gap", "sprint", "near_bld")


def per_game(r):
    n = r["beh_game"]["n"].clamp(min=1).numpy()
    g = {"surv": r["surv"].float().mean(1).numpy() / 60}
    for k in BEH:
        g[k] = r["beh_game"][k].numpy() / n
    enc = np.array(r["enc"], dtype=float).reshape(-1, 4)
    B = len(g["surv"])
    pack = np.full(B, np.nan); encm = np.full(B, np.nan)
    for b in range(B):
        e = enc[enc[:, 0] == b]
        if len(e):
            pack[b] = (e[:, 1] >= 2).mean(); encm[b] = e[:, 2].mean()
    g["pack"], g["enc"] = pack, encm
    g["ncatch"] = r["catches"].sum(-1).float().numpy()
    g["_enc_raw"] = enc
    return g


def hier(seeds, key, n=2000, rng=np.random.default_rng(0)):
    """seeds: 每个种子一个按局的数组。返回 (组均值, 下界, 上界, 各种子均值)。"""
    xs = [s[key][~np.isnan(s[key])] for s in seeds]
    means = np.array([x.mean() for x in xs])
    bs = []
    for _ in range(n):
        pick = rng.integers(len(xs), size=len(xs))
        bs.append(np.mean([rng.choice(xs[i], size=len(xs[i])).mean() for i in pick]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return means.mean(), lo, hi, means


def fmt(t):
    return f"{t[0]:.2f} [{t[1]:.2f}, {t[2]:.2f}]（种子 " + " / ".join(f"{m:.2f}" for m in t[3]) + "）"


if __name__ == "__main__":
    res = {}
    base = per_game(torch.load("results/eval/baseline.pt", weights_only=False)["script_vs_script"])
    res["baseline"] = base
    print("== 基线 脚本猫 vs 脚本老鼠：存活 %.2f min，包围度 %.2f" % (base["surv"].mean(), np.nanmean(base["enc"])))
    for gk, gname in GROUPS.items():
        try:
            evs = [torch.load(f"results/eval/{gk}_s{s}.pt", weights_only=False) for s in SEEDS]
        except FileNotFoundError as e:
            print("缺少", e.filename); continue
        for mk, mname in MATCHES.items():
            seeds = [per_game(e[f"{gk}_s{s}/{mk}"]) for s, e in zip(SEEDS, evs)]
            res[(gk, mk)] = seeds
            print(f"== {gname} | {mname}：存活 " + fmt(hier(seeds, "surv")))
            print("   " + " ".join(f"{k} {hier(seeds, k)[0]:.2f}" for k in BEH)
                  + f" | 包围度 {hier(seeds, 'enc')[0]:.2f} 身边≥2猫 {hier(seeds, 'pack')[0]:.2f} 抓捕数 {hier(seeds, 'ncatch')[0]:.1f}")
        for s in SEEDS:
            ck = torch.load(f"results/campus_ppo/{gk}_s{s}.pt", weights_only=False, map_location="cpu")
            res[(gk, s, "log")] = ck["log"]
            res[(gk, s, "restarts")] = ck.get("restarts", 0)
    # 包围度按抓捕时场上猫数分层（对手固定为脚本老鼠）
    print("== 包围度按场上猫数分层（对手：脚本老鼠）")
    rows = [("脚本猫", [base])] + [(GROUPS[g], res[(g, "learned_cat_vs_mixed_mouse")]) for g in ("indiv", "team", "mix", "near", "rand")
                                    if (g, "learned_cat_vs_mixed_mouse") in res]
    strat = {}
    for name, seeds in rows:
        e = np.concatenate([s["_enc_raw"] for s in seeds])
        line = []
        for lo, hi in ((10, 13), (13, 17), (17, 22), (22, 31)):
            m = (e[:, 3] >= lo) & (e[:, 3] < hi)
            line.append((lo, hi - 1, e[m, 2].mean() if m.any() else np.nan, int(m.sum())))
        strat[name] = line
        print(f"   {name}: " + " | ".join(f"{a}-{b} 只 {v:.2f}（{n}）" for a, b, v, n in line))
    res["enc_by_ncat"] = strat
    torch.save(res, "results/campus_coevo_stats.pt")
    print("saved")
