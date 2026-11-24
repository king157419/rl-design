"""评估：学出来的猫 / 老鼠 vs 脚本，存活时间、老鼠行为、猫的围剿指标、藏身位置热力。

用法：python campus_eval.py indiv team
"""
import math
import sys

import torch

import campus_ppo as P
from campus import BUSH, GAP, CCfg, Campus
from campus_scripted import HIDE, LEARNED, RUN, cat_actions, mouse_actions

B = 64  # 每个训练种子 64 局，2 个种子合计 128 局
T_EVAL = 1800.0
EVAL_SEED = 9001  # 与所有训练种子都不同


def encircle(env, victims, cat_before):
    """每个被抓的人：15 m 内有几只猫；包围度 = 1 - 最大角度空缺/2π（只算 20 m 内的猫，至少 2 只）。
    只算抓捕前就已经是猫的人，同一步刚被抓的老鼠不算围捕者。返回 (局号, 15 m 内猫数, 包围度, 当时猫数)。"""
    out = []
    bi, vi = torch.nonzero(victims, as_tuple=True)
    for b, v in zip(bi.tolist(), vi.tolist()):
        cats = cat_before[b].clone()
        d = (env.pos[b] - env.pos[b, v]).norm(dim=-1)
        k15 = int(((d < 15) & cats).sum())
        m = (d < 20) & cats
        if m.sum() >= 2:
            a = torch.atan2(*(env.pos[b, m] - env.pos[b, v]).flip(-1).T).sort().values
            gaps = torch.cat([a[1:] - a[:-1], (a[:1] + 2 * math.pi - a[-1:])])
            enc = 1 - gaps.max().item() / (2 * math.pi)
        else:
            enc = 0.0
        out.append((b, k15, enc, int(cat_before[b].sum())))
    return out


@torch.no_grad()
def match(cat_src, mouse_src, ck=None, share_r=30.0, seed=EVAL_SEED, heat=False):
    """cat_src/mouse_src: 'script' 或 'learned'；mouse_src 也可以是 'mixed'（跑/躲各半）。
    评估用独立的环境种子，并固定动作采样的随机数，结果不受调用顺序影响。"""
    torch.manual_seed(seed)
    # 评估环境与训练时规则一致（反事实组“隐蔽无优势”也在同样的规则下评估）
    env = Campus(CCfg(T=T_EVAL, **((ck or {}).get("cfg_kw") or {})), B=B, seed=seed)
    netM = netC = None
    if ck is not None:
        netM, netC = P.Net(ck["d_in"], P.N_ACT_M).cuda(), P.Net(ck["d_in"], P.N_ACT_C).cuda()
        netM.load_state_dict(ck["mouse"]); netC.load_state_dict(ck["cat"])
        share_r = ck.get("share_r", share_r)
    env.reset()
    strat = torch.full((B, 30), HIDE, device="cuda")
    if mouse_src == "mixed":
        strat[:, 10:20] = RUN
    z = lambda: torch.zeros(B, device="cuda")
    beh = dict(still=z(), bush=z(), gap=z(), sprint=z(), near_bld=z(), n=z())  # 逐局累计，便于按局自举
    enc = []
    H = torch.zeros(env.m["H"] // 8 + 1, env.m["W"] // 8 + 1, device="cuda") if heat else None
    while not env.done:
        prey = P.known_prey(env, share_r)
        has, j, dmin = P.update_mem(env, prey)
        if mouse_src == "learned" or cat_src == "learned":
            obs = P.observe(env, prey)
        if mouse_src == "learned":
            am = torch.distributions.Categorical(logits=netM(obs)[0]).sample()
            vm = P.mouse_vel(env, am)
        else:
            vm = mouse_actions(env, strat)
        if cat_src == "learned":
            ac = torch.distributions.Categorical(logits=netC(obs)[0]).sample()
            vc = P.cat_vel(env, ac, prey, has, j, dmin)
        else:
            vc = cat_actions(env)
        cat_before = env.is_cat.clone()
        newly = env.step(torch.where(env.is_cat[..., None], vc, vm))
        if newly.any():
            enc += encircle(env, newly, cat_before)
        mo = (~env.is_cat).float()
        spd = env.vel.norm(dim=-1)
        ct = env.cell_type(env.pos)
        beh["still"] += ((spd < 0.3).float() * mo).sum(-1)
        beh["bush"] += ((ct == BUSH).float() * mo).sum(-1)
        beh["gap"] += ((ct == GAP).float() * mo).sum(-1)
        beh["sprint"] += ((spd > 2.6).float() * mo).sum(-1)
        r = P.rays(env)
        beh["near_bld"] += (((r.amin(-1) * env.c.R) < 2.0).float() * mo).sum(-1)
        beh["n"] += mo.sum(-1)
        if heat:
            i = (env.pos[..., 1] / env.m["res"] / 8).long()
            jj = (env.pos[..., 0] / env.m["res"] / 8).long()
            H.index_put_((i[~env.is_cat], jj[~env.is_cat]), torch.ones_like(i[~env.is_cat], dtype=torch.float), accumulate=True)
    beh_game = {k: v.cpu() for k, v in beh.items()}  # 每局的“老鼠·步”计数
    n = max(beh["n"].sum().item(), 1)
    beh = {k: v.sum().item() / n for k, v in beh.items() if k != "n"}
    surv = env.caught_t.clamp(max=env.c.T)[:, 10:].cpu()
    c = env.catches.cpu()
    cs = c.sort(-1).values
    nn_ = cs.shape[-1]
    gini = ((2 * torch.arange(1, nn_ + 1) - nn_ - 1) * cs).sum(-1) / (nn_ * cs.sum(-1).clamp(min=1))
    # 只在开局 10 只猫之间算基尼，避免从未当过猫的人把指标抬高
    ci = c[:, :10].sort(-1).values
    gini0 = ((2 * torch.arange(1, 11) - 11) * ci).sum(-1) / (10 * ci.sum(-1).clamp(min=1))
    return dict(surv=surv, strat=strat[:, 10:].cpu(), beh=beh, beh_game=beh_game, enc=enc, catches=c,
                top=c.max(-1).values.float().mean().item(), gini=gini.mean().item(), gini0=gini0.mean().item(),
                heat=H.cpu() if heat else None)


def summarize(name, r):
    enc = torch.tensor(r["enc"], dtype=torch.float) if r["enc"] else torch.zeros(1, 4)
    pack = (enc[:, 1] >= 2).float().mean().item()
    print(f"{name}: 老鼠存活 {r['surv'].float().mean():.0f}s | 静止 {r['beh']['still']:.2f} 灌木 {r['beh']['bush']:.2f} "
          f"夹缝 {r['beh']['gap']:.2f} 冲刺 {r['beh']['sprint']:.2f} 贴楼 {r['beh']['near_bld']:.2f} | "
          f"抓捕时身边≥2猫 {pack:.2f} 包围度 {enc[:, 2].mean():.2f}（{len(r['enc'])} 次）| 最佳猫 {r['top']:.1f} 只 "
          f"基尼(开局猫) {r['gini0']:.2f}", flush=True)


def main(tags, out_path="results/campus_eval.pt", baseline=True):
    out = {}
    if baseline:
        out["script_vs_script"] = match("script", "mixed", heat=True)
        summarize("脚本猫 vs 脚本老鼠(跑躲各半)", out["script_vs_script"])
    for tag in tags:
        ck = torch.load(f"results/campus_ppo/{tag}.pt")
        for cs, ms in (("learned", "learned"), ("learned", "mixed"), ("script", "learned")):
            k = f"{tag}/{cs}_cat_vs_{ms}_mouse"
            out[k] = match(cs, ms, ck=ck, heat=(ms == "learned"))
            summarize(k, out[k])
        out[f"{tag}/log"] = ck["log"]
        torch.save(out, out_path)
    torch.save(out, out_path)


if __name__ == "__main__":
    # 用法：python campus_eval.py 输出文件 [--no-baseline] tag1 tag2 ...
    args = sys.argv[1:]
    path = args.pop(0)
    base = "--no-baseline" not in args
    main([a for a in args if a != "--no-baseline"], out_path=path, baseline=base)
