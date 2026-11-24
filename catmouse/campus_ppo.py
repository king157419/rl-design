"""校园地图上猫、老鼠同时学习（两套 PPO，各自一张网络，所有猫共用猫网络、所有老鼠共用老鼠网络）。

老鼠被抓后改由猫网络控制。奖励：
  老鼠：每活 1 分钟 +1；被抓即终止。
  猫：reward="indiv" 时自己抓到一只 +1（对应"抓得最多的猫有奖"）；reward="team" 时全队每抓一只，每只猫 +1/猫数。
动作：
  老鼠 27 个：停 / 8 方向 × {走, 慢跑, 冲刺} / 沿路去最近藏身点（灌木或夹缝）/ 冲刺逃离。
  猫 20 个：停 / 8 方向 × {慢跑, 冲刺} / 沿路追最近已知老鼠（带拦截预判）/ 去最后见到老鼠的地方 / 巡逻（去最久没人搜过的区域）。
  "包抄"没有现成动作——如果出现，是猫自己用方向动作学出来的。
猫之间可以喊话：share_r 米以内的队友看到的老鼠，自己也知道。

用法：python campus_ppo.py --tag indiv --reward indiv --iters 600
"""
import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from campus import BUSH, GAP, CCfg, Campus
from campus_scripted import HIDE, flee_vel, pick_spot

K_OTH, K_MATE = 5, 4
A8 = torch.arange(8) * (2 * math.pi / 8)
N_ACT_M, N_ACT_C = 27, 20
NEAR_R = 15.0      # “按参与分配”奖励：抓捕时受害者多少米内的猫算参与者
SNAP_EVERY = 25    # 对手池：每多少轮存一份快照
N_OPP = 4          # 对手池：每次 reset 每方抽几份历史版本（按局轮流分配）
PATCH = [(dx, dy) for dy in range(-6, 7, 2) for dx in range(-6, 7, 2)]


def dirs(dev):
    return torch.stack([A8.cos(), A8.sin()], -1).to(dev)


def known_prey(env, share_r):
    """猫 i 知道老鼠 j 的位置：自己盯着，或 share_r 内的队友盯着。"""
    sp = env.spot & ~env.is_cat[:, None, :]
    if share_r <= 0:
        return sp & env.is_cat[..., None]
    near = (env.dist < share_r) & env.is_cat[:, None, :] & env.is_cat[:, :, None]
    sh = torch.bmm(near.float(), sp.float()) > 0
    return (sp | sh) & env.is_cat[..., None]


def update_mem(env, prey):
    # 老鼠：最后见到的威胁；猫：最后知道的老鼠
    th = env.threat
    dth = torch.where(th, env.dist, torch.full_like(env.dist, 1e9))
    d1, j1 = dth.min(-1)
    seen = d1 < 1e8
    env.last_threat = torch.where(seen[..., None], torch.gather(env.pos, 1, j1[..., None].expand(-1, -1, 2)), env.last_threat)
    env.last_seen_t = torch.where(seen, torch.full_like(env.last_seen_t, env.t), env.last_seen_t)
    env.seen_threat |= seen
    dp = torch.where(prey, env.dist, torch.full_like(env.dist, 1e9))
    d2, j2 = dp.min(-1)
    has = d2 < 1e8
    env.mem_pos = torch.where(has[..., None], torch.gather(env.pos, 1, j2[..., None].expand(-1, -1, 2)), env.mem_pos)
    env.mem_t = torch.where(has, torch.full_like(env.mem_t, env.t), env.mem_t)
    return has, j2, d2


def rays(env):
    c, D = env.c, dirs(env.dev)
    steps = torch.arange(1, int(c.R) + 1, device=env.dev).float()
    p = env.pos[:, :, None, None, :] + D[None, None, :, None, :] * steps[:, None]
    m = env.m
    i = (p[..., 1] / m["res"]).long().clamp(0, m["H"] - 1)
    j = (p[..., 0] / m["res"]).long().clamp(0, m["W"] - 1)
    blk = m["block_los"][i, j]  # (B,N,8,S)
    first = torch.where(blk, steps, torch.full_like(steps, c.R + 1)).amin(-1)
    return (first / c.R).clamp(max=1.0)


def patch(env):
    off = torch.tensor(PATCH, device=env.dev, dtype=torch.float32)
    ct = env.cell_type(env.pos[:, :, None, :] + off)
    return torch.cat([(ct == BUSH).float(), (ct == GAP).float(), (ct == 1).float() + (ct == 4).float()], -1)


def nearest(env, mask, tgt_pos, tgt_vel, dist, K):
    c = env.c
    dm = torch.where(mask, dist, torch.full_like(dist, 1e9))
    ds, idx = dm.sort(-1)
    ds, idx = ds[..., :K], idx[..., :K]
    valid = (ds < 1e8).float()
    B, N = env.B, env.N
    P = torch.gather(tgt_pos[:, None].expand(B, N, -1, 2), 2, idx[..., None].expand(-1, -1, -1, 2))
    V = torch.gather(tgt_vel[:, None].expand(B, N, -1, 2), 2, idx[..., None].expand(-1, -1, -1, 2))
    rel = (P - env.pos[:, :, None]) / c.R
    return torch.cat([rel * valid[..., None], V / c.v_sprint * valid[..., None], valid[..., None]], -1).reshape(B, N, -1)


def observe(env, prey):
    c, m = env.c, env.m
    B, N = env.B, env.N
    ct = env.cell_type(env.pos)
    own = torch.cat([
        env.pos / torch.tensor([m["Wm"], m["Hm"]], device=env.dev),
        env.vel / c.v_sprint,
        (env.stam / c.stamina).clamp(0, 1)[..., None],
        torch.full((B, N, 1), env.t / 3600.0, device=env.dev),
        torch.stack([(ct == 0), (ct == BUSH), (ct == GAP)], -1).float(),
        env.is_cat.float().mean(-1, keepdim=True).expand(B, N)[..., None],
        env.is_cat.float()[..., None],
    ], -1)
    # 对手：老鼠看"威胁"（猫 + 认不出的路人），猫看"已知老鼠"
    if env.M:
        allp = torch.cat([env.pos, env.npc], 1)
        allv = torch.cat([env.vel, env.npc_vel], 1)  # 路人用真实速度，不给网络额外的辨认线索
        alld = torch.cat([env.dist, env.npc_dist], -1)
        mth = torch.cat([env.threat, env.npc_threat], -1)
        mprey = torch.cat([prey, torch.zeros_like(env.npc_threat)], -1)
    else:
        allp, allv, alld, mth, mprey = env.pos, env.vel, env.dist, env.threat, prey
    opp_mask = torch.where(env.is_cat[..., None], mprey, mth)
    opp = nearest(env, opp_mask, allp, allv, alld, K_OTH)
    eye = torch.eye(N, dtype=torch.bool, device=env.dev)
    same = (env.is_cat[:, :, None] == env.is_cat[:, None, :]) & ~eye
    mate = nearest(env, same & env.vis, env.pos, env.vel, env.dist, K_MATE)  # 只看得见视线内的队友，不穿墙
    lt = torch.where(env.is_cat[..., None], env.mem_pos, env.last_threat)
    lt_t = torch.where(env.is_cat, env.mem_t, env.last_seen_t)
    mem = torch.cat([((lt - env.pos) / c.R).clamp(-5, 5), ((env.t - lt_t) / 60).clamp(max=1)[..., None]], -1)
    spot_d = env.hide_spot - env.pos
    hint = torch.cat([(spot_d / 50).clamp(-1, 1), ((env.wp - env.pos) / 100).clamp(-1, 1)], -1)
    return torch.cat([own, rays(env), patch(env), opp, mate, mem, hint], -1)


def mouse_vel(env, a):
    c = env.c
    D = dirs(env.dev)
    spd = torch.tensor([c.v_walk, c.v_jog, c.v_sprint], device=env.dev)
    k = (a - 1).clamp(0, 23)
    v = D[k % 8] * spd[k // 8][..., None]
    v = torch.where((a == 0)[..., None], torch.zeros_like(v), v)
    # 藏身：需要时重新挑点
    # 只改老鼠自己的藏身状态：猫那一行的“虚拟老鼠动作”不能写进猫的观测
    mo = ~env.is_cat
    near = (env.threat & (env.dist < c.flee_d)).any(-1) & mo
    env.need_spot |= near
    want = (a == 25) & env.need_spot & ~near & mo
    if want.any():
        kind = torch.ones(env.B, env.N, 2, dtype=torch.bool, device=env.dev)
        env.hide_spot = torch.where(want[..., None], pick_spot(env, kind), env.hide_spot)
        env.need_spot &= ~want
    to = env.hide_spot - env.pos
    hv = torch.where(to.norm(dim=-1, keepdim=True) > 0.4, env.nav_dir(env.pos, env.hide_spot) * c.v_walk, torch.zeros_like(to))
    v = torch.where((a == 25)[..., None], hv, v)
    if (a == 26).any():
        fv = flee_vel(env, env.threat, env.npc_threat, c.v_sprint)
        v = torch.where((a == 26)[..., None], fv, v)
    return v


def cat_vel(env, a, prey, has, j, dmin):
    c = env.c
    D = dirs(env.dev)
    spd = torch.tensor([c.v_jog, c.v_sprint], device=env.dev)
    k = (a - 1).clamp(0, 15)
    v = D[k % 8] * spd[k // 8][..., None]
    v = torch.where((a == 0)[..., None], torch.zeros_like(v), v)
    # 追
    tgt = torch.gather(env.pos, 1, j[..., None].expand(-1, -1, 2))
    tvel = torch.gather(env.vel, 1, j[..., None].expand(-1, -1, 2))
    tau = (dmin.clamp(max=1e8) / c.v_sprint).clamp(max=2.0)[..., None]
    aim = tgt + tvel * tau
    aim = torch.where(env.walkable(aim)[..., None], aim, tgt)
    goal_chase = torch.where(has[..., None], aim, env.mem_pos)
    # 巡逻（共享搜索记忆）
    dc = torch.cdist(env.pos, env.cell_ctr[None].expand(env.B, -1, -1))
    seen_cell = ((dc < env.G * 0.6) & env.is_cat[..., None]).any(1)
    env.cell_t = torch.where(seen_cell, torch.full_like(env.cell_t, env.t), env.cell_t)
    reached = ((env.wp - env.pos).norm(dim=-1) < 3.0) & env.is_cat  # 只改猫自己的巡逻目标，不把搜索记忆泄露给老鼠
    nC = env.cell_ctr.shape[0]
    cand = torch.randint(0, nC, (env.B, env.N, 4), device=env.dev, generator=env.gen)
    ctt = torch.gather(env.cell_t[:, None, :].expand(-1, env.N, -1), 2, cand)
    pick = torch.gather(cand, 2, ctt.argmin(-1, keepdim=True)).squeeze(-1)
    new_wp = env.cell_ctr[pick] + (env.rand(env.B, env.N, 2) - 0.5) * env.G
    new_wp = torch.where(env.walkable(new_wp)[..., None], new_wp, env.cell_ctr[pick])
    env.wp = torch.where(reached[..., None], new_wp, env.wp)
    goal = torch.where((a == 17)[..., None], goal_chase, torch.where((a == 18)[..., None], env.mem_pos, env.wp))
    macro = a >= 17
    nd = env.nav_dir(env.pos, goal)
    ms = torch.where((a == 17) & has, c.v_sprint, c.v_jog)[..., None]
    return torch.where(macro[..., None], nd * ms, v)


class Net(nn.Module):
    def __init__(self, d_in, n_act, h=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.pi, self.v = nn.Linear(h, n_act), nn.Linear(h, 1)

    def forward(self, x):
        z = self.body(x)
        return self.pi(z), self.v(z).squeeze(-1)


def reset(env):
    env.reset()


def ppo_update(net, opt, O, A, LP, ADV, RET, clip=0.2, epochs=4, mb=16384):
    if ADV.numel() < 2:  # 有效样本不足两个时 std 为 NaN，跳过这次更新
        return
    ADV = (ADV - ADV.mean()) / (ADV.std(unbiased=False) + 1e-8)
    n = O.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=O.device)
        for s in range(0, n, mb):
            b = perm[s:s + mb]
            logits, v = net(O[b])
            dist = torch.distributions.Categorical(logits=logits)
            lp = dist.log_prob(A[b])
            ratio = (lp - LP[b]).exp()
            pl = -torch.min(ratio * ADV[b], ratio.clamp(1 - clip, 1 + clip) * ADV[b]).mean()
            vl = 0.5 * (v - RET[b]).pow(2).mean()
            ent = dist.entropy().mean()
            opt.zero_grad()
            (pl + vl - 0.01 * ent).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
    return ent.item()


def gae(R, V, D, M_next, gamma, lam):
    """R,V[:-1],D: (T,B,N)。M_next[t]：t+1 时刻该智能体是否仍由这张网络控制。"""
    T = R.shape[0]
    adv = torch.zeros_like(R[0])
    out = torch.zeros_like(R)
    for t in reversed(range(T)):
        nt = (~D[t] & M_next[t]).float()
        delta = R[t] + gamma * V[t + 1] * nt - V[t]
        adv = delta + gamma * lam * nt * adv
        out[t] = adv
    return out


def train(tag, reward="indiv", iters=600, B=64, T_roll=128, seed=0, share_r=30.0, T_ep=1800.0, out="results/campus_ppo",
          init=None, freeze_mouse=False, freeze_cat=False, cfg_kw=None, pool=False, pool_max=24):
    torch.manual_seed(seed)
    dev = "cuda"
    env = Campus(CCfg(T=T_ep, **(cfg_kw or {})), B=B, seed=seed)
    reset(env)
    prey = known_prey(env, share_r)
    update_mem(env, prey)
    d_in = observe(env, prey).shape[-1]
    netM, netC = Net(d_in, N_ACT_M).to(dev), Net(d_in, N_ACT_C).to(dev)
    if init:
        ck = torch.load(init)
        netM.load_state_dict(ck["mouse"]); netC.load_state_dict(ck["cat"])
    optM = torch.optim.Adam(netM.parameters(), lr=3e-4)
    optC = torch.optim.Adam(netC.parameters(), lr=3e-4)
    gamma, lam = 0.995, 0.95
    log, t0, it0, snaps, it_now, restarts = [], time.time(), 0, [], [0], 0
    ck_path = Path(out) / f"{tag}.pt"
    if init is None and ck_path.exists():  # 续跑：同一 tag 已有检查点就接着训练
        ck = torch.load(ck_path, weights_only=False)
        ok = ("optM" in ck and ck.get("seed") == seed and ck.get("reward") == reward and (ck.get("cfg_kw") or {}) == (cfg_kw or {})
              and ck.get("pool", False) == pool and ck.get("pool_max", 24) == pool_max
              and ck.get("share_r", share_r) == share_r and ck.get("T_ep", T_ep) == T_ep and ck.get("B", B) == B)
        if not ok:  # 配置不一致时不要悄悄从头训练、覆盖原检查点
            raise SystemExit(f"[{tag}] 已有检查点的配置与本次参数不一致，换一个 tag 或先移走 {ck_path}")
        if ok:
            netM.load_state_dict(ck["mouse"]); netC.load_state_dict(ck["cat"])
            optM.load_state_dict(ck["optM"]); optC.load_state_dict(ck["optC"])
            log, it0 = ck["log"], ck["it"] + 1
            t0 -= log[-1]["time"] if log else 0
            snaps = ck.get("snaps", [])
            restarts = ck.get("restarts", 0) + 1  # 续跑是“热重启”：环境和随机数状态不恢复，记录次数
            torch.manual_seed(seed * 1000 + it0)
            print(f"[{tag}] resume from it {it0}", flush=True)
    # 历史对手池（虚拟自我对弈）：每 SNAP_EVERY 轮存一份双方网络，每局开始时随机抽一份当对手
    half_a = torch.arange(B, device=dev) < B // 2  # 这些局里猫是历史版本，只训练老鼠
    half_b = ~half_a                               # 这些局里老鼠是历史版本，只训练猫
    # 每次 reset 各抽 N_OPP 份历史版本，按局轮流分配（每份管 B/2/N_OPP 局），避免整批共用一个对手
    oldM = [Net(d_in, N_ACT_M).to(dev) for _ in range(N_OPP)]
    oldC = [Net(d_in, N_ACT_C).to(dev) for _ in range(N_OPP)]
    ver = torch.arange(B, device=dev) % N_OPP
    use_count = dict(ck.get("use_count", {})) if it0 > 0 else {}  # 续跑时接着累计

    def snapshot():
        snaps.append({"it": it_now[0], "mouse": {k: v.detach().cpu().clone() for k, v in netM.state_dict().items()},
                      "cat": {k: v.detach().cpu().clone() for k, v in netC.state_dict().items()}})
        del snaps[:-pool_max]

    def pick_opponents():
        if not snaps:
            snapshot()
        for n in range(N_OPP):
            i, k = torch.randint(len(snaps), (2,)).tolist()
            oldM[n].load_state_dict(snaps[i]["mouse"]); oldC[n].load_state_dict(snaps[k]["cat"])
            for s in (i, k):
                use_count[snaps[s]["it"]] = use_count.get(snaps[s]["it"], 0) + 1

    def old_logits(nets, obs):
        L = torch.stack([n(obs)[0] for n in nets])          # (N_OPP, B, N, A)
        return L[ver, torch.arange(B, device=dev)]           # 每局取分到的那一份

    if pool:
        pick_opponents()
    for it in range(it0, iters):
        it_now[0] = it
        if pool and it % SNAP_EVERY == 0 and it > 0 and (not snaps or snaps[-1]["it"] < it):
            snapshot()
        O, AM, AC, LPM, LPC, VM, VC, RM, RC, MM, MC, D = ([] for _ in range(12))
        ep_stats = []
        for _ in range(T_roll):
            if env.done:
                sv = (env.caught_t.clamp(max=env.c.T) * env.init_mouse).sum(-1) / env.init_mouse.sum(-1).clamp(min=1)
                ep_stats.append((env.caught_t[env.init_mouse].clamp(max=env.c.T).mean().item(),
                                 env.catches.max(-1).values.mean().item(), env.is_cat.float().mean().item(),
                                 sv[half_a].mean().item(), sv[half_b].mean().item()))
                reset(env)
                if pool:
                    pick_opponents()
            prey = known_prey(env, share_r)
            has, j, dmin = update_mem(env, prey)
            obs = observe(env, prey)
            cat = env.is_cat.clone()
            # 逐局终止：老鼠已经全被抓的局，后面的步不再产生训练样本
            act = (~cat.all(-1, keepdim=True)).expand_as(cat)
            with torch.no_grad():
                lm, vm = netM(obs)
                lc, vc = netC(obs)
                if pool:  # 前一半局：猫由历史版本控制；后一半局：老鼠由历史版本控制
                    lm = torch.where(half_b[:, None, None], old_logits(oldM, obs), lm)
                    lc = torch.where(half_a[:, None, None], old_logits(oldC, obs), lc)
                dm, dc = torch.distributions.Categorical(logits=lm), torch.distributions.Categorical(logits=lc)
                am, ac = dm.sample(), dc.sample()
                lpm, lpc = dm.log_prob(am), dc.log_prob(ac)
            vel = torch.where(cat[..., None], cat_vel(env, ac, prey, has, j, dmin), mouse_vel(env, am))
            before = env.catches.clone()
            env.step(vel)
            got = env.catches - before
            rm = (~cat & ~env.is_cat).float() * (env.c.dt / 60.0)
            team = cat.float() * got.sum(-1, keepdim=True) / cat.float().sum(-1, keepdim=True).clamp(min=1)
            if reward == "indiv":
                rc = got * cat.float()
            elif reward == "team":
                rc = team
            elif reward == "mix":  # 个人、团队各占一半
                rc = 0.5 * got * cat.float() + 0.5 * team
            elif reward == "near":  # 按参与分配：每次抓捕的 1 分平分给抓捕时在受害者 15 m 内的猫（差分奖励的近似）
                newly = env.is_cat & ~cat
                near = cat[:, :, None] & newly[:, None, :] & (env.dist < NEAR_R)
                rc = (near.float() / near.float().sum(1, keepdim=True).clamp(min=1)).sum(-1)
            elif reward == "rand":  # 随机分配对照：分到功劳的猫数与 near 相同，但从全体猫中随机抽，与是否在场无关
                newly = env.is_cat & ~cat
                near = cat[:, :, None] & newly[:, None, :] & (env.dist < NEAR_R)
                k = near.sum(1, keepdim=True)                                   # (B,1,N) 每个受害者该分给几只猫
                score = torch.where(cat[:, :, None], torch.rand_like(env.dist), torch.full_like(env.dist, 2.0))
                rank = score.argsort(1).argsort(1)                              # 每个受害者一列，猫的随机名次
                pick = cat[:, :, None] & newly[:, None, :] & (rank < k)
                rc = (pick.float() / pick.float().sum(1, keepdim=True).clamp(min=1)).sum(-1)
            else:
                raise ValueError(reward)
            done = torch.full_like(cat, env.done) | env.is_cat.all(-1, keepdim=True).expand_as(cat)
            O.append(obs); AM.append(am); AC.append(ac); LPM.append(lpm); LPC.append(lpc)
            mm, mc = ~cat & act, cat & act
            if pool:  # 历史版本控制的一方不产生训练样本
                mm, mc = mm & ~half_b[:, None], mc & ~half_a[:, None]
            VM.append(vm); VC.append(vc); RM.append(rm); RC.append(rc); MM.append(mm); MC.append(mc); D.append(done)
        with torch.no_grad():
            prey = known_prey(env, share_r)
            update_mem(env, prey)
            ob = observe(env, prey)
            VM.append(netM(ob)[1]); VC.append(netC(ob)[1])
            mm, mc = ~env.is_cat, env.is_cat.clone()
            if pool:
                mm, mc = mm & ~half_b[:, None], mc & ~half_a[:, None]
            MM.append(mm); MC.append(mc)
        st = lambda L: torch.stack(L)
        O, AM, AC, LPM, LPC = st(O), st(AM), st(AC), st(LPM), st(LPC)
        VM, VC, RM, RC, MM, MC, D = st(VM), st(VC), st(RM), st(RC), st(MM), st(MC), st(D)
        advM = gae(RM, VM, D, MM[1:], gamma, lam)
        advC = gae(RC, VC, D, MC[1:], gamma, lam)
        flat = lambda x: x.reshape(-1, *x.shape[3:])
        selM, selC = MM[:-1].reshape(-1), MC[:-1].reshape(-1)
        Of = flat(O)
        entM = entC = float("nan")
        if not freeze_mouse and selM.any():
            entM = ppo_update(netM, optM, Of[selM], AM.reshape(-1)[selM], LPM.reshape(-1)[selM], advM.reshape(-1)[selM],
                              (advM + VM[:-1]).reshape(-1)[selM])
        if not freeze_cat and selC.any():
            entC = ppo_update(netC, optC, Of[selC], AC.reshape(-1)[selC], LPC.reshape(-1)[selC], advC.reshape(-1)[selC],
                              (advC + VC[:-1]).reshape(-1)[selC])
        # 动作使用频率（看猫用不用"追"宏、老鼠用不用"藏"宏）
        fm = torch.bincount(AM.reshape(-1)[selM], minlength=N_ACT_M).float()
        fc = torch.bincount(AC.reshape(-1)[selC], minlength=N_ACT_C).float()
        rec = {"it": it, "time": time.time() - t0, "entM": entM, "entC": entC,
               "m_still": (fm[0] / fm.sum()).item(), "m_hide": (fm[25] / fm.sum()).item(), "m_flee": (fm[26] / fm.sum()).item(),
               "m_sprint": (fm[17:25].sum() / fm.sum()).item(),
               "c_chase": (fc[17] / fc.sum().clamp(min=1)).item(), "c_mem": (fc[18] / fc.sum().clamp(min=1)).item(),
               "c_patrol": (fc[19] / fc.sum().clamp(min=1)).item(), "c_dir": (fc[1:17].sum() / fc.sum().clamp(min=1)).item()}
        if ep_stats:
            rec["surv"] = sum(e[0] for e in ep_stats) / len(ep_stats)
            rec["top_catch"] = sum(e[1] for e in ep_stats) / len(ep_stats)
            rec["frac_cat_end"] = sum(e[2] for e in ep_stats) / len(ep_stats)
            if pool:  # surv_a：当前老鼠对历史猫；surv_b：历史老鼠对当前猫
                rec["surv_a"] = sum(e[3] for e in ep_stats) / len(ep_stats)
                rec["surv_b"] = sum(e[4] for e in ep_stats) / len(ep_stats)
        log.append(rec)
        if it % 10 == 0 or ep_stats:
            print(f"[{tag}] it {it} {rec['time']:.0f}s " + " ".join(f"{k}={v:.3f}" for k, v in rec.items() if k not in ("it", "time")), flush=True)
        if (it + 1) % 10 == 0 or it == iters - 1:
            Path(out).mkdir(parents=True, exist_ok=True)
            torch.save({"mouse": netM.state_dict(), "cat": netC.state_dict(), "d_in": d_in, "log": log, "reward": reward,
                        "share_r": share_r, "T_ep": T_ep, "cfg_kw": cfg_kw or {}, "seed": seed, "it": it,
                        "optM": optM.state_dict(), "optC": optC.state_dict(), "pool": pool, "pool_max": pool_max, "B": B,
                        "snaps": snaps if pool else [], "use_count": use_count if pool else {},
                        "restarts": restarts}, str(ck_path) + ".tmp")
            os.replace(str(ck_path) + ".tmp", ck_path)
    return netM, netC


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="indiv")
    ap.add_argument("--reward", default="indiv")
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--share_r", type=float, default=30.0)
    ap.add_argument("--T_ep", type=float, default=1800.0)
    ap.add_argument("--no_conceal", action="store_true", help="反事实：静止、灌木、夹缝都不再更难被发现")
    ap.add_argument("--pool", action="store_true", help="和历史版本的对手对局（虚拟自我对弈）")
    ap.add_argument("--pool_max", type=int, default=24, help="池里最多保留几份快照；1 = 只用最新一份（对照组）")
    a = ap.parse_args()
    kw = dict(f_still=1.0, f_bush=1.0, f_gap=1.0) if a.no_conceal else None
    train(a.tag, a.reward, a.iters, a.B, seed=a.seed, share_r=a.share_r, T_ep=a.T_ep, cfg_kw=kw, pool=a.pool,
          pool_max=a.pool_max)
