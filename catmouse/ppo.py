"""探针实验 2：老鼠共享一个 PPO 策略，面对脚本猫训练；只奖励"活着"。
看它最后学成"躲"还是"跑"。

用法：python ppo.py --cond real --seed 0 --iters 300
cond:
  real  = 有掩体(8) + 体力有限(20 s) + 静止不显眼(s=0.5)    —— 接近真实游戏
  open  = 无掩体 + 体力无限 + 动静一样显眼(s=1)             —— 物理上"躲"没有优势
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from env import Cfg, CatMouse
from scripted import HIDE, LEARNED, RUN, cat_actions, play

CONDS = {
    "real": dict(n_obs=8, stamina=20.0, s=0.5),
    "open": dict(n_obs=0, stamina=1e9, s=1.0),
    "real_fov": dict(n_obs=8, stamina=20.0, s=0.5, fov=120.0),
}
NK = 5  # 观测里最近的几个人
ANG = torch.arange(8) * (2 * math.pi / 8)
SPEEDS = None  # 动作：停 + 8 方向 × {走, 慢跑, 冲刺}
N_ACT = 1 + 8 * 3


def action_table(c, dev):
    dirs = torch.stack([torch.cos(ANG), torch.sin(ANG)], -1).to(dev)
    v = [torch.zeros(1, 2, device=dev)]
    for sp in (c.v_walk, c.v_jog, c.v_sprint):
        v.append(dirs * sp)
    return torch.cat(v)  # (25,2)


def rays(env):
    """8 个方向到墙/障碍物的距离，按 R 归一化并截断到 1。"""
    c = env.c
    dirs = torch.stack([torch.cos(ANG), torch.sin(ANG)], -1).to(env.dev)  # (8,2)
    p = env.pos[:, :, None, :]  # (B,N,1,2)
    d = dirs.clamp(min=-1, max=1)
    d = torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
    # 墙
    tw = torch.where(d > 0, (torch.tensor([c.W, c.H], device=env.dev) - p) / d, (0 - p) / d).amin(-1)
    best = tw
    if env.K:
        lo, hi = env.obs_rects[:, :2], env.obs_rects[:, 2:]
        pp, dd = p[..., None, :], d[:, None, :]  # (B,N,1,1,2), (8,1,2)
        t1, t2 = (lo - pp) / dd, (hi - pp) / dd
        tmin = torch.minimum(t1, t2).amax(-1)
        tmax = torch.maximum(t1, t2).amin(-1)
        hit = (tmax > tmin.clamp(min=0)) & (tmax > 0)
        th = torch.where(hit, tmin.clamp(min=0), torch.full_like(tmin, 1e9)).amin(-1)
        best = torch.minimum(best, th)
    return (best / c.R).clamp(max=1.0)


def observe(env):
    c = env.c
    B, N = env.B, env.N
    own = torch.cat([
        env.pos / torch.tensor([c.W, c.H], device=env.dev),
        env.vel / c.v_sprint,
        env.heading,
        (env.stam / min(c.stamina, 20.0)).clamp(max=1)[..., None],
        torch.full((B, N, 1), env.t / c.T, device=env.dev),
    ], -1)
    vis = env.V
    dm = torch.where(vis, env.dist, torch.full_like(env.dist, 1e9))
    dsort, idx = dm.sort(-1)
    idx, dsort = idx[..., :NK], dsort[..., :NK]
    valid = (dsort < 1e8).float()
    rel = (torch.gather(env.pos[:, None].expand(B, N, N, 2), 2, idx[..., None].expand(-1, -1, -1, 2)) - env.pos[:, :, None]) / c.R
    bel = torch.gather(env.belief.float(), 2, idx)
    nb = torch.cat([rel * valid[..., None], (bel * valid)[..., None], valid[..., None]], -1).reshape(B, N, -1)
    since = ((env.t - env.last_seen_t) / 30.0).clamp(max=1)[..., None]
    lrel = ((env.last_threat - env.pos) / c.R).clamp(-3, 3) * env.seen_threat[..., None]
    return torch.cat([own, rays(env), nb, since, lrel], -1)


def update_threat_memory(env):
    threat = env.V & env.belief
    dth = torch.where(threat, env.dist, torch.full_like(env.dist, 1e9))
    dmin, jn = dth.min(-1)
    seen = dmin < 1e8
    env.last_threat = torch.where(seen[..., None], torch.gather(env.pos, 1, jn[..., None].expand(-1, -1, 2)), env.last_threat)
    env.last_seen_t = torch.where(seen, torch.full_like(env.last_seen_t, env.t), env.last_seen_t)
    env.seen_threat |= seen


class Net(nn.Module):
    def __init__(self, d_in, h=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh())
        self.pi, self.v = nn.Linear(h, N_ACT), nn.Linear(h, 1)

    def forward(self, x):
        z = self.body(x)
        return self.pi(z), self.v(z).squeeze(-1)


def make_env(cond, B, seed, obs_seed=0):
    c = Cfg(obs_seed=obs_seed, **CONDS[cond])
    e = CatMouse(c, B=B, seed=seed)
    return e


def reset(env):
    env.reset()
    env.last_seen_t = torch.full((env.B, env.N), -1e9, device=env.dev)


def train(cond, seed, iters, B=256, T_roll=64, out="results/ppo"):
    torch.manual_seed(seed)
    dev = "cuda"
    layouts = [0, 1, 2, 3]  # 训练时轮换 4 张布局，避免背地图
    envs = [make_env(cond, B, seed * 10 + i, obs_seed=l) for i, l in enumerate(layouts)]
    for e in envs:
        reset(e)
    acts = action_table(envs[0].c, dev)
    d_in = observe(envs[0]).shape[-1]
    net = Net(d_in).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    gamma, lam, clip = 0.995, 0.95, 0.2
    log = []
    t0 = time.time()
    ei = 0
    for it in range(iters):
        env = envs[ei % len(envs)]
        if env.done:
            reset(env)
        O, A, LP, V, R, M, D = [], [], [], [], [], [], []
        ep_surv = []
        for _ in range(T_roll):
            if env.done:
                ep_surv.append(env.caught_t[env.init_mouse].clamp(max=env.c.T).mean().item())
                reset(env)
            update_threat_memory(env)
            obs = observe(env)
            mouse = ~env.is_cat
            with torch.no_grad():
                logits, v = net(obs)
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample()
                lp = dist.log_prob(a)
            vel = acts[a]
            vel = torch.where(env.is_cat[..., None], cat_actions(env), vel)
            env.step(vel)
            still = ~env.is_cat  # 这一步之后还活着
            r = (mouse & still).float() * 0.025
            done = mouse & ~still
            if env.done:
                done = mouse.clone()  # 时间到：按终止处理（观测里有剩余时间）
            O.append(obs); A.append(a); LP.append(lp); V.append(v); R.append(r); M.append(mouse); D.append(done)
        with torch.no_grad():
            update_threat_memory(env)
            _, v_last = net(observe(env))
        V.append(v_last)
        # GAE（每个智能体独立；只有当老鼠的时刻算样本）
        adv = torch.zeros_like(R[0])
        ADV = [None] * T_roll
        for t in reversed(range(T_roll)):
            nonterm = (~D[t]).float()
            delta = R[t] + gamma * V[t + 1] * nonterm - V[t]
            adv = delta + gamma * lam * nonterm * adv
            ADV[t] = adv
        O, A, LP, Vs, ADV, M = (torch.stack(x) for x in (O, A, LP, V[:-1], ADV, M))
        RET = ADV + Vs
        sel = M.reshape(-1)
        O, A, LP, ADV, RET = (x.reshape(-1, *x.shape[3:])[sel] if x.dim() > 3 else x.reshape(-1)[sel] for x in (O, A, LP, ADV, RET))
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        n = O.shape[0]
        for _ in range(4):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, 16384):
                b = perm[s:s + 16384]
                logits, v = net(O[b])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(A[b])
                ratio = (lp - LP[b]).exp()
                pl = -torch.min(ratio * ADV[b], ratio.clamp(1 - clip, 1 + clip) * ADV[b]).mean()
                vl = 0.5 * (v - RET[b]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pl + vl - 0.01 * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        ei += 1
        if ep_surv or it % 20 == 0:
            rec = {"it": it, "time": time.time() - t0, "surv": (sum(ep_surv) / len(ep_surv)) if ep_surv else None,
                   "ent": ent.item(), "n": n}
            log.append(rec)
            if ep_surv:
                print(f"[{cond} s{seed}] it {it} t {rec['time']:.0f}s surv {rec['surv']:.1f} ent {rec['ent']:.2f}", flush=True)
    Path(out).mkdir(parents=True, exist_ok=True)
    torch.save({"net": net.state_dict(), "d_in": d_in, "cond": cond, "seed": seed, "log": log}, f"{out}/{cond}_s{seed}.pt")
    return net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="real")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=300)
    a = ap.parse_args()
    train(a.cond, a.seed, a.iters)
