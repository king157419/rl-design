"""评估 PPO 老鼠：在训练没见过的 2 张布局上，与脚本"躲"/"跑"比较存活时间，并统计行为。"""
import json

import torch

import ppo
from env import Cfg, CatMouse
from scripted import HIDE, LEARNED, RUN, cat_actions, mouse_actions

B = 512
TEST_LAYOUTS = (10, 11)


def dist_to_cover(env):
    if env.K == 0:
        return torch.full((env.B, env.N), float("nan"), device=env.dev)
    r = env.obs_rects
    p = env.pos[..., None, :]
    q = torch.stack([p[..., 0].clamp(r[:, 0], r[:, 2]), p[..., 1].clamp(r[:, 1], r[:, 3])], -1)
    return (p - q).norm(dim=-1).amin(-1)


@torch.no_grad()
def rollout(env, strategy, net=None):
    acts = ppo.action_table(env.c, env.dev)
    env.reset()
    still = walk = jog = sprint = cover = wall = n = 0.0
    while not env.done:
        ppo.update_threat_memory(env)
        a = mouse_actions(env, strategy)
        if net is not None:
            logits, _ = net(ppo.observe(env))
            a_l = acts[torch.distributions.Categorical(logits=logits).sample()]
            a = torch.where((strategy == LEARNED)[..., None], a_l, a)
        a = torch.where(env.is_cat[..., None], cat_actions(env), a)
        env.step(a)
        m = (~env.is_cat).float()
        spd = env.vel.norm(dim=-1)
        still += ((spd < 0.3).float() * m).sum().item()
        walk += (((spd >= 0.3) & (spd <= 1.6)).float() * m).sum().item()
        jog += (((spd > 1.6) & (spd <= 2.6)).float() * m).sum().item()
        sprint += ((spd > 2.6).float() * m).sum().item()
        dc = dist_to_cover(env)
        if env.K:
            cover += (((dc < 1.5).float()) * m).sum().item()
        dw = torch.minimum(torch.minimum(env.pos[..., 0], env.c.W - env.pos[..., 0]), torch.minimum(env.pos[..., 1], env.c.H - env.pos[..., 1]))
        wall += ((dw < 3).float() * m).sum().item()
        n += m.sum().item()
    surv = env.caught_t.clamp(max=env.c.T)[:, 10:]
    beh = {k: v / max(n, 1) for k, v in dict(still=still, walk=walk, jog=jog, sprint=sprint, near_cover=cover, near_wall=wall).items()}
    return surv.cpu(), beh


def main():
    out = {}
    for cond in ("real", "real_fov", "open"):
        for strat_name in ("run", "hide"):
            sv, bh = [], None
            for lay in TEST_LAYOUTS:
                env = CatMouse(Cfg(obs_seed=lay, **ppo.CONDS[cond]), B=B, seed=500 + lay)
                s = torch.full((B, 30), RUN if strat_name == "run" else HIDE, device="cuda")
                surv, bh = rollout(env, s)
                sv.append(surv)
            out[f"{cond}/{strat_name}"] = {"surv": torch.cat(sv), "beh": bh}
            print(cond, strat_name, f"{torch.cat(sv).mean():.1f}", {k: round(v, 3) for k, v in bh.items()}, flush=True)
        for seed in (0, 1, 2):
            try:
                ck = torch.load(f"results/ppo/{cond}_s{seed}.pt")
            except FileNotFoundError:
                continue
            net = ppo.Net(ck["d_in"]).cuda()
            net.load_state_dict(ck["net"])
            sv, bhs = [], []
            for lay in TEST_LAYOUTS:
                env = CatMouse(Cfg(obs_seed=lay, **ppo.CONDS[cond]), B=B, seed=700 + lay)
                s = torch.full((B, 30), LEARNED, device="cuda")
                surv, bh = rollout(env, s, net)
                sv.append(surv)
                bhs.append(bh)
            bh = {k: sum(b[k] for b in bhs) / len(bhs) for k in bhs[0]}
            out[f"{cond}/ppo_s{seed}"] = {"surv": torch.cat(sv), "beh": bh, "log": ck["log"]}
            print(cond, f"ppo s{seed}", f"{torch.cat(sv).mean():.1f}", {k: round(v, 3) for k, v in bh.items()}, flush=True)
    torch.save(out, "results/eval_ppo.pt")


if __name__ == "__main__":
    main()
