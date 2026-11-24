"""校园地图上的脚本策略。
猫：盯住的老鼠 -> 预判拦截；跟丢 -> 去最后看见的位置找 cat_mem 秒；否则按共享搜索记忆巡逻（去最久没人看过的格子）。
老鼠：
  RUN   慢跑游走，见到威胁（猫或认不出的路人）就冲刺逃；
  HIDE  走到最近的灌木丛/夹缝藏好不动，猫进 flee_d 才逃，逃完换个藏身点；
  HIDE_BUSH / HIDE_GAP  只去灌木丛 / 只去夹缝；
  FREEZE  原地站着不动（不找掩体），猫进 flee_d 才逃。
  HIDE_RANDOM  开局走到全图随机分配的一个藏身点（用来回答"哪里最好躲"），之后同 HIDE。
"""
import math

import torch

from campus import BUSH, GAP

RUN, HIDE, HIDE_BUSH, HIDE_GAP, FREEZE, HIDE_RANDOM, LEARNED = 0, 1, 2, 3, 4, 5, 9
_DIRS = None


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def flee_vel(env, threat, npc_threat, speed):
    """在 16 个方向里挑：远离所有威胁（1/d 加权），且前方 1.5 m、3 m 都能走。"""
    global _DIRS
    if _DIRS is None or _DIRS.device != torch.device(env.dev):
        a = torch.arange(16, device=env.dev) * (2 * math.pi / 16)
        _DIRS = torch.stack([a.cos(), a.sin()], -1)
    away = ((env.pos[:, :, None] - env.pos[:, None]) * (threat.float() / env.dist.clamp(min=0.5) ** 2)[..., None]).sum(2)
    if env.M:
        away += ((env.pos[:, :, None] - env.npc[:, None]) * (npc_threat.float() / env.npc_dist.clamp(min=0.5) ** 2)[..., None]).sum(2)
    away = _unit(away)
    score = (_DIRS * away[:, :, None]).sum(-1)  # (B,N,16)
    p = env.pos[:, :, None]
    ok = env.walkable(p + _DIRS * 1.5) & env.walkable(p + _DIRS * 3.0)
    score = torch.where(ok, score, score - 3)
    k = score.argmax(-1)
    return _DIRS[k] * speed


def cat_actions(env):
    c = env.c
    prey = env.spot & ~env.is_cat[:, None, :]
    dm = torch.where(prey, env.dist, torch.full_like(env.dist, 1e9))
    dmin, j = dm.min(-1)
    has = dmin < 1e8
    tgt = torch.gather(env.pos, 1, j[..., None].expand(-1, -1, 2))
    tvel = torch.gather(env.vel, 1, j[..., None].expand(-1, -1, 2))
    tau = (dmin.clamp(max=1e8) / c.v_sprint).clamp(max=2.0)[..., None]
    aim = tgt + tvel * tau
    aim = torch.where(env.walkable(aim)[..., None], aim, tgt)
    env.mem_pos = torch.where(has[..., None], tgt, env.mem_pos)
    env.mem_t = torch.where(has, torch.full_like(env.mem_t, env.t), env.mem_t)
    searching = ~has & (env.t - env.mem_t < c.cat_mem) & ((env.mem_pos - env.pos).norm(dim=-1) > 1.5)
    # 共享搜索记忆
    dc = torch.cdist(env.pos, env.cell_ctr[None].expand(env.B, -1, -1))
    seen_cell = ((dc < env.G * 0.6) & env.is_cat[..., None]).any(1)
    env.cell_t = torch.where(seen_cell, torch.full_like(env.cell_t, env.t), env.cell_t)
    reached = (env.wp - env.pos).norm(dim=-1) < 3.0
    nC = env.cell_ctr.shape[0]
    cand = torch.randint(0, nC, (env.B, env.N, 4), device=env.dev, generator=env.gen)
    ct = torch.gather(env.cell_t[:, None, :].expand(-1, env.N, -1), 2, cand)
    pick = torch.gather(cand, 2, ct.argmin(-1, keepdim=True)).squeeze(-1)
    new_wp = env.cell_ctr[pick] + (env.rand(env.B, env.N, 2) - 0.5) * env.G
    new_wp = torch.where(env.walkable(new_wp)[..., None], new_wp, env.cell_ctr[pick])
    stuck = (env.vel.norm(dim=-1) < 0.2) & (env.t > c.head_start) & ~has
    # 只改猫自己的巡逻目标；老鼠的游走目标由 mouse_actions 管
    env.wp = torch.where(((reached | stuck) & env.is_cat)[..., None], new_wp, env.wp)
    goal = torch.where(has[..., None], aim, torch.where(searching[..., None], env.mem_pos, env.wp))
    spd = torch.where(has, c.v_sprint, c.v_jog)[..., None]
    d = env.nav_dir(env.pos, goal)
    d = _unit(d + 0.1 * env.randn(env.B, env.N, 2))
    return d * spd


def pick_spot(env, kind_mask):
    """每个人选藏身点：离自己近、离最后见到的猫远。kind_mask (B,N,3)：可选 灌木/夹缝。返回 (B,N,2)。"""
    S, ty = env.m["spots"], env.m["spot_type"]
    d_me = torch.cdist(env.pos, S[None].expand(env.B, -1, -1))  # (B,N,K)
    d_th = torch.cdist(env.last_threat, S[None].expand(env.B, -1, -1))
    score = d_me - 0.5 * torch.where(env.seen_threat[..., None], d_th.clamp(max=60), torch.zeros_like(d_th))
    allow = torch.where(ty == BUSH, kind_mask[..., 0:1], kind_mask[..., 1:2])
    score = torch.where(allow, score, torch.full_like(score, 1e9))
    score = score + env.rand(*score.shape) * 4.0  # 不要所有人挤同一个点
    k = score.argmin(-1)
    return S[k]


def mouse_actions(env, strategy):
    c = env.c
    threat, nthreat = env.threat, env.npc_threat
    any_threat = threat.any(-1) | (nthreat.any(-1) if env.M else False)
    near = (threat & (env.dist < c.flee_d)).any(-1)
    if env.M:
        near |= (nthreat & (env.npc_dist < c.flee_d)).any(-1)
    dth = torch.where(threat, env.dist, torch.full_like(env.dist, 1e9))
    dmin, jn = dth.min(-1)
    seen = dmin < 1e8
    env.last_threat = torch.where(seen[..., None], torch.gather(env.pos, 1, jn[..., None].expand(-1, -1, 2)), env.last_threat)
    env.seen_threat |= seen
    flee = flee_vel(env, threat, nthreat, c.v_sprint)
    # RUN
    # 只更新老鼠自己的游走目标，不改猫的巡逻目标（猫的目标由 cat_actions 管）
    reached = ((env.wp - env.pos).norm(dim=-1) < 3.0) & ~env.is_cat
    env.wp = torch.where(reached[..., None], env.rand_open(env.B, env.N), env.wp)
    run_v = torch.where(any_threat[..., None], flee, env.nav_dir(env.pos, env.wp) * c.v_jog)
    # HIDE 系列
    hid_any = (strategy == HIDE) | (strategy == HIDE_RANDOM)
    kind = torch.stack([hid_any | (strategy == HIDE_BUSH), hid_any | (strategy == HIDE_GAP)], -1)
    # 只改老鼠自己的藏身状态；猫的 need_spot/hide_spot 属于学习猫的观测，不能被脚本老鼠的逻辑改写
    mo = ~env.is_cat
    env.need_spot |= near & mo
    pick = env.need_spot & ~near & mo
    if pick.any():
        env.hide_spot = torch.where(pick[..., None], pick_spot(env, kind), env.hide_spot)
    env.need_spot &= ~pick
    to = env.hide_spot - env.pos
    far = to.norm(dim=-1, keepdim=True) > 0.4
    hide_v = torch.where(far, env.nav_dir(env.pos, env.hide_spot) * c.v_walk, torch.zeros_like(to))
    hide_v = torch.where(near[..., None], flee, hide_v)
    freeze_v = torch.where(near[..., None], flee, torch.zeros_like(flee))
    out = torch.where((strategy == RUN)[..., None], run_v, hide_v)
    return torch.where((strategy == FREEZE)[..., None], freeze_v, out)


@torch.no_grad()
def play(env, strategy, learned_fn=None, cat_fn=None, record=None):
    env.reset()
    hr = strategy == HIDE_RANDOM
    if hr.any():
        S = env.m["spots"]
        k = torch.randint(0, S.shape[0], hr.shape, device=env.dev, generator=env.gen)
        env.hide_spot = torch.where(hr[..., None], S[k], env.hide_spot)
        env.need_spot &= ~hr
        env.first_spot = torch.where(hr[..., None], S[k], torch.full_like(S[k], float("nan")))
    while not env.done:
        a = mouse_actions(env, strategy)
        if learned_fn is not None:
            a = torch.where((strategy == LEARNED)[..., None], learned_fn(env), a)
        ca = cat_actions(env) if cat_fn is None else cat_fn(env)
        a = torch.where(env.is_cat[..., None], ca, a)
        env.step(a)
        if record is not None:
            record(env)
    return env.caught_t.clamp(max=env.c.T)
