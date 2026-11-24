"""脚本策略：猫（追/搜/巡逻），老鼠（RUN 跑 / HIDE 躲）。"""
import torch

RUN, HIDE, LEARNED = 0, 1, 2


def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def wall_push(env):
    c, p = env.c, env.pos
    f = torch.zeros_like(p)
    for axis, L in ((0, c.W), (1, c.H)):
        lo, hi = p[..., axis].clamp(min=0.3), (L - p[..., axis]).clamp(min=0.3)
        f[..., axis] += torch.where(lo < 6, 1 / lo**2, 0.0) - torch.where(hi < 6, 1 / hi**2, 0.0)
    return f


def flee_dir(env, threat):
    """远离所有看得见的威胁（按 1/d^2 加权），再加墙的排斥。"""
    diff = env.pos[:, :, None, :] - env.pos[:, None, :, :]  # i - j
    w = threat.float() / env.dist.clamp(min=0.5) ** 2
    f = (diff * w[..., None]).sum(2)
    f = _unit(f) + 3.0 * wall_push(env)
    return _unit(f)


def cat_actions(env):
    c = env.c
    vis_mouse = env.V & ~env.is_cat[:, None, :]
    dm = torch.where(vis_mouse, env.dist, torch.full_like(env.dist, 1e9))
    dmin, j = dm.min(-1)
    has = dmin < 1e8
    tgt = torch.gather(env.pos, 1, j[..., None].expand(-1, -1, 2))
    # 拦截：瞄准对方 τ 秒后的位置（τ = 距离 / 冲刺速度，最多 2 s），而不是对方现在的位置
    tvel = torch.gather(env.vel, 1, j[..., None].expand(-1, -1, 2))
    tau = (dmin.clamp(max=1e8) / c.v_sprint).clamp(max=2.0)[..., None]
    tgt_aim = tgt + tvel * tau
    env.mem_pos = torch.where(has[..., None], tgt, env.mem_pos)
    env.mem_t = torch.where(has, torch.full_like(env.mem_t, env.t), env.mem_t)
    searching = ~has & (env.t - env.mem_t < c.cat_mem) & ((env.mem_pos - env.pos).norm(dim=-1) > 1.0)
    # 搜索记忆：猫附近 6 m 内的格子记为"刚看过"
    dc = (env.pos[:, :, None, :] - env.cell_ctr).norm(dim=-1)  # (B,N,G*G)
    seen_cell = ((dc < 6.0) & env.is_cat[..., None]).any(1)
    env.cell_t = torch.where(seen_cell, torch.full_like(env.cell_t, env.t), env.cell_t)
    # 巡逻点：随机抽 4 个格子，去最久没看过的那个（含墙边和角落）
    reached = (env.wp - env.pos).norm(dim=-1) < 2.0
    cand = torch.randint(0, env.G * env.G, (env.B, env.N, 4), device=env.dev, generator=env.gen)
    ct = torch.gather(env.cell_t[:, None, :].expand(-1, env.N, -1), 2, cand)
    pick = torch.gather(cand, 2, ct.argmin(-1, keepdim=True)).squeeze(-1)
    new_wp = env.cell_ctr[pick] + (env.rand(env.B, env.N, 2) - 0.5) * torch.tensor([c.W, c.H], device=env.dev) / env.G
    new_wp = env.push_out(new_wp)
    env.wp = torch.where(reached[..., None], new_wp, env.wp)
    goal = torch.where(has[..., None], tgt_aim, torch.where(searching[..., None], env.mem_pos, env.wp))
    spd = torch.where(has, c.v_sprint, c.v_jog)[..., None]
    d = _unit(goal - env.pos + 0.15 * env.randn(env.B, env.N, 2))
    return d * spd


def nearest_cover(env):
    """选最近的障碍物，躲到它"背对最后见到的猫"的那一面，离边 0.8 m。
    从没见过猫就躲在离自己最近的那一面。没有障碍物就原地不动。"""
    if env.K == 0:
        return env.pos.clone()
    r = env.obs_rects
    p = env.pos[..., None, :]
    q = torch.stack([p[..., 0].clamp(r[:, 0], r[:, 2]), p[..., 1].clamp(r[:, 1], r[:, 3])], -1)
    k = (p - q).norm(dim=-1).argmin(-1)
    rk = r[k]  # (B,N,4)
    ctr, half = (rk[..., :2] + rk[..., 2:]) / 2, (rk[..., 2:] - rk[..., :2]) / 2
    qk = torch.gather(q, 2, k[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
    near_side = qk + 0.8 * _unit(env.pos - qk)
    u = _unit(ctr - env.last_threat + 1e-3)
    t = torch.minimum(half[..., 0] / u[..., 0].abs().clamp(min=1e-6), half[..., 1] / u[..., 1].abs().clamp(min=1e-6))
    far_side = ctr + u * (t[..., None] + 0.8)
    return torch.where(env.seen_threat[..., None], far_side, near_side)


def mouse_actions(env, strategy):
    """strategy: (B,N) 取 RUN 或 HIDE。返回 (B,N,2)。"""
    c = env.c
    threat = env.V & env.belief
    near_threat = threat & (env.dist < c.flee_d)
    any_threat, any_near = threat.any(-1), near_threat.any(-1)
    fd = flee_dir(env, threat)
    dth = torch.where(threat, env.dist, torch.full_like(env.dist, 1e9))
    dmin, jn = dth.min(-1)
    seen = dmin < 1e8
    env.last_threat = torch.where(seen[..., None], torch.gather(env.pos, 1, jn[..., None].expand(-1, -1, 2)), env.last_threat)
    env.seen_threat |= seen
    # 跑：慢跑游走，见猫就全速逃
    reached = (env.wp - env.pos).norm(dim=-1) < 2.0
    env.wp = torch.where(reached[..., None], env.rand_points(env.B, env.N), env.wp)
    run_v = torch.where(any_threat[..., None], fd * c.v_sprint, _unit(env.wp - env.pos) * c.v_jog)
    # 躲：去最近掩体贴着不动；猫进 flee_d 才逃，逃完换掩体
    env.need_spot |= any_near
    calm = ~any_near
    pick = env.need_spot & calm
    env.hide_spot = torch.where(pick[..., None], nearest_cover(env), env.hide_spot)
    env.need_spot &= ~pick
    to_spot = env.hide_spot - env.pos
    dist_spot = to_spot.norm(dim=-1, keepdim=True)
    hide_v = torch.where(dist_spot > 0.3, _unit(to_spot) * c.v_walk, torch.zeros_like(to_spot))
    hide_v = torch.where(any_near[..., None], fd * c.v_sprint, hide_v)
    return torch.where((strategy == HIDE)[..., None], hide_v, run_v)


@torch.no_grad()
def play(env, strategy, learned_fn=None):
    """跑完一局（B 局并行）。返回存活时间 (B,N)（没被抓记为 T）。"""
    env.reset()
    while not env.done:
        a = mouse_actions(env, strategy)
        if learned_fn is not None:
            a = torch.where((strategy == LEARNED)[..., None], learned_fn(env), a)
        a = torch.where(env.is_cat[..., None], cat_actions(env), a)
        env.step(a)
    return env.caught_t.clamp(max=env.c.T)
