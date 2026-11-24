"""猫捉老鼠批量环境（PyTorch，GPU 上同时跑 B 局）。

规则（来自真实线下游戏）：N 人，开局 n_cats 只猫，其余是老鼠；猫碰到老鼠（距离 < r_touch）老鼠就变猫；
没有安全区；开局猫原地不动 head_start 秒。
视野：半径 R，被矩形障碍物挡住看不见；静止的人只在 s*R 以内可被发现（s=1 表示动静一样显眼）。
信息：hidden_info=True 时，老鼠只有亲眼看到某人被抓，才知道他已经变成猫。
"""
from dataclasses import dataclass, field

import torch


@dataclass
class Cfg:
    W: float = 50.0
    H: float = 50.0
    N: int = 30
    n_cats: int = 10
    dt: float = 0.25
    T: float = 300.0
    head_start: float = 10.0
    R: float = 15.0
    s: float = 1.0
    r_touch: float = 0.6
    v_sprint: float = 4.0
    v_jog: float = 2.5
    v_walk: float = 1.5
    hidden_info: bool = False
    n_obs: int = 6
    obs_seed: int = 0
    cat_mem: float = 10.0
    flee_d: float = 8.0
    fov: float = 360.0  # 视野角（度）；360 表示全方位。2 m 以内永远能察觉
    stamina: float = 1e9  # 能全速冲刺的秒数；1e9 表示体力无限
    rec_still: float = 1.0  # 走路/站着时每秒恢复的冲刺秒数
    rec_jog: float = 0.25
    obstacles: list = field(default=None)  # 可直接给 [(x0,y0,x1,y1),...]，优先于随机生成


def make_obstacles(cfg: Cfg):
    if cfg.obstacles is not None:
        return torch.tensor(cfg.obstacles, dtype=torch.float32).reshape(-1, 4)
    g = torch.Generator().manual_seed(cfg.obs_seed)
    rects = []
    tries = 0
    while len(rects) < cfg.n_obs and tries < 5000:
        tries += 1
        w, h = (2 + 4 * torch.rand(2, generator=g)).tolist()
        x0 = 3 + (cfg.W - 6 - w) * torch.rand(1, generator=g).item()
        y0 = 3 + (cfg.H - 6 - h) * torch.rand(1, generator=g).item()
        r = (x0, y0, x0 + w, y0 + h)
        if all(r[0] > o[2] + 2 or r[2] < o[0] - 2 or r[1] > o[3] + 2 or r[3] < o[1] - 2 for o in rects):
            rects.append(r)
    return torch.tensor(rects, dtype=torch.float32).reshape(-1, 4)


class CatMouse:
    def __init__(self, cfg: Cfg, B: int, device="cuda", seed=0):
        self.c, self.B, self.dev = cfg, B, device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.obs_rects = make_obstacles(cfg).to(device)
        self.K = self.obs_rects.shape[0]
        self.reset()

    # ---------- 工具 ----------
    def rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.dev)

    def randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.dev)

    def rand_points(self, *shape):
        c = self.c
        p = self.rand(*shape, 2) * torch.tensor([c.W - 2, c.H - 2], device=self.dev) + 1
        return self.push_out(p)

    def inside(self, p, margin=0.0):
        """p (...,2) -> (...,K) 是否在障碍物里。"""
        if self.K == 0:
            return torch.zeros(*p.shape[:-1], 0, dtype=torch.bool, device=self.dev)
        r = self.obs_rects
        x, y = p[..., None, 0], p[..., None, 1]
        return (x > r[:, 0] - margin) & (x < r[:, 2] + margin) & (y > r[:, 1] - margin) & (y < r[:, 3] + margin)

    def _rect_of(self, p):
        """每个点所在的障碍物（障碍物互不重叠，最多一个）。返回 (inside 布尔, 该矩形 (...,4))。"""
        ins = self.inside(p)
        k = ins.float().argmax(-1)
        return ins.any(-1), self.obs_rects[k]

    def push_out(self, p, margin=0.05):
        if self.K == 0:
            return p
        m, r = self._rect_of(p)
        x, y = p[..., 0], p[..., 1]
        d = torch.stack([x - r[..., 0], r[..., 2] - x, y - r[..., 1], r[..., 3] - y], -1)
        face = d.argmin(-1)
        nx = torch.where(face == 0, r[..., 0] - margin, torch.where(face == 1, r[..., 2] + margin, x))
        ny = torch.where(face == 2, r[..., 1] - margin, torch.where(face == 3, r[..., 3] + margin, y))
        return torch.where(m[..., None], torch.stack([nx, ny], -1), p)

    def los(self, pos):
        """(B,N,N) 视线是否通畅（slab 法判线段与矩形相交）。"""
        B, N = pos.shape[:2]
        if self.K == 0:
            return torch.ones(B, N, N, dtype=torch.bool, device=self.dev)
        p = pos[:, :, None, None, :]
        d = (pos[:, None, :, None, :] - p)
        d = torch.where(d.abs() < 1e-6, torch.full_like(d, 1e-6), d)
        lo, hi = self.obs_rects[:, :2], self.obs_rects[:, 2:]
        t1, t2 = (lo - p) / d, (hi - p) / d
        tmin = torch.minimum(t1, t2).amax(-1)
        tmax = torch.maximum(t1, t2).amin(-1)
        hit = (tmax > tmin.clamp(min=0) + 1e-4) & (tmin < 1 - 1e-4)
        return ~hit.any(-1)

    # ---------- 状态 ----------
    def reset(self):
        c, B, N = self.c, self.B, self.N if hasattr(self, "N") else self.c.N
        self.N = N
        self.t = 0.0
        self.pos = self.rand_points(B, N)
        self.vel = torch.zeros(B, N, 2, device=self.dev)
        self.stam = torch.full((B, N), float(c.stamina), device=self.dev)
        ang = self.rand(B, N) * 6.2831853
        self.heading = torch.stack([ang.cos(), ang.sin()], -1)
        self.is_cat = torch.zeros(B, N, dtype=torch.bool, device=self.dev)
        self.is_cat[:, : c.n_cats] = True
        self.init_mouse = ~self.is_cat.clone()
        self.caught_t = torch.full((B, N), float("inf"), device=self.dev)
        self.catches = torch.zeros(B, N, device=self.dev)
        self.belief = self.is_cat[:, None, :].expand(B, N, N).clone()
        self.mem_pos = self.pos.clone()
        self.mem_t = torch.full((B, N), -1e9, device=self.dev)
        self.wp = self.rand_points(B, N)
        # 猫方共享的"搜索记忆"：把场地分成 G×G 格，记录每格最后一次被猫看过的时间
        self.G = 10
        self.cell_t = torch.zeros(B, self.G * self.G, device=self.dev)
        gx = (torch.arange(self.G, device=self.dev) + 0.5) * (c.W / self.G)
        gy = (torch.arange(self.G, device=self.dev) + 0.5) * (c.H / self.G)
        self.cell_ctr = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
        self.hide_spot = self.pos.clone()
        self.last_threat = self.pos.clone()
        self.seen_threat = torch.zeros(B, N, dtype=torch.bool, device=self.dev)
        self.last_seen_t = torch.full((B, N), -1e9, device=self.dev)
        self.need_spot = torch.ones(B, N, dtype=torch.bool, device=self.dev)
        self.dist = (self.pos[:, :, None] - self.pos[:, None]).norm(dim=-1)
        self.V = self.visibility()
        return self

    def visibility(self):
        c = self.c
        moving = self.vel.norm(dim=-1) > 0.3
        rng = torch.where(moving, torch.full_like(self.caught_t, c.R), torch.full_like(self.caught_t, c.R * c.s))
        eye = torch.eye(self.N, dtype=torch.bool, device=self.dev)
        vis = (self.dist < rng[:, None, :]) & self.los(self.pos) & ~eye
        if c.fov < 360:
            d = self.pos[:, None, :, :] - self.pos[:, :, None, :]  # j - i
            cosang = (d * self.heading[:, :, None, :]).sum(-1) / self.dist.clamp(min=1e-6)
            import math
            in_cone = cosang >= math.cos(math.radians(c.fov / 2))
            vis &= in_cone | (self.dist < 2.0)
        return vis

    def threat_mask(self):
        """(B,N,N) 观察者 i 认为 j 是猫且看得见 j。"""
        return self.V & self.belief

    # ---------- 推进 ----------
    def step(self, vel):
        c = self.c
        sp = vel.norm(dim=-1, keepdim=True)
        vmax = torch.where(self.stam > 0, c.v_sprint, c.v_jog)[..., None]
        vel = torch.where(sp > vmax, vel / sp.clamp(min=1e-6) * vmax, vel)
        if self.t < c.head_start:
            vel = torch.where(self.is_cat[..., None], torch.zeros_like(vel), vel)
        new = self.pos + vel * c.dt
        new = torch.stack([new[..., 0].clamp(0.3, c.W - 0.3), new[..., 1].clamp(0.3, c.H - 0.3)], -1)
        # 撞墙/撞障碍：去掉法向分量，沿边滑向更近的角
        if self.K:
            m, r = self._rect_of(new)
            if m.any():
                x, y = self.pos[..., 0], self.pos[..., 1]
                d = torch.stack([x - r[..., 0], r[..., 2] - x, y - r[..., 1], r[..., 3] - y], -1)
                face = (-d).argmax(-1)  # 上一步在矩形外最远的方向 = 撞进来的面
                nrm = torch.stack([torch.where(face == 0, -1.0, torch.where(face == 1, 1.0, 0.0)),
                                   torch.where(face == 2, -1.0, torch.where(face == 3, 1.0, 0.0))], -1)
                vn = (vel * nrm).sum(-1, keepdim=True)
                vt = vel - vn * nrm
                tang = torch.stack([-nrm[..., 1], nrm[..., 0]], -1)
                ctr = (r[..., :2] + r[..., 2:]) / 2
                sgn = torch.sign(((self.pos - ctr) * tang).sum(-1, keepdim=True) + 1e-3)
                vnorm = vel.norm(dim=-1, keepdim=True)
                vt = torch.where(vt.norm(dim=-1, keepdim=True) < 0.3 * vnorm, tang * sgn * vnorm, vt)
                new = torch.where(m[..., None], self.pos + vt * c.dt, new)
            new = self.push_out(new)
        self.vel = (new - self.pos) / c.dt
        self.pos = new
        spd_ = self.vel.norm(dim=-1, keepdim=True)
        self.heading = torch.where(spd_ > 0.3, self.vel / spd_.clamp(min=1e-6), self.heading)
        spd = self.vel.norm(dim=-1)
        drain = ((spd - c.v_jog) / (c.v_sprint - c.v_jog)).clamp(min=0) * c.dt
        rec = torch.where(spd <= c.v_walk + 0.05, c.rec_still, torch.where(spd <= c.v_jog + 0.05, c.rec_jog, 0.0)) * c.dt
        self.stam = (self.stam - drain + rec).clamp(max=c.stamina)
        self.dist = (self.pos[:, :, None] - self.pos[:, None]).norm(dim=-1)
        V_before = self.visibility()
        # 接触 -> 转化
        cat_mouse = self.is_cat[:, :, None] & ~self.is_cat[:, None, :]
        touch = cat_mouse & (self.dist < c.r_touch)
        newly = touch.any(1)
        if self.t < c.head_start:
            newly &= False
        if newly.any():
            dmask = torch.where(touch, self.dist, torch.full_like(self.dist, 1e9))
            catcher = dmask.argmin(1)  # (B,N) 对每只被抓老鼠，最近的那只猫
            add = torch.zeros_like(self.catches)
            add.scatter_add_(1, catcher, newly.float())
            self.catches += add
            self.caught_t = torch.where(newly, torch.full_like(self.caught_t, self.t), self.caught_t)
            self.is_cat |= newly
            if c.hidden_info:
                seen = V_before & newly[:, None, :]
                self.belief |= seen
            # 被抓的人自己当然知道
            self.belief |= torch.diag_embed(newly)
        if not c.hidden_info:
            self.belief = self.is_cat[:, None, :].expand_as(self.belief).clone()
        # 猫永远知道真相
        self.belief = torch.where(self.is_cat[:, :, None], self.is_cat[:, None, :].expand_as(self.belief), self.belief)
        self.V = self.visibility()
        self.t += c.dt
        return newly

    @property
    def done(self):
        return self.t >= self.c.T or bool(self.is_cat.all())
