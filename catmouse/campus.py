"""上海大学宝山校区 A–G 楼区域的猫捉老鼠批量环境（PyTorch，GPU 上同时跑 B 局）。

地图来自 OpenStreetMap（map/build_map.py），0.5 m 一格：平地 / 建筑（不能进、挡视线）/ 灌木丛 / 空调外机夹缝 / 场地外。
规则（用户 2026-11-05 确认）：
  - 30 人，开局 10 只猫；碰一下就算；没有安全区；只能在平地上，不能进楼；晚上进行。
  - 有人被抓变猫，所有人马上都知道（不需要隐藏身份）。
  - 场上有夜跑的路人：老鼠在 d_id 米以外分不清他是不是猫。
发现模型（"被看见"是概率事件，而不是一进视野就暴露）：
  观察者 i 对目标 j，只要视线不被楼挡住、距离 < R、在视野角内，每秒以速率
      rate × g(距离) × f_动静(j) × f_地形(j)
  发现 j；g(d)=1（d<R/2），到 R 线性降到 0。一旦发现就一直盯住，直到视线被挡或超出 R。
  1 m 以内必定发现。
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
OPEN, BLD, BUSH, GAP, OUT = 0, 1, 2, 3, 4


@dataclass
class CCfg:
    N: int = 30
    n_cats: int = 10
    dt: float = 0.5
    T: float = 1800.0  # 一轮 30 分钟（实际游戏是上下两轮，各 30 分钟）
    head_start: float = 60.0  # 开局猫原地数 60 秒
    R: float = 20.0  # 晚上路灯下能认出人的最远距离
    fov: float = 360.0
    r_touch: float = 0.6
    v_sprint: float = 4.0
    v_jog: float = 2.5
    v_walk: float = 1.5
    v_slow: float = 0.8  # 在灌木丛 / 夹缝里只能慢慢挪
    stamina: float = 20.0
    rec_still: float = 1.0
    rec_jog: float = 0.25
    rate: float = 4.0  # 空地上走动的人，近处每秒被发现的速率（≈ 一步就暴露）
    f_still: float = 0.25  # 站着不动：发现速率乘 0.25
    f_bush: float = 0.2  # 在灌木丛里：再乘 0.2
    f_gap: float = 0.05  # 在空调外机夹缝里：再乘 0.05
    n_npc: int = 6  # 夜跑路人
    d_id: float = 5.0  # 5 m 以内才认得出路人不是猫
    spawn_r: float = 15.0  # 开局所有人在集合点 15 m 内
    cat_mem: float = 30.0
    flee_d: float = 10.0
    map_file: str = "map/campus_grid.npz"
    nav_file: str = "map/campus_nav.npz"


_CACHE = {}


def load_map(cfg, dev):
    key = (cfg.map_file, cfg.nav_file, dev)
    if key in _CACHE:
        return _CACHE[key]
    z = np.load(HERE / cfg.map_file)
    nv = np.load(HERE / cfg.nav_file)
    grid = torch.as_tensor(z["grid"].astype(np.int64), device=dev)
    res = float(z["res"])
    labels = json.loads(str(z["labels"]))
    m = dict(
        grid=grid, res=res, H=grid.shape[0], W=grid.shape[1], Wm=grid.shape[1] * res, Hm=grid.shape[0] * res,
        labels=labels,
        nodes=torch.as_tensor(nv["nodes"], device=dev),
        next_hop=torch.as_tensor(nv["next_hop"].astype(np.int32), device=dev).short(),
        cell2node=torch.as_tensor(nv["cell2node"], device=dev).long(),
        cell=float(nv["cell"]),
    )
    g = z["grid"]
    walk = np.isin(g, (OPEN, BUSH, GAP))
    m["walk"] = torch.as_tensor(walk, device=dev)
    m["block_los"] = torch.as_tensor(np.isin(g, (BLD, OUT)), device=dev)
    # 藏身候选点：灌木丛每 2 m 取一个格子，夹缝每个格子都要
    ii, jj = np.nonzero(np.isin(g, (BUSH, GAP)))
    keep = (ii % 4 == 0) & (jj % 4 == 0) | (g[ii, jj] == GAP)
    ii, jj = ii[keep], jj[keep]
    m["spots"] = torch.as_tensor(np.stack([(jj + 0.5) * res, (ii + 0.5) * res], -1), dtype=torch.float32, device=dev)
    m["spot_type"] = torch.as_tensor(g[ii, jj].astype(np.int64), device=dev)
    # 可走的空地格（出生点、随机游走目标用）
    oi, oj = np.nonzero(g == OPEN)
    m["open_pts"] = torch.as_tensor(np.stack([(oj + 0.5) * res, (oi + 0.5) * res], -1), dtype=torch.float32, device=dev)
    # 集合点：实际游戏在 C、D、CJ、DJ 四栋楼之间的空地集合，取四栋楼中心的平均位置附近最近的空地
    c = np.mean(np.array([labels[k] for k in ("C楼", "D楼", "CJ楼", "DJ楼")]), 0)
    d = ((m["open_pts"].cpu().numpy() - c) ** 2).sum(-1)
    m["rally"] = m["open_pts"][int(d.argmin())]
    _CACHE[key] = m
    return m


class Campus:
    def __init__(self, cfg: CCfg, B: int, device="cuda", seed=0):
        self.c, self.B, self.dev, self.N = cfg, B, device, cfg.N
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.m = load_map(cfg, device)
        self.M = cfg.n_npc
        self._ex = torch.tensor([1.0, 0.0], device=device)
        self._ey = torch.tensor([0.0, 1.0], device=device)
        self.reset()

    # ---------- 地图查询 ----------
    def rand(self, *s):
        return torch.rand(*s, generator=self.gen, device=self.dev)

    def randn(self, *s):
        return torch.randn(*s, generator=self.gen, device=self.dev)

    def cell_type(self, p):
        m = self.m
        i = (p[..., 1] / m["res"]).long().clamp(0, m["H"] - 1)
        j = (p[..., 0] / m["res"]).long().clamp(0, m["W"] - 1)
        return m["grid"][i, j]

    def walkable(self, p):
        m = self.m
        i = (p[..., 1] / m["res"]).long().clamp(0, m["H"] - 1)
        j = (p[..., 0] / m["res"]).long().clamp(0, m["W"] - 1)
        return m["walk"][i, j]

    def seg_clear(self, a, b, S=24, walk=False):
        """a,b (...,2)：线段上 S 个采样点都不被挡（walk=True 时要求都可走）。"""
        m = self.m
        t = torch.linspace(0, 1, S, device=self.dev)
        p = a[..., None, :] + (b - a)[..., None, :] * t[:, None]
        i = (p[..., 1] / m["res"]).long().clamp(0, m["H"] - 1)
        j = (p[..., 0] / m["res"]).long().clamp(0, m["W"] - 1)
        if walk:
            return m["walk"][i, j].all(-1)
        return ~m["block_los"][i, j].any(-1)

    def node_of(self, p):
        m = self.m
        CH, CW = m["cell2node"].shape
        ci = (p[..., 1] / m["cell"]).long().clamp(0, CH - 1)
        cj = (p[..., 0] / m["cell"]).long().clamp(0, CW - 1)
        return m["cell2node"][ci, cj]

    def nav_dir(self, pos, goal):
        """沿寻路表绕开楼走向 goal 的单位方向。"""
        m = self.m
        u, g = self.node_of(pos), self.node_of(goal)
        h = m["next_hop"][u, g].long().clamp(min=0)
        h2 = m["next_hop"][h, g].long().clamp(min=0)
        p1, p2 = m["nodes"][h], m["nodes"][h2]
        near = (goal - pos).norm(dim=-1) < 40
        direct = near & self.seg_clear(pos, goal, S=48, walk=True)
        ok2 = self.seg_clear(pos, p2, S=16, walk=True)
        tgt = torch.where(direct[..., None], goal, torch.where(ok2[..., None], p2, p1))
        same = (u == g)[..., None]
        tgt = torch.where(same & ~direct[..., None], goal, tgt)
        v = tgt - pos
        return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def rand_open(self, *shape):
        P = self.m["open_pts"]
        k = torch.randint(0, P.shape[0], shape, generator=self.gen, device=self.dev)
        return P[k] + (self.rand(*shape, 2) - 0.5) * 0.4

    def spawn_near(self, ctr, *shape):
        p = ctr + (self.rand(*shape, 2) * 2 - 1) * self.c.spawn_r
        bad = ~self.walkable(p)
        for _ in range(20):
            if not bad.any():
                break
            q = ctr + (self.rand(*shape, 2) * 2 - 1) * self.c.spawn_r
            p = torch.where(bad[..., None], q, p)
            bad = ~self.walkable(p)
        return torch.where(bad[..., None], ctr.expand_as(p), p)

    # ---------- 状态 ----------
    def reset(self):
        c, B, N, M = self.c, self.B, self.N, self.M
        self.t = 0.0
        self.pos = self.spawn_near(self.m["rally"], B, N)
        self.vel = torch.zeros(B, N, 2, device=self.dev)
        self.stam = torch.full((B, N), float(c.stamina), device=self.dev)
        ang = self.rand(B, N) * 6.2831853
        self.heading = torch.stack([ang.cos(), ang.sin()], -1)
        self.is_cat = torch.zeros(B, N, dtype=torch.bool, device=self.dev)
        self.is_cat[:, : c.n_cats] = True
        self.init_mouse = ~self.is_cat.clone()
        self.caught_t = torch.full((B, N), float("inf"), device=self.dev)
        self.catcher = torch.full((B, N), -1, dtype=torch.long, device=self.dev)
        self.catches = torch.zeros(B, N, device=self.dev)
        self.spot = torch.zeros(B, N, N, dtype=torch.bool, device=self.dev)  # i 发现并盯住了 j
        # 夜跑路人
        self.npc = self.rand_open(B, M) if M else torch.zeros(B, 0, 2, device=self.dev)
        self.npc_wp = self.rand_open(B, M) if M else self.npc.clone()
        self.npc_vel = torch.zeros_like(self.npc)
        self.npc_spot = torch.zeros(B, N, M, dtype=torch.bool, device=self.dev)
        # 脚本策略用的记忆
        self.mem_pos = self.pos.clone()
        self.mem_t = torch.full((B, N), -1e9, device=self.dev)
        self.wp = self.rand_open(B, N)
        self.G = 20.0  # 搜索记忆格 20 m
        gw, gh = math.ceil(self.m["Wm"] / self.G), math.ceil(self.m["Hm"] / self.G)
        gx = (torch.arange(gw, device=self.dev) + 0.5) * self.G
        gy = (torch.arange(gh, device=self.dev) + 0.5) * self.G
        ctr = torch.stack(torch.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
        # 只保留中心附近有空地的格
        d = torch.cdist(ctr, self.m["open_pts"][::7])
        self.cell_ctr = ctr[d.min(-1).values < self.G / 2]
        self.cell_t = torch.zeros(B, self.cell_ctr.shape[0], device=self.dev)
        self.hide_spot = self.pos.clone()
        self.need_spot = torch.ones(B, N, dtype=torch.bool, device=self.dev)
        self.last_threat = self.pos.clone()
        self.seen_threat = torch.zeros(B, N, dtype=torch.bool, device=self.dev)
        self.last_seen_t = torch.full((B, N), -1e9, device=self.dev)
        self._perceive()
        return self

    # ---------- 感知 ----------
    def _cand(self, obs_pos, obs_head, tgt_pos):
        """(B,n,m)：视线通、距离 < R、在视野角内。"""
        c = self.c
        diff = tgt_pos[:, None] - obs_pos[:, :, None]
        dist = diff.norm(dim=-1)
        inr = dist < c.R
        a = obs_pos[:, :, None].expand(-1, -1, tgt_pos.shape[1], -1)
        b = tgt_pos[:, None].expand(-1, obs_pos.shape[1], -1, -1)
        los = torch.zeros_like(inr)
        if inr.any():
            los[inr] = self.seg_clear(a[inr], b[inr], S=int(c.R) + 1)
        cand = inr & los
        if c.fov < 360:
            cosang = (diff * obs_head[:, :, None]).sum(-1) / dist.clamp(min=1e-6)
            cand &= (cosang >= math.cos(math.radians(c.fov / 2))) | (dist < 2.0)
        return cand, dist

    def detect_rate(self, tgt_pos, tgt_vel, dist):
        c = self.c
        moving = tgt_vel.norm(dim=-1) > 0.3
        ct = self.cell_type(tgt_pos)
        f = torch.where(moving, 1.0, c.f_still)
        f = f * torch.where(ct == BUSH, c.f_bush, torch.where(ct == GAP, c.f_gap, 1.0))
        g = (2 * (1 - dist / c.R)).clamp(0, 1)
        return c.rate * g * f[:, None, :]

    def _perceive(self):
        c = self.c
        cand, self.dist = self._cand(self.pos, self.heading, self.pos)
        cand &= ~torch.eye(self.N, dtype=torch.bool, device=self.dev)
        self.vis = cand  # 视线通且在 20 m 内（看队友用）
        p = 1 - torch.exp(-self.detect_rate(self.pos, self.vel, self.dist) * c.dt)
        new = cand & ((self.rand(*cand.shape) < p) | (self.dist < 1.0))
        self.spot = (self.spot & cand) | new
        if self.M:
            cn, self.npc_dist = self._cand(self.pos, self.heading, self.npc)
            pn = 1 - math.exp(-c.rate * c.dt)  # 路人一直在跑，很显眼
            g = (2 * (1 - self.npc_dist / c.R)).clamp(0, 1)
            newn = cn & (self.rand(*cn.shape) < pn * g)
            self.npc_spot = (self.npc_spot & cn) | newn
        else:
            self.npc_dist = torch.zeros(self.B, self.N, 0, device=self.dev)
        # 老鼠眼里的"威胁"：看见的猫 + 5 m 外认不出的夜跑路人
        self.threat = self.spot & self.is_cat[:, None, :]
        self.npc_threat = self.npc_spot & (self.npc_dist > c.d_id)

    # ---------- 推进 ----------
    def step(self, vel):
        c = self.c
        sp = vel.norm(dim=-1, keepdim=True)
        ct = self.cell_type(self.pos)
        vmax = torch.where(self.stam > 0, c.v_sprint, c.v_jog)
        vmax = torch.where((ct == BUSH) | (ct == GAP), torch.full_like(vmax, c.v_slow), vmax)[..., None]
        vel = torch.where(sp > vmax, vel / sp.clamp(min=1e-6) * vmax, vel)
        if self.t < c.head_start:
            vel = torch.where(self.is_cat[..., None], torch.zeros_like(vel), vel)
        old = self.pos
        p = self.pos
        sub = 4
        ex, ey = self._ex, self._ey
        dsub = torch.full((self.B, self.N, self.N), 1e9, device=self.dev)  # 子步内两两最近距离，避免迎面穿过时漏抓
        for _ in range(sub):
            d = vel * (c.dt / sub)
            full, xo, yo = p + d, p + d * ex, p + d * ey
            p = torch.where(self.walkable(full)[..., None], full,
                            torch.where(self.walkable(xo)[..., None], xo, torch.where(self.walkable(yo)[..., None], yo, p)))
            dsub = torch.minimum(dsub, (p[:, :, None] - p[:, None]).norm(dim=-1))
        self.pos = p
        self.vel = (p - old) / c.dt
        spd = self.vel.norm(dim=-1)
        self.heading = torch.where((spd > 0.3)[..., None], self.vel / spd[..., None].clamp(min=1e-6), self.heading)
        drain = ((spd - c.v_jog) / (c.v_sprint - c.v_jog)).clamp(min=0) * c.dt
        rec = torch.where(spd <= c.v_walk + 0.05, c.rec_still, torch.where(spd <= c.v_jog + 0.05, c.rec_jog, 0.0)) * c.dt
        self.stam = (self.stam - drain + rec).clamp(max=c.stamina)
        # 路人
        if self.M:
            reach = (self.npc_wp - self.npc).norm(dim=-1) < 3
            self.npc_wp = torch.where(reach[..., None], self.rand_open(self.B, self.M), self.npc_wp)
            nd = self.nav_dir(self.npc, self.npc_wp)
            q = self.npc + nd * c.v_jog * c.dt
            okq = self.walkable(q)
            new_npc = torch.where(okq[..., None], q, self.npc)
            self.npc_vel = (new_npc - self.npc) / c.dt
            self.npc = new_npc
            self.npc_wp = torch.where(okq[..., None], self.npc_wp, self.rand_open(self.B, self.M))
        # 接触 -> 转化
        self.dist = (self.pos[:, :, None] - self.pos[:, None]).norm(dim=-1)
        touch = self.is_cat[:, :, None] & ~self.is_cat[:, None, :] & (torch.minimum(self.dist, dsub) < c.r_touch)
        newly = touch.any(1)
        if self.t < c.head_start:
            newly &= False
        if newly.any():
            dmask = torch.where(touch, self.dist, torch.full_like(self.dist, 1e9))
            who = dmask.argmin(1)
            add = torch.zeros_like(self.catches)
            add.scatter_add_(1, who, newly.float())
            self.catches += add
            self.catcher = torch.where(newly, who, self.catcher)
            self.caught_t = torch.where(newly, torch.full_like(self.caught_t, self.t), self.caught_t)
            self.is_cat |= newly
        self.t += c.dt
        self._perceive()
        return newly

    @property
    def done(self):
        return self.t >= self.c.T or bool(self.is_cat.all())
