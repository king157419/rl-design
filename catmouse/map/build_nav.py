"""寻路表：把场地按 CELL 米分成粗格，算任意两格之间"下一步去哪个格"（绕开楼）。

粗格的代表点 = 格内离中心最近的可走细格；两个相邻粗格之间连边的条件是代表点之间的直线全程可走。
next_hop[u, g]：从粗格 u 去粗格 g，下一步该去的粗格（u==g 时为 g；不连通为 -1）。
"""
import math
import sys

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

CELL = 4.0


def seg_walkable(walk, res, a, b):
    L = np.linalg.norm(b - a)
    n = max(2, int(L / (res / 2)) + 1)
    t = np.linspace(0, 1, n)[:, None]
    p = a + (b - a) * t
    i, j = (p[:, 1] / res).astype(int), (p[:, 0] / res).astype(int)
    return bool(walk[i, j].all())


def main(src="campus_grid.npz", out="campus_nav.npz"):
    z = np.load(src)
    g, res = z["grid"], float(z["res"])
    walk = np.isin(g, (0, 2, 3))
    H, W = g.shape
    k = int(round(CELL / res))
    CH, CW = math.ceil(H / k), math.ceil(W / k)
    rep = np.full((CH, CW, 2), np.nan)
    for ci in range(CH):
        for cj in range(CW):
            blk = walk[ci * k:(ci + 1) * k, cj * k:(cj + 1) * k]
            if blk.mean() < 0.15:
                continue
            ii, jj = np.nonzero(blk)
            c = (blk.shape[0] - 1) / 2
            m = np.argmin((ii - c) ** 2 + (jj - c) ** 2)
            rep[ci, cj] = ((cj * k + jj[m] + 0.5) * res, (ci * k + ii[m] + 0.5) * res)  # (x,y) 米
    ok = ~np.isnan(rep[..., 0])
    idx = -np.ones((CH, CW), int)
    idx[ok] = np.arange(ok.sum())
    nodes = rep[ok]
    rows, cols, w = [], [], []
    for ci in range(CH):
        for cj in range(CW):
            if not ok[ci, cj]:
                continue
            for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1)):
                ni, nj = ci + di, cj + dj
                if 0 <= ni < CH and 0 <= nj < CW and ok[ni, nj] and seg_walkable(walk, res, rep[ci, cj], rep[ni, nj]):
                    d = float(np.linalg.norm(rep[ci, cj] - rep[ni, nj]))
                    rows += [idx[ci, cj], idx[ni, nj]]
                    cols += [idx[ni, nj], idx[ci, cj]]
                    w += [d, d]
    n = len(nodes)
    A = coo_matrix((w, (rows, cols)), shape=(n, n)).tocsr()
    nh = np.full((n, n), -1, np.int16)
    dist = np.zeros((n, n), np.float16)
    for s in range(0, n, 1000):
        src_ = np.arange(s, min(n, s + 1000))
        D, P = dijkstra(A, directed=False, indices=src_, return_predecessors=True)
        # 以目标 g 为源跑最短路：P[g, u] 就是 u 去 g 的下一跳
        P = np.where(P < 0, -1, P)
        nh[:, src_] = P.T.astype(np.int16)
        dist[:, src_] = np.where(np.isinf(D), 6e4, D).T.astype(np.float16)
        print(f"{min(n, s + 1000)}/{n}", flush=True)
    nh[np.arange(n), np.arange(n)] = np.arange(n)
    # 每个粗格（含不可走格）对应最近的节点，方便查询
    ci, cj = np.meshgrid(np.arange(CH), np.arange(CW), indexing="ij")
    ctr = np.stack([(cj + 0.5) * k * res, (ci + 0.5) * k * res], -1).reshape(-1, 2)
    from scipy.spatial import cKDTree
    _, near = cKDTree(nodes).query(ctr)
    cell2node = np.where(idx.reshape(-1) >= 0, idx.reshape(-1), near).reshape(CH, CW)
    comp = (dist[0] < 6e4).mean()
    np.savez_compressed(out, nodes=nodes.astype(np.float32), next_hop=nh, dist=dist, cell2node=cell2node.astype(np.int32),
                        cell=CELL, edges=len(w) // 2)
    print(f"粗格 {CH}×{CW}，节点 {n}，边 {len(w)//2}，与节点 0 连通的比例 {comp:.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
