"""从 OpenStreetMap 数据（shu_raw.json，上海大学宝山校区）生成 A–G 教学楼区域的栅格地图。

格子类型：
  0 平地（路、广场、草坪，能跑能走）
  1 建筑（不能进，挡视线）——规则是只能在平地上，不能进教学楼
  2 灌木丛（OSM 的 natural=wood / tree_row；能进，只能走，被发现的概率低）
  3 夹缝（空调外机和楼之间的缝。OSM 没有空调外机，按固定随机种子沿楼外墙放置；能进，只能走，被发现的概率很低）
  4 场地外（不能去）
输出 campus_grid.npz：grid (H,W) uint8、res（米/格）、labels（楼名 -> 中心坐标，米）。
"""
import json
import math
import sys

import numpy as np
from PIL import Image, ImageDraw

RES = 0.5  # 米/格
MARGIN = 25.0  # 楼群外扩多少米作为场地边界
GAP_EVERY = 14.0  # 平均每隔多少米外墙有一台空调外机
GAP_LEN, GAP_DEPTH = 2.0, 1.0  # 夹缝长 2 m、深 1 m（外机本身再占 0.5 m，挡路挡视线）

TARGET = [x + "楼" for x in "A B C D E F G AJ BJ CJ DJ EJ FJ GJ".split()]


def main(src="shu_raw.json", out="campus_grid.npz", seed=0):
    d = json.load(open(src, encoding="utf8"))
    nodes = {e["id"]: (e["lat"], e["lon"]) for e in d["elements"] if e["type"] == "node"}
    ways = [e for e in d["elements"] if e["type"] == "way"]
    tgt = [w for w in ways if w.get("tags", {}).get("name") in TARGET]
    lat0 = np.mean([nodes[n][0] for w in tgt for n in w["nodes"]])
    lon0 = np.mean([nodes[n][1] for w in tgt for n in w["nodes"]])
    ky, kx = 111320.0, 111320.0 * math.cos(math.radians(lat0))

    def xy(n):
        la, lo = nodes[n]
        return (lo - lon0) * kx, (la - lat0) * ky

    pts = np.array([xy(n) for w in tgt for n in w["nodes"]])
    x0, y0 = pts.min(0) - MARGIN
    x1, y1 = pts.max(0) + MARGIN
    Wm, Hm = x1 - x0, y1 - y0
    W, H = int(math.ceil(Wm / RES)), int(math.ceil(Hm / RES))

    def pix(n):
        x, y = xy(n)
        return ((x - x0) / RES, (y1 - y) / RES)  # 图像坐标：y 向下

    img = Image.new("L", (W, H), 0)
    dr = ImageDraw.Draw(img)
    bld = Image.new("L", (W, H), 0)
    db = ImageDraw.Draw(bld)
    for w in ways:
        t = w.get("tags", {})
        ns = [n for n in w["nodes"] if n in nodes]
        if len(ns) < 2:
            continue
        poly = [pix(n) for n in ns]
        if t.get("natural") == "wood" and len(ns) >= 3:
            dr.polygon(poly, fill=2)
        elif t.get("natural") == "tree_row":
            dr.line(poly, fill=2, width=int(round(2.0 / RES)))
    for w in ways:
        t = w.get("tags", {})
        ns = [n for n in w["nodes"] if n in nodes]
        if "building" in t and len(ns) >= 3:
            poly = [pix(n) for n in ns]
            dr.polygon(poly, fill=1)
            db.polygon(poly, fill=1)
    g = np.array(img, dtype=np.uint8)
    b = np.array(bld, dtype=np.uint8).astype(bool)

    # 空调外机 + 夹缝：沿 A–G 群楼外墙。外机（0.5 m）挡路挡视线，外机和墙之间留 1 m 深的缝。
    rng = np.random.default_rng(seed)
    n_gap = 0
    for w in tgt:
        ns = [n for n in w["nodes"] if n in nodes]
        P = np.array([xy(n) for n in ns])
        if not np.allclose(P[0], P[-1]):
            P = np.vstack([P, P[:1]])
        area2 = np.sum(P[:-1, 0] * P[1:, 1] - P[1:, 0] * P[:-1, 1])
        sgn = 1.0 if area2 > 0 else -1.0  # 逆时针时外法向是 (dy,-dx)
        for a, c in zip(P[:-1], P[1:]):
            L = np.linalg.norm(c - a)
            if L < GAP_LEN + 1:
                continue
            u = (c - a) / L
            nrm = sgn * np.array([u[1], -u[0]])
            k = rng.poisson(L / GAP_EVERY)
            for _ in range(k):
                s = rng.uniform(0.5, L - GAP_LEN - 0.5)
                for off in np.arange(0, GAP_LEN, RES / 2):
                    for dep in np.arange(0.1, GAP_DEPTH + 0.5, RES / 2):
                        p = a + u * (s + off) + nrm * dep
                        i, j = int((y1 - p[1]) / RES), int((p[0] - x0) / RES)
                        if 0 <= i < H and 0 <= j < W and not b[i, j]:
                            g[i, j] = 3 if dep <= GAP_DEPTH else 1
                n_gap += 1
    # 场地边界
    g[:2, :] = g[-2:, :] = 4
    g[:, :2] = g[:, -2:] = 4

    labels = {}
    for w in tgt:
        P = np.array([xy(n) for n in w["nodes"] if n in nodes])
        cx, cy = P.mean(0)
        labels[w["tags"]["name"]] = (float(cx - x0), float(y1 - cy))  # 场地坐标（米），y 向下与栅格一致
    np.savez_compressed(out, grid=g, res=RES, labels=json.dumps(labels, ensure_ascii=False),
                        origin=json.dumps(dict(lat0=lat0, lon0=lon0, x0=float(x0), y1=float(y1))))
    cnt = {k: int((g == k).sum() * RES * RES) for k in range(5)}
    print(f"场地 {Wm:.0f} m × {Hm:.0f} m，栅格 {W}×{H}；面积(m²) 平地 {cnt[0]} 建筑 {cnt[1]} 灌木 {cnt[2]} 夹缝 {cnt[3]}；空调外机 {n_gap} 台")


if __name__ == "__main__":
    main(*sys.argv[1:])
