"""从实验结果直接生成报告表格的 LaTeX 行，避免手抄数字。输出到 ../report/parts/tab_*.tex。"""
from pathlib import Path

import numpy as np
import torch

from campus_scripted import FREEZE, HIDE, HIDE_BUSH, HIDE_GAP, RUN
from plot_probe import boot_ci

OUT = Path("../report/parts")
rows = {r["tag"]: r for r in torch.load("results/campus_probe.pt", weights_only=False) if r["tag"].startswith("mixed")}
CONDS = [("base", "基准"), ("still_same", "静止无优势"), ("terrain_same", "地形无优势"), ("conceal_same", "两者都无优势"),
         ("no_npc", "无夜跑路人"), ("many_npc", "20 个路人")]
ORDER = [FREEZE, HIDE_BUSH, HIDE, HIDE_GAP]


def gm(v, m):
    return ((v * m).sum(1) / m.sum(1).clamp(min=1)).numpy()


def n(x):
    return f"{x:.1f}".replace("-", "$-$")


lines = []
for tag, lab in CONDS:
    r = rows.get("mixed_" + tag)
    if r is None:
        continue
    s, v = r["strat"][:, 10:], r["surv"][:, 10:].float() / 60
    run = gm(v, s == RUN)
    cells = [lab, n(boot_ci(run)[0])]
    for k in ORDER:
        g = gm(v, s == k)
        dm, dl, dh = boot_ci(g - run)
        cells.append(f"{n(boot_ci(g)[0])}（{'+' if dm >= 0 else ''}{n(dm)} [{n(dl)}, {n(dh)}]）")
    lines.append("    " + " & ".join(cells) + r"\\")
(OUT / "tab_campus_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf8")
print("\n".join(lines))
