"""Nature 风格出图的统一导出：按 nature-figure 规范做版面对齐检查、导出 PDF/SVG/PNG、
再跑字号（≥5 pt）和文字碰撞两项 PDF 检查。"""
import json
import subprocess
import sys
from pathlib import Path

NS = Path("D:/desk/cum/research/external/nature-skills/skills/nature-figure/scripts")
sys.path.insert(0, str(NS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402

QA = Path("figures/qa")


def save(fig, out: Path, exclude=(), exemptions=(), strict=True):
    QA.mkdir(parents=True, exist_ok=True)
    name = out.stem
    rep = require_matplotlib_panel_alignment(
        fig, json_out=QA / f"{name}.alignment.json", tolerance_pt=1.5, gutter_tolerance_pt=1.5,
        strict=strict, exclude_axes=exclude, exemptions=list(exemptions))
    print(f"[{name}] 版面对齐: {rep.get('status', rep.get('verdict', '?'))}")
    for ext in ("pdf", "svg"):
        fig.savefig(out.with_suffix("." + ext), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    pdf = out.with_suffix(".pdf")
    r1 = subprocess.run([sys.executable, str(NS / "audit_pdf_text.py"), str(pdf), "--min-pt", "5"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(f"[{name}] 字号检查 (退出码 {r1.returncode}): " + (r1.stdout.strip().splitlines() or [""])[-1])
    r2 = subprocess.run([sys.executable, str(NS / "audit_figure_collisions.py"), str(pdf),
                         "--json-out", str(QA / f"{name}.collision.json"),
                         "--overlay-pdf", str(QA / f"{name}.collision.pdf")],
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [l for l in r2.stdout.splitlines() if l.strip()]
    print(f"[{name}] 碰撞检查 (退出码 {r2.returncode}): " + " | ".join(lines[:6]))
    return r1.returncode, r2.returncode
