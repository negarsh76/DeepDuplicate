"""CSV and self-contained HTML reports."""
from __future__ import annotations

import base64
import csv
import html
import io
from pathlib import Path
from typing import List, Sequence

from .io import load_rgb
from .pipeline import Match

BADGE = {"exact": "#e5534b", "near": "#d29922", "visual": "#539bf5", "edited": "#986ee2"}


def write_csv(matches: Sequence[Match], path: str | Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["query", "match", "kind", "cosine", "phash_dist", "ssim", "inliers", "evidence"])
        w.writeheader()
        for m in matches:
            d = m.to_dict(); d["evidence"] = "; ".join(d["evidence"]); w.writerow(d)


def _thumb(path: str, edge: int = 220) -> str:
    im = load_rgb(path, edge)
    if im is None:
        return ""
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def write_html(matches: Sequence[Match], path: str | Path, title: str = "DeepDuplicate report") -> None:
    rows: List[str] = []
    for m in matches:
        ev = html.escape(" · ".join(m.evidence))
        rows.append(f"""<div class="card"><div class="pair">
  <figure><img src="{_thumb(m.query)}"><figcaption>{html.escape(Path(m.query).name)}</figcaption></figure>
  <figure><img src="{_thumb(m.match)}"><figcaption>{html.escape(Path(m.match).name)}</figcaption></figure></div>
  <div class="meta"><span class="badge" style="background:{BADGE.get(m.kind, '#777')}">{m.kind}</span>
  cos {m.cosine:.3f} — {ev}</div></div>""")
    counts = {k: sum(m.kind == k for m in matches) for k in BADGE}
    summary = " · ".join(f'<span class="badge" style="background:{BADGE[k]}">{k} {v}</span>' for k, v in counts.items())
    Path(path).write_text(f"""<!doctype html><html><head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>body{{font-family:system-ui,sans-serif;background:#0d1117;color:#e6edf3;margin:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:16px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:12px}}
.pair{{display:flex;gap:10px}} figure{{margin:0;flex:1;text-align:center}}
img{{max-width:100%;max-height:220px;border-radius:6px}} figcaption{{font-size:12px;color:#8b949e;word-break:break-all}}
.meta{{font-size:13px;margin-top:8px;color:#adbac7}} .badge{{color:#fff;padding:2px 8px;border-radius:999px;font-size:12px;margin-right:6px}}
</style></head><body><h1>{html.escape(title)}</h1><p>{len(matches)} duplicate pairs &nbsp; {summary}</p>
<div class="grid">{''.join(rows)}</div></body></html>""", encoding="utf-8")
