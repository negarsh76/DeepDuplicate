"""Command-line interface: ``deepdup``."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embed import BACKBONES
from .io import move_unique
from .pipeline import Config, DuplicateFinder, group_matches
from .report import write_csv, write_html


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="deepdup", description="Find exact, near and visual duplicate images.")
    ap.add_argument("folder", help="folder to scan (or the NEW images when --against is given)")
    ap.add_argument("--against", metavar="REF", help="reference folder: report images in FOLDER already present in REF")
    ap.add_argument("--backbone", choices=BACKBONES, default="auto")
    ap.add_argument("--cos-accept", type=float, default=Config.cos_accept)
    ap.add_argument("--phash", type=int, default=Config.phash_accept, help="max pHash Hamming distance")
    ap.add_argument("--no-geometric", action="store_true", help="skip keypoint verification (faster)")
    ap.add_argument("--out", default="deepdup_report", help="output prefix for .csv / .html / .json")
    ap.add_argument("--move-to", metavar="DIR", help="move duplicates here (never deletes); keeps the larger file")
    ap.add_argument("--dry-run", action="store_true", help="with --move-to: only print what would move")
    a = ap.parse_args(argv)

    cfg = Config(backbone=a.backbone, cos_accept=a.cos_accept, phash_accept=a.phash, geometric=not a.no_geometric)
    finder = DuplicateFinder(cfg)
    matches = finder.find_against(a.folder, a.against) if a.against else finder.find_within(a.folder)

    write_csv(matches, f"{a.out}.csv"); write_html(matches, f"{a.out}.html")
    groups = group_matches(matches) if not a.against else [[m.query, m.match] for m in matches]
    Path(f"{a.out}.json").write_text(json.dumps({"matches": [m.to_dict() for m in matches], "groups": groups}, indent=2))
    kinds = {k: sum(m.kind == k for m in matches) for k in ("exact", "near", "visual", "edited")}
    print(f"✔ {len(matches)} duplicate pairs {kinds}  →  {a.out}.html / .csv / .json")

    if a.move_to:
        dest = Path(a.move_to)
        if a.against:
            victims = {Path(m.query) for m in matches}           # new copies of reference images
        else:                                                    # keep the largest file in each group
            victims = {Path(p) for g in groups for p in sorted(g, key=lambda p: -Path(p).stat().st_size)[1:]}
        for v in sorted(victims):
            print(("[dry-run] would move " if a.dry_run else "moving ") + str(v))
            if not a.dry_run:
                move_unique(v, dest)


if __name__ == "__main__":
    main()
