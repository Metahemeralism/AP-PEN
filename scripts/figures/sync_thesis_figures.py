"""
Copy the figures main(3).tex asks for into the layout it asks for.

WHY THIS EXISTS
---------------
The figure scripts write into provenance-named directories (figures/pinn/,
figures/real_pinn/, figures/monte_carlo_sim/, ...), which is the layout
CLAUDE.md documents and which keeps "which script produced this" obvious. The
thesis source instead references two flat directories, figures/results/ and
figures/methodology/, grouped by where a figure appears in the document. Both
conventions are reasonable and neither should be forced on the other, so this
script bridges them by COPYING rather than moving: the provenance layout stays
authoritative and regenerable, and the thesis layout is a derived view.

Without this, every \includegraphics in main(3).tex resolves to a missing file.

Run:  python scripts/figures/sync_thesis_figures.py
      python scripts/figures/sync_thesis_figures.py --check   (report only)

The mapping is derived by scanning main(3).tex for \includegraphics paths and
locating each basename under figures/, so adding a figure to the thesis needs
no edit here -- only that the file exists somewhere under figures/.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TEX = REPO / "main(3).tex"
FIGROOT = REPO / "figures"

# Directories that are derived views, not sources. Never searched for
# originals, so a stale copy can never masquerade as the real thing.
DERIVED = {"results", "methodology"}


def wanted_paths(tex: Path) -> list[str]:
    src = re.sub(r"(?<!\\)%.*", "", tex.read_text())
    return sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", src)))


def find_source(basename: str) -> Path | None:
    """Locate a figure by basename anywhere under figures/, excluding the
    derived directories. Prefers PDF siblings implicitly since the thesis
    always asks for .pdf."""
    for cand in FIGROOT.rglob(basename):
        if not set(cand.relative_to(FIGROOT).parts) & DERIVED:
            return cand
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report what would be copied, change nothing")
    args = ap.parse_args()

    if not TEX.exists():
        print(f"error: {TEX} not found", file=sys.stderr)
        return 2

    copied = missing = uptodate = 0
    for rel in wanted_paths(TEX):
        dest = REPO / rel
        base = Path(rel).name
        src = find_source(base)
        if src is None:
            print(f"  MISSING  {rel}  (no source named {base} under figures/)")
            missing += 1
            continue
        if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
            uptodate += 1
            continue
        if args.check:
            print(f"  WOULD COPY  {src.relative_to(REPO)}  ->  {rel}")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            print(f"  copied  {src.relative_to(REPO)}  ->  {rel}")
        copied += 1

    verb = "would copy" if args.check else "copied"
    print(f"\n{verb} {copied}, already current {uptodate}, missing {missing}")
    if missing:
        print("Missing figures need their generating script run first "
              "(see scripts/figures/make_*.py).")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
