"""
Thesis figure style: one visual system for every plot in the document.

WHY THIS EXISTS
---------------
Figures drawn ad-hoc in notebooks drift. Panel A ends up 12x6 inches with a
`blue` line and 14pt sans-serif labels; panel B ends up 10x4 with viridis and
11pt. Bound together in a thesis they read as five different documents. The fix
is the same one used for code: hoist the shared decisions into one module and
have every figure *import* them rather than restate them.

This is the plotting analogue of the Single Responsibility Principle — the
figure script decides *what data is shown*; this module decides *what it looks
like*. Changing the thesis font later is then a one-line edit here, not a
find-and-replace across every notebook.

DESIGN DECISIONS ENCODED HERE
-----------------------------
1. Serif type at document size. Matplotlib's sans-serif default is visually
   foreign inside a serif thesis; math is set in Computer Modern so that
   `$\\delta_t$` in a figure matches `$\\delta_t$` in the body text.

2. Figures are sized in *final printed inches*, never scaled by LaTeX.
   `\\includegraphics[width=\\textwidth]` on an oversized figure shrinks the
   fonts along with it, which is why notebook figures come out with unreadable
   axis labels. Draw at final size, include at scale 1.0.

3. A validated colour palette (see PALETTE below), not a colormap picked by
   feel. Hues are assigned to *meaning* (spot / convenience yield), fixed once,
   and never cycled — so a colour means the same thing in every chapter.

4. Colour is never the only channel. Every categorical distinction is paired
   with a line style or a direct label, so the figures survive greyscale
   printing and colour-vision deficiency.

USAGE
-----
    from gs_wamol.utils.thesis_style import use_thesis_style, PALETTE, figsize, save
    use_thesis_style()
    fig, axes = plt.subplots(1, 2, figsize=figsize("full", ratio=0.42))
    ...
    save(fig, "mc1_state_paths")
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
# Text width of a standard A4 thesis body (12pt, ~1in margins) in inches.
# Measure yours once with \the\textwidth in LaTeX (it prints in pt; /72.27 for
# inches) and set it here. Everything else is derived from this single number,
# so the whole document rescales from one edit.
TEXT_WIDTH_IN = 6.30

# Named fractions of the text block. "full" spans the measure; "half" is for
# two figures placed side by side as separate LaTeX floats (note this is NOT
# the same as a two-panel figure, which is still "full").
_WIDTH_FRACTIONS = {"full": 1.0, "wide": 0.85, "half": 0.48, "third": 0.31}


def figsize(width: str = "full", ratio: float = 0.618) -> tuple[float, float]:
    """Return (w, h) in inches for a given text-width fraction and aspect ratio.

    `ratio` is height/width. Defaults to 1/phi (~0.618), which is a comfortable
    single-panel proportion. For a 1x2 side-by-side panel, each subplot is
    already half as wide, so pass a *smaller* ratio (~0.40-0.45) or the pair
    comes out unusably tall.
    """
    if width not in _WIDTH_FRACTIONS:
        raise ValueError(f"width must be one of {sorted(_WIDTH_FRACTIONS)}, got {width!r}")
    w = TEXT_WIDTH_IN * _WIDTH_FRACTIONS[width]
    return (w, w * ratio)


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------
# Categorical hues, in fixed slot order. This exact four-colour set was checked
# with a palette validator over ALL pairs (not just adjacent ones, because
# these appear together in scatter plots where any two may sit side by side):
#
#   lightness band  PASS   chroma floor  PASS   contrast vs paper  PASS
#   CVD separation  PASS   worst all-pairs dE 9.2 (deuteranopia), target >= 8
#
# The obvious first choice — blue/orange/GREEN — fails: green vs orange is
# dE 3.2 under protanopia, i.e. indistinguishable to ~1% of male readers. That
# is precisely the kind of thing you cannot catch by looking at your own screen,
# which is why the palette is validated rather than chosen.
PALETTE = {
    "blue": "#2a78d6",
    "orange": "#eb6834",
    "violet": "#4a3aa7",
    "aqua": "#1baf7a",
}

# Semantic aliases. Figure code should use THESE, not the hue names, so that the
# meaning of a colour is legible at the call site and stays stable document-wide.
# Rule: a colour follows the *entity*, never its position in a plot.
C_SPOT = PALETTE["blue"]        # S_t and anything priced off it (futures)
C_DELTA = PALETTE["orange"]     # delta_t, the latent inversion target
C_MODEL = PALETTE["violet"]     # fitted / recovered quantities (PINN output)
C_BASELINE = PALETTE["aqua"]    # the Kalman filter benchmark

# Neutral ink ramp. Used for structure that is context rather than subject:
# path ensembles, reference lines, grid, axis labels. Keeping these grey is what
# lets the two accent hues actually carry meaning.
INK = {
    "primary": "#1a1a19",    # titles, axis labels
    "secondary": "#52514e",  # tick labels, annotations
    "muted": "#8a8983",      # reference lines, direct-label leader lines
    "faint": "#c9c8c2",      # ensemble paths, grid
}

# Sequential ramp for ORDERED magnitude (time to maturity, calendar date).
# One hue, light -> dark. Maturity is an ordered variable, so it must not get a
# categorical or rainbow map: with viridis the reader has to consult a legend to
# recover the ordering that "darker = longer" gives away for free. Steps start
# at the light end no paler than 2:1 against white, so the first curve is still
# visible on paper.
_BLUE_STEPS = [
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQ_BLUE = LinearSegmentedColormap.from_list("thesis_seq_blue", _BLUE_STEPS)

# Diverging ramp for POLARITY — a quantity read as above/below a meaningful
# centre, e.g. delta_t relative to its long-run mean alpha, or backwardation vs
# contango. Two opposed hues with a NEUTRAL GREY midpoint: the midpoint must
# read as "nothing", which is why a rainbow (where the middle is a vivid green)
# misleads here.
DIV_BLUE_RED = LinearSegmentedColormap.from_list(
    "thesis_div", ["#184f95", "#2a78d6", "#e8e7e3", "#e34948", "#a32020"]
)


# ---------------------------------------------------------------------------
# The style itself
# ---------------------------------------------------------------------------

def use_thesis_style(usetex: bool = False) -> None:
    """Install the thesis rcParams globally. Call once at the top of a script.

    `usetex=True` routes all text through a real LaTeX installation, giving
    perfect typographic identity with the document at the cost of a much slower
    render and a hard dependency on a working TeX toolchain. The default
    (`False`) uses matplotlib's bundled Computer Modern mathtext, which is
    visually near-identical for the symbols this thesis uses and always works.
    """
    mpl.rcParams.update({
        # -- type ------------------------------------------------------------
        "font.family": "serif",
        "font.serif": ["CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",   # match the document's math font
        "text.usetex": usetex,
        # Sizes are absolute points at final print size. 9pt body / 8pt ticks
        # sits one step below the thesis's 11-12pt text, which is the
        # conventional relationship for figure type.
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,

        # -- ink -------------------------------------------------------------
        "text.color": INK["primary"],
        "axes.labelcolor": INK["primary"],
        "axes.edgecolor": INK["muted"],
        "xtick.color": INK["secondary"],
        "ytick.color": INK["secondary"],

        # -- chart furniture recedes ------------------------------------------
        # Only the left and bottom spines are kept. The top/right spines box the
        # data in without conveying anything; removing them lowers the ink-to-
        # information ratio and is standard in published figures.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.grid": True,
        "grid.color": INK["faint"],
        "grid.linewidth": 0.4,
        "grid.alpha": 0.7,
        "axes.axisbelow": True,   # grid behind the data, never over it

        # -- marks -----------------------------------------------------------
        # 1.4pt is the thin-mark target: heavy enough to survive printing,
        # light enough that overlapping paths stay individually readable.
        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "lines.solid_capstyle": "round",
        "axes.prop_cycle": mpl.cycler(color=list(PALETTE.values())),

        # -- legend ----------------------------------------------------------
        "legend.frameon": False,   # the frame adds a box without adding meaning
        "legend.handlelength": 1.6,
        "legend.borderpad": 0.3,
        "legend.columnspacing": 1.2,

        # -- output ----------------------------------------------------------
        "figure.dpi": 140,          # on-screen preview only
        "savefig.dpi": 400,         # raster fallback; PDF output is vector
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,         # embed real fonts, not Type-3 outlines --
                                    # required for text-searchable, editable PDFs
                                    # and by most journal/university submission
                                    # checkers
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIGURE_DIR = Path(__file__).resolve().parents[3] / "notebooks" / "figures" / "methodology"


def save(fig, name: str, directory: Path | None = None) -> Path:
    """Write `name` as both PDF (for LaTeX) and PNG (for previewing in notebooks).

    PDF is the format that actually goes in the thesis: it stays vector, so the
    lines and type remain sharp at any zoom and at the printer's resolution. The
    PNG exists purely so the figure can be eyeballed quickly.
    """
    directory = directory or FIGURE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    pdf_path = directory / f"{name}.pdf"
    fig.savefig(pdf_path)
    fig.savefig(directory / f"{name}.png")
    return pdf_path


def direct_label(ax, x, y, text, color, *, dx=6, dy=0, ha="left", va="center", size=8):
    """Place a series label next to the line itself instead of in a legend box.

    A legend forces the reader to make a colour->name lookup for every glance.
    A label at the end of the line removes that indirection entirely, and it is
    what keeps the figure readable in greyscale, where the colour lookup fails.
    Reserve the legend for when lines are too dense to label in place.
    """
    ax.annotate(
        text, xy=(x, y), xycoords="data",
        xytext=(dx, dy), textcoords="offset points",
        color=color, ha=ha, va=va, fontsize=size,
    )


def panel_tag(ax, letter: str, *, dx=-0.14, dy=1.06):
    """Stamp an (a)/(b) tag on a subplot so the caption can refer to panels."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color=INK["primary"],
            ha="left", va="top")


def despine_shared_y(ax):
    """Hide y tick labels on the right-hand panel of a shared-scale pair.

    Repeating identical tick labels on both panels of a side-by-side comparison
    is redundant ink, and worse, it subtly implies the two axes are independent
    when the whole point of the pairing is that they are the same scale.
    """
    ax.tick_params(labelleft=False)
    ax.set_ylabel("")
