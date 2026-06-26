"""Modified from https://gist.github.com/emilyhunt/b1ef82df9939136c2791bfa66a9a76fd"""

import matplotlib.pyplot as plt
from pathlib import Path


# Define paths to where your figures should be saved.
FIGS = Path("./figures")
FIGS_PRES = Path("./figs_presentations")

# Figure size constants (in inches, for matplotlib's sake)
# Defined for the Astronomy & Astrophysics journal's style
COLWIDTH = 3.54
TEXTWIDTH = 7.24
HEIGHT = 9.75

# Color cycles
# Below are colorblind friendly color cycles from Petroff+21:
# https://arxiv.org/abs/2107.02270
cycle6 = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
cycle8 = [
    "#1845fb",
    "#ff5e02",
    "#c91f16",
    "#c849a9",
    "#adad7d",
    "#86c8dd",
    "#578dff",
    "#656364",
]
cycle10 = [
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
    "#832db6",
    "#a96b59",
    "#e76300",
    "#b9ac70",
    "#717581",
    "#92dadd",
]


def setup_matplotlib(no_log=False, latex=True):
    """Default matplotlib config. Inherited from Emily's 'Improving the open cluster
    census...' papers. Requires a local LaTeX install to generate LaTeX fonts properly.

    Args:
        no_log (bool): If your plots contain no log plots, set this to true - it will
            ensure that minorticks are automatically added to your plots. Otherwise,
            calling ax.minorticks_on() manually is necessary. Default: False
        latex (bool): Whether or not to use LaTeX for making prettier plot labels.
            Requires a local LaTeX install; I recommend installing something like
            TeXLive on your system if you don't have it. Default: True
    """
    if latex:
        plt.rc("text", usetex=True)
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")
        plt.rc("font", family="serif")
        
    plt.rc("font", size=10)
    plt.rc("legend", edgecolor="k", framealpha=1.0, fontsize=7)
    plt.rc(
        "figure", dpi=150, figsize=(COLWIDTH, 3.0), facecolor="w"
    )
    plt.rcParams['figure.constrained_layout.use'] = True
    plt.rc("savefig", dpi=300, format="pdf", facecolor="w", transparent=False)
    plt.rc("xtick", top=True, direction="in")
    plt.rc("ytick", right=True, direction="in")

    # Optional extras if you have no log plots, avoiding ax.minorticks_on():
    if no_log:
        plt.rc("xtick.minor", visible=True)
        plt.rc("ytick.minor", visible=True)
