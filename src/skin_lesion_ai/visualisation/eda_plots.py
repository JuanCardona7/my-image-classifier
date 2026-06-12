from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------
# Global plotting settings
# ---------------------------------------------------------------------

EDA_STYLE = {
    "figure_size": (7, 4),
    "title_size": 13,
    "label_size": 11,
    "tick_size": 10,
    "bar_alpha": 0.85,
    "dpi": 300,
}


def set_eda_style() -> None:
    """
    Apply common visual settings for all EDA plots.
    """
    sns.set_theme(
        style="whitegrid",
        context="notebook",
        font_scale=1.0,
    )

    plt.rcParams.update(
        {
            "figure.dpi": EDA_STYLE["dpi"],
            "savefig.dpi": EDA_STYLE["dpi"],
            "axes.titlesize": EDA_STYLE["title_size"],
            "axes.labelsize": EDA_STYLE["label_size"],
            "xtick.labelsize": EDA_STYLE["tick_size"],
            "ytick.labelsize": EDA_STYLE["tick_size"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_plot(
    fig: plt.Figure,
    output_path: str | Path | None = None,
    tight: bool = True,
) -> None:
    """
    Save a figure if an output path is provided.
    """
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            output_path,
            bbox_inches="tight" if tight else None,
        )
