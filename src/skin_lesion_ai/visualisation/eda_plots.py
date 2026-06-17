from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter


# ---------------------------------------------------------------------
# Global plotting settings
# ---------------------------------------------------------------------

EDA_STYLE = {
    "figure_size": (7, 4),
    "wide_figure_size": (9, 4.5),
    "title_size": 13,
    "label_size": 11,
    "tick_size": 10,
    "bar_alpha": 0.85,
    "dpi": 300,
}

EDA_PALETTE = {
    "female": "#4C72B0",
    "male": "#55A868",
    "missing": "#9E9E9E",
}

SEX_PALETTE = {
    "Mujer": "#4C72B0",
    "Hombre": "#DD8452",
    "Sexo desconocido": "#9E9E9E",
}

ATTRIBUTION_PALETTE = {
    "ACEMID": "#4C72B0",
    "Hospital Clínic": "#6A9F58",
    "Univ. Athens": "#C44E52",
    "Frazer Institute": "#8172B3",
    "MSKCC": "#CCB974",
    "Basel": "#64B5CD",
    "ViDIR Vienna": "#8C8C8C",
}

# ---------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------

SEX_LABELS = {
    "female": "Mujer",
    "male": "Hombre",
    "Missing": "Sexo desconocido",
}

SEX_ORDER = ["Mujer", "Hombre", "Sexo desconocido"]

DIAGNOSTIC_LABELS = {
    "benign_non_biopsied": "Benigna no biopsiada",
    "benign_biopsied": "Benigna biopsiada",
    "indeterminate_biopsied": "Indeterminada biopsiada",
    "malignant_biopsied": "Maligna biopsiada",
}

ATTRIBUTION_LABELS = {
    "ACEMID": "ACEMID",
    "Hospital Clínic": "Hospital Clínic",
    "Athens": "Univ. Athens",
    "Queensland": "Frazer Institute",
    "Frazer": "Frazer Institute",
    "Memorial Sloan": "MSKCC",
    "Basel": "Basel",
    "ViDIR": "ViDIR Vienna",
    "Vienna": "ViDIR Vienna",
}


def simplify_attribution_name(attribution: str) -> str:
    """
    Convert long attribution names into short display labels.
    """
    attribution = str(attribution)

    for keyword, label in ATTRIBUTION_LABELS.items():
        if keyword in attribution:
            return label

    return attribution


def simplify_attribution_series(series):
    """
    Convert attribution names in a pandas Series to
    short display labels.
    """
    return series.apply(simplify_attribution_name)


# ---------------------------------------------------------------------
# Style functions
# ---------------------------------------------------------------------


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
    name: str,
    subfolder: str = "EDA",
    plots_dir: str | Path = "plots",
    extension: str = "png",
    add_timestamp: bool = True,
    tight: bool = True,
) -> Path:
    """
    Save a figure inside plots/<subfolder>/ using a given name.

    The output filename can include a timestamp to avoid overwriting
    previous versions.
    """
    plots_dir = Path(plots_dir)
    output_dir = plots_dir / subfolder
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_name = name.strip().lower().replace(" ", "_").replace("/", "_")

    if add_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_name}_{timestamp}.{extension}"
    else:
        filename = f"{safe_name}.{extension}"

    output_path = output_dir / filename

    fig.savefig(
        output_path,
        bbox_inches="tight" if tight else None,
    )

    return output_path


def format_percent_axis(
    ax: plt.Axes,
    axis: str = "y",
    decimals: int = 0,
) -> None:
    """
    Format x or y axis as percentage.
    """
    formatter = PercentFormatter(xmax=100, decimals=decimals)

    if axis == "y":
        ax.yaxis.set_major_formatter(formatter)
    elif axis == "x":
        ax.xaxis.set_major_formatter(formatter)
    else:
        raise ValueError("axis must be either 'x' or 'y'.")


def add_bar_labels(
    ax: plt.Axes,
    fmt: str = "{:.2f}",
    suffix: str = "",
    padding: float = 2,
    fontsize: int | None = None,
) -> None:
    """
    Add value labels to vertical bars.
    """
    if fontsize is None:
        fontsize = EDA_STYLE["tick_size"]

    for patch in ax.patches:
        height = patch.get_height()

        if height == 0:
            continue

        ax.annotate(
            f"{fmt.format(height)}{suffix}",
            xy=(
                patch.get_x() + patch.get_width() / 2,
                height,
            ),
            xytext=(0, padding),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )


def clean_axis_labels(
    ax: plt.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
) -> None:
    """
    Apply common axis labels and title.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)


def apply_label_mapping(series, mapping: dict):
    """
    Replace raw category values by display labels.
    """
    return series.map(mapping).fillna(series)
