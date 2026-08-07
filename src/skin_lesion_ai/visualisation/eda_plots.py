from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve


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

STATISTIC_PALETTE = {
    "distribution": "#4C72B0",
    "mean": "#C44E52",
    "median": "#55A868",
}

PREDICTIVE_PALETTE = {
    "hypothesis_1": "#4C72B0",
    "hypothesis_2": "#DD8452",
}

SIZE_GROUP_PALETTE = {
    "<3 mm": "#4C72B0",
    "3–<5 mm": "#6A9F58",
    "5–<7 mm": "#C44E52",
    "7–<10 mm": "#8172B3",
    "10–<15 mm": "#CCB974",
    "≥15 mm": "#64B5CD",
}

ANATOM_SITE_PALETTE = {
    "Torso posterior": "#4C72B0",
    "Extremidad inferior": "#6A9F58",
    "Torso anterior": "#DD8452",
    "Extremidad superior": "#8172B3",
    "Cabeza/cuello": "#C44E52",
    "Localización desconocida": "#9E9E9E",
}

# ---------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------

ANATOM_SITE_LABELS = {
    "posterior torso": "Torso posterior",
    "lower extremity": "Extremidad inferior",
    "anterior torso": "Torso anterior",
    "upper extremity": "Extremidad superior",
    "head/neck": "Cabeza/cuello",
    "Missing": "Localización desconocida",
}

ANATOM_SITE_DISPLAY_LABELS = {
    "Posterior torso": "Torso posterior",
    "Lower extremity": "Extremidad inferior",
    "Anterior torso": "Torso anterior",
    "Upper extremity": "Extremidad superior",
    "Head/neck": "Cabeza/cuello",
    "Missing": "Localización desconocida",
}

ANATOM_SITE_ORDER = [
    "Torso posterior",
    "Extremidad inferior",
    "Torso anterior",
    "Extremidad superior",
    "Cabeza/cuello",
    "Localización desconocida",
]

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


# ---------------------------------------------------------------------
# Model evaluation plots
# ---------------------------------------------------------------------


def save_validation_evaluation_plots(
    train_data,
    validation_data,
    selected_threshold: float,
    target_sensitivity: float,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save the four standard train/validation evaluation figures.

    Both dataframes must contain ``y_true`` and ``y_prob``. The clinical
    summary uses validation only because the operating threshold is selected
    on that split.
    """
    set_eda_style()
    output_dir = Path(output_dir)

    train_color = PREDICTIVE_PALETTE["hypothesis_1"]
    validation_color = PREDICTIVE_PALETTE["hypothesis_2"]
    split_data = (
        ("Train", train_data, train_color),
        ("Validation", validation_data, validation_color),
    )

    def save_direct(fig, name: str) -> Path:
        path = save_plot(
            fig=fig,
            name=name,
            subfolder="",
            plots_dir=output_dir,
            extension="jpg",
            add_timestamp=False,
        )
        plt.close(fig)
        return path

    # 1. Precision-Recall curve
    fig_pr, ax_pr = plt.subplots(figsize=EDA_STYLE["figure_size"])
    for split_name, data, color in split_data:
        precision, recall, _ = precision_recall_curve(data["y_true"], data["y_prob"])
        score = auc(recall, precision)
        ax_pr.plot(
            recall,
            precision,
            color=color,
            linewidth=2,
            label=f"{split_name} (PR-AUC = {score:.3f})",
        )

    validation_prevalence = validation_data["y_true"].mean()
    ax_pr.axhline(
        validation_prevalence,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        label=f"Validation baseline = {validation_prevalence:.3f}",
    )
    ax_pr.set(
        xlabel="Recall (sensitivity)",
        ylabel="Precision (PPV)",
        title="Precision-Recall curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax_pr.legend(loc="best")
    pr_path = save_direct(fig_pr, "precision_recall_curve")

    # Common validation operating-point metrics
    y_true = validation_data["y_true"].to_numpy(dtype=int)
    y_prob = validation_data["y_prob"].to_numpy(dtype=float)
    y_pred = (y_prob >= selected_threshold).astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    # 2. ROC curve
    fig_roc, ax_roc = plt.subplots(figsize=EDA_STYLE["figure_size"])
    for split_name, data, color in split_data:
        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        score = roc_auc_score(data["y_true"], data["y_prob"])
        ax_roc.plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label=f"{split_name} (ROC-AUC = {score:.3f})",
        )

    ax_roc.scatter(
        1 - specificity,
        sensitivity,
        color=validation_color,
        edgecolor="black",
        s=65,
        zorder=4,
        label=f"Selected threshold = {selected_threshold:.3f}",
    )
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.2)
    ax_roc.set(
        xlabel="False positive rate",
        ylabel="Sensitivity",
        title="ROC curve",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    ax_roc.legend(loc="lower right")
    roc_path = save_direct(fig_roc, "roc_curve")

    # 3. Combined PR and ROC figure
    fig_combined, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for split_name, data, color in split_data:
        precision, recall, _ = precision_recall_curve(data["y_true"], data["y_prob"])
        axes[0].plot(
            recall,
            precision,
            color=color,
            linewidth=2,
            label=f"{split_name} ({auc(recall, precision):.3f})",
        )

        fpr, tpr, _ = roc_curve(data["y_true"], data["y_prob"])
        axes[1].plot(
            fpr,
            tpr,
            color=color,
            linewidth=2,
            label=f"{split_name} ({roc_auc_score(data['y_true'], data['y_prob']):.3f})",
        )

    axes[0].axhline(
        validation_prevalence,
        color="#777777",
        linestyle="--",
        linewidth=1.2,
        label=f"Baseline ({validation_prevalence:.3f})",
    )
    axes[0].set(
        xlabel="Recall (sensitivity)",
        ylabel="Precision (PPV)",
        title="Precision-Recall",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axes[0].legend(loc="best")

    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1.2)
    axes[1].scatter(
        1 - specificity,
        sensitivity,
        color=validation_color,
        edgecolor="black",
        s=60,
        zorder=4,
        label=f"Threshold ({selected_threshold:.3f})",
    )
    axes[1].set(
        xlabel="False positive rate",
        ylabel="Sensitivity",
        title="ROC",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axes[1].legend(loc="lower right")
    fig_combined.tight_layout()
    combined_path = save_direct(fig_combined, "pr_roc_curves")

    # 4. Validation clinical summary
    fig_summary = plt.figure(figsize=EDA_STYLE["wide_figure_size"])
    grid = GridSpec(1, 2, figure=fig_summary, width_ratios=[1.15, 1])
    ax_matrix = fig_summary.add_subplot(grid[0, 0])
    ax_metrics = fig_summary.add_subplot(grid[0, 1])

    sns.heatmap(
        np.array([[tn, fp], [fn, tp]]),
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=False,
        linewidths=0.8,
        linecolor="white",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
        ax=ax_matrix,
    )
    ax_matrix.set(
        xlabel="Predicted class",
        ylabel="True class",
        title="Validation confusion matrix",
    )

    ax_metrics.axis("off")
    ax_metrics.set_title("Clinical operating point")
    ax_metrics.text(
        0.05,
        0.95,
        (
            f"Selected threshold\n{selected_threshold:.4f}\n\n"
            f"Target sensitivity\n≥ {target_sensitivity:.1%}\n\n"
            f"Sensitivity\n{sensitivity:.2%}\n\n"
            f"Specificity\n{specificity:.2%}\n\n"
            f"PPV\n{ppv:.2%}\n\n"
            f"NPV\n{npv:.2%}"
        ),
        transform=ax_metrics.transAxes,
        ha="left",
        va="top",
        fontsize=EDA_STYLE["label_size"],
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "white",
            "edgecolor": validation_color,
            "linewidth": 1.5,
        },
    )
    fig_summary.tight_layout()
    summary_path = save_direct(fig_summary, "clinical_summary")

    return {
        "precision_recall_curve": pr_path,
        "roc_curve": roc_path,
        "pr_roc_curves": combined_path,
        "clinical_summary": summary_path,
    }
