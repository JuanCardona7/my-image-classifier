import sys
from subprocess import run

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from skin_lesion_ai.utils.data_utils import (
    get_project_root,
    load_metadata_parquet,
    save_metadata_parquet,
)


GROUP_COLUMN = "patient_id"
LESION_ID_COLUMN = "isic_id"


def split_stratified_group(
    df: pd.DataFrame,
    target_column: str,
    second_split_size: float,
    random_state: int,
    group_column: str = GROUP_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dataframe by patient while stratifying the selected target."""

    n_splits = int(round(1 / second_split_size))

    if not np.isclose(second_split_size, 1 / n_splits):
        raise ValueError(
            "second_split_size must correspond to one complete fold "
            "of StratifiedGroupKFold."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    main_indices, second_indices = next(
        splitter.split(
            X=df,
            y=df[target_column],
            groups=df[group_column],
        )
    )

    return (
        df.iloc[main_indices].copy(),
        df.iloc[second_indices].copy(),
    )


def create_train_val_test_split(
    df: pd.DataFrame,
    target_column: str,
    random_state: int,
) -> dict[str, pd.DataFrame]:
    """Create an independent 80/10/10 train, validation and test split."""

    train_df, test_val_df = split_stratified_group(
        df=df,
        target_column=target_column,
        second_split_size=0.2,
        random_state=random_state,
    )

    test_df, val_df = split_stratified_group(
        df=test_val_df,
        target_column=target_column,
        second_split_size=0.5,
        random_state=random_state,
    )

    return {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }


def validate_input_data(df: pd.DataFrame) -> None:
    """Validate the columns and identifiers required to generate both splits."""

    required_columns = {
        LESION_ID_COLUMN,
        GROUP_COLUMN,
        "target_biopsy",
        "target_malignant",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    if df[LESION_ID_COLUMN].isna().any():
        raise ValueError(f"{LESION_ID_COLUMN} contains missing values.")

    if not df[LESION_ID_COLUMN].is_unique:
        raise ValueError(f"{LESION_ID_COLUMN} must be unique.")

    if df[GROUP_COLUMN].isna().any():
        raise ValueError(f"{GROUP_COLUMN} contains missing values.")

    if df["target_biopsy"].isna().any():
        raise ValueError("target_biopsy contains missing values.")

    if not df["target_biopsy"].isin([0, 1]).all():
        raise ValueError("target_biopsy must contain only 0 and 1.")

    h2_mask = df["target_malignant"].notna()

    if not h2_mask.any():
        raise ValueError("No lesions are available for hypothesis 2.")

    if not df.loc[h2_mask, "target_malignant"].isin([0, 1]).all():
        raise ValueError("target_malignant must contain only 0, 1 or NA.")

    if not df.loc[h2_mask, "target_biopsy"].eq(1).all():
        raise ValueError(
            "All lesions with target_malignant must also have target_biopsy equal to 1."
        )


def validate_split(
    df_base: pd.DataFrame,
    splits: dict[str, pd.DataFrame],
    target_column: str,
    hypothesis: str,
    size_tolerance: float = 0.02,
    prevalence_tolerance: float = 0.02,
) -> None:
    """Validate coverage, class balance and absence of data leakage."""

    expected_proportions = {
        "train": 0.80,
        "val": 0.10,
        "test": 0.10,
    }

    patient_sets = {
        name: set(split_df[GROUP_COLUMN]) for name, split_df in splits.items()
    }

    lesion_sets = {
        name: set(split_df[LESION_ID_COLUMN]) for name, split_df in splits.items()
    }

    split_pairs = [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]

    for first, second in split_pairs:
        if patient_sets[first].intersection(patient_sets[second]):
            raise ValueError(
                f"{hypothesis}: patient leakage between {first} and {second}."
            )

        if lesion_sets[first].intersection(lesion_sets[second]):
            raise ValueError(
                f"{hypothesis}: lesion overlap between {first} and {second}."
            )

    all_split_lesions = set().union(*lesion_sets.values())
    all_base_lesions = set(df_base[LESION_ID_COLUMN])

    if all_split_lesions != all_base_lesions:
        raise ValueError(
            f"{hypothesis}: some lesions are missing or assigned incorrectly."
        )

    base_prevalence = df_base[target_column].mean()

    print(f"\n{hypothesis} split validation")
    print("-" * (len(hypothesis) + 17))
    print(
        f"Base: {len(df_base):,} lesions | "
        f"{df_base[GROUP_COLUMN].nunique():,} patients | "
        f"positive rate: {base_prevalence:.4%}"
    )

    for name in ("train", "val", "test"):
        split_df = splits[name]

        actual_proportion = len(split_df) / len(df_base)
        split_prevalence = split_df[target_column].mean()

        class_counts = (
            split_df[target_column]
            .astype(int)
            .value_counts()
            .reindex([0, 1], fill_value=0)
        )

        if abs(actual_proportion - expected_proportions[name]) > size_tolerance:
            raise ValueError(
                f"{hypothesis}: {name} size is outside the allowed tolerance."
            )

        if class_counts.eq(0).any():
            raise ValueError(
                f"{hypothesis}: {name} does not contain both target classes."
            )

        if abs(split_prevalence - base_prevalence) > prevalence_tolerance:
            raise ValueError(
                f"{hypothesis}: {name} target prevalence is outside "
                "the allowed tolerance."
            )

        print(
            f"{name}: {len(split_df):,} lesions "
            f"({actual_proportion:.2%}) | "
            f"{split_df[GROUP_COLUMN].nunique():,} patients | "
            f"0/1: {class_counts[0]:,}/{class_counts[1]:,} | "
            f"positive rate: {split_prevalence:.4%}"
        )

    print(f"{hypothesis}: validation passed.")


def load_or_create_final_metadata() -> pd.DataFrame:
    """Load the latest final metadata file or generate it from raw data."""

    repo_root = get_project_root()
    script_path = repo_root / "scripts" / "final_preprocess_data.py"

    try:
        return load_metadata_parquet(
            stage="processed",
            filename="final_preprocessed_from_raw",
            timestamp_flag=True,
        )

    except FileNotFoundError:
        print(
            "No final_preprocessed_from_raw parquet was found. "
            "Running final_preprocess_data.py..."
        )

        run(
            [sys.executable, str(script_path)],
            cwd=str(repo_root),
            check=True,
        )

        return load_metadata_parquet(
            stage="processed",
            filename="final_preprocessed_from_raw",
            timestamp_flag=True,
        )


def save_splits(
    splits: dict[str, pd.DataFrame],
    hypothesis: str,
) -> None:
    """Save train, validation and test metadata using the project convention."""

    for split_name, split_df in splits.items():
        output_path = save_metadata_parquet(
            split_df,
            stage="processed",
            name=f"{split_name}_split_{hypothesis.lower()}",
            timestamp=True,
        )

        print(f"{hypothesis} {split_name} saved to: {output_path}")


def main() -> None:
    """Generate, validate and save independent splits for H1 and H2."""

    df_preprocessed = load_or_create_final_metadata()

    print(f"Final preprocessed metadata loaded: {df_preprocessed.shape}")

    validate_input_data(df_preprocessed)

    # H1 uses all lesions and keeps only the biopsy target.
    df_h1 = df_preprocessed.drop(columns="target_malignant").copy()

    df_h1["target_biopsy"] = df_h1["target_biopsy"].astype("int8")

    # H2 uses only lesions with a known malignancy label
    # and keeps only the malignancy target.
    df_h2 = (
        df_preprocessed.loc[df_preprocessed["target_malignant"].notna()]
        .drop(columns="target_biopsy")
        .copy()
    )

    df_h2["target_malignant"] = df_h2["target_malignant"].astype("int8")

    h1_splits = create_train_val_test_split(
        df=df_h1,
        target_column="target_biopsy",
        random_state=42,
    )

    validate_split(
        df_base=df_h1,
        splits=h1_splits,
        target_column="target_biopsy",
        hypothesis="H1",
    )

    h2_splits = create_train_val_test_split(
        df=df_h2,
        target_column="target_malignant",
        random_state=37,
    )

    validate_split(
        df_base=df_h2,
        splits=h2_splits,
        target_column="target_malignant",
        hypothesis="H2",
    )

    save_splits(
        splits=h1_splits,
        hypothesis="H1",
    )

    save_splits(
        splits=h2_splits,
        hypothesis="H2",
    )


if __name__ == "__main__":
    main()
