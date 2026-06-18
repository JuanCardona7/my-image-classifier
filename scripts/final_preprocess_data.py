from __future__ import annotations

import sys
from subprocess import run

import numpy as np
import pandas as pd

from skin_lesion_ai.utils.data_utils import (
    get_project_root,
    load_metadata_parquet,
    save_metadata_parquet,
)

"""
Generate the final preprocessed metadata dataframe after EDA_02.ipynb.

This script takes the interim dataframe generated after EDA_01
(`preprocessed_from_raw_<timestamp>.parquet`) and applies the preprocessing
decisions derived from EDA_02.

If the interim dataframe does not exist, the script first runs
`scripts/preprocess_data.py`, which generates it from the raw metadata files.

Preprocessing decisions applied
-------------------------------

1. Keep only the modelling-relevant columns:
   - lesion identifier: `isic_id`
   - patient identifier: `patient_id`
   - original diagnostic group: `diagnostic_group`
   - four metadata predictors:
       * `sex`
       * `age_approx`
       * `anatom_site_general`
       * `clin_size_long_diam_mm`

2. Remove records according to the EDA_02 recommendations:
   - remove patients with missing `sex`
   - remove patients with missing `age_approx`
   - remove patients younger than 18 years
   - remove patients with systematic or quasi-systematic missing
     `anatom_site_general`, defined as >=97% of their lesions missing
     anatomical site
   - remove only the punctual lesions with missing `anatom_site_general`
     among patients not already removed at patient level

3. Add target variables:
   - `target_biopsy`:
       0 = benign non-biopsied lesion
       1 = biopsied lesion
       This corresponds to the main modelling hypothesis: biopsy
       recommendation.
   - `target_malignant`:
       1 = malignant biopsied lesion
       0 = benign biopsied or indeterminate biopsied lesion
       <NA> = benign non-biopsied lesion
       This corresponds to the secondary modelling hypothesis: malignancy
       classification among biopsied lesions.

4. Add initial feature encodings while preserving the original variables:
   - `sex_male`:
       1 = male
       0 = female
   - `anatom_site_general_code`:
       integer encoding of anatomical site, mainly for algorithms or analyses
       requiring a single numeric representation. This encoding should be used
       with caution because it introduces an artificial order.
   - one-hot encoded anatomical site columns:
       * `anatom_site__anterior_torso`
       * `anatom_site__head_neck`
       * `anatom_site__lower_extremity`
       * `anatom_site__posterior_torso`
       * `anatom_site__upper_extremity`
   - `clin_size_long_diam_mm_log1p`:
       log1p transformation of lesion diameter.

Final dataframe columns
-----------------------

The final dataframe contains:

- `isic_id`
- `patient_id`
- `diagnostic_group`
- `target_biopsy`
- `target_malignant`
- `sex`
- `sex_male`
- `age_approx`
- `anatom_site_general`
- `anatom_site_general_code`
- `anatom_site__anterior_torso`
- `anatom_site__head_neck`
- `anatom_site__lower_extremity`
- `anatom_site__posterior_torso`
- `anatom_site__upper_extremity`
- `clin_size_long_diam_mm`
- `clin_size_long_diam_mm_log1p`

The resulting dataframe is saved to `data/processed/metadata` as:
`final_preprocessed_from_raw_<timestamp>.parquet`.
"""


ANATOM_SITE_MAPPING: dict[str, int] = {
    "anterior torso": 1,
    "posterior torso": 2,
    "upper extremity": 3,
    "lower extremity": 4,
    "head/neck": 5,
}

ANATOM_SITE_DUMMY_COLUMNS: list[str] = [
    "anatom_site__anterior_torso",
    "anatom_site__head_neck",
    "anatom_site__lower_extremity",
    "anatom_site__posterior_torso",
    "anatom_site__upper_extremity",
]

FINAL_COLUMNS: list[str] = [
    "isic_id",
    "patient_id",
    "diagnostic_group",
    "target_biopsy",
    "target_malignant",
    "sex",
    "sex_male",
    "age_approx",
    "anatom_site_general",
    "anatom_site_general_code",
    "anatom_site__anterior_torso",
    "anatom_site__head_neck",
    "anatom_site__lower_extremity",
    "anatom_site__posterior_torso",
    "anatom_site__upper_extremity",
    "clin_size_long_diam_mm",
    "clin_size_long_diam_mm_log1p",
]


def load_or_create_interim_metadata() -> pd.DataFrame:
    """Load latest interim metadata file or generate it from raw data."""

    repo_root = get_project_root()
    script_path = repo_root / "scripts" / "preprocess_data.py"

    try:
        return load_metadata_parquet(
            stage="interim",
            filename="preprocessed_from_raw",
            timestamp_flag=True,
        )

    except FileNotFoundError:
        run(
            [sys.executable, str(script_path)],
            cwd=str(repo_root),
            check=True,
        )

        return load_metadata_parquet(
            stage="interim",
            filename="preprocessed_from_raw",
            timestamp_flag=True,
        )


def get_patients_with_missing(df: pd.DataFrame, column: str) -> np.ndarray:
    """Return unique patient IDs with at least one missing value in a column."""

    return df.loc[df[column].isna(), "patient_id"].dropna().unique()


def get_patients_under_18(df: pd.DataFrame) -> np.ndarray:
    """Return unique patient IDs with age_approx below 18."""

    return (
        df.loc[
            df["age_approx"].notna() & (df["age_approx"] < 18),
            "patient_id",
        ]
        .dropna()
        .unique()
    )


def summarise_anatom_site_missing_by_patient(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise missing anatomical site values at patient level."""

    summary = (
        df.groupby("patient_id", observed=True)
        .agg(
            n_lesions=("isic_id", "count"),
            n_missing_anatom_site=(
                "anatom_site_general",
                lambda values: values.isna().sum(),
            ),
            n_available_anatom_site=(
                "anatom_site_general",
                lambda values: values.notna().sum(),
            ),
        )
        .reset_index()
    )

    summary["pct_missing_anatom_site"] = (
        summary["n_missing_anatom_site"] / summary["n_lesions"] * 100
    )

    return summary


def get_anatom_site_exclusions(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return patient-level and lesion-level exclusions for anatom_site_general.

    Patients with >=97% missing anatomical site are removed completely.
    For remaining patients with punctual missing anatomical site, only the
    affected lesions are removed.
    """

    anatom_summary = summarise_anatom_site_missing_by_patient(df)

    affected = anatom_summary.loc[anatom_summary["n_missing_anatom_site"] > 0].copy()

    patients_systematic_missing = (
        affected.loc[
            affected["pct_missing_anatom_site"] >= 97,
            "patient_id",
        ]
        .dropna()
        .tolist()
    )

    patients_punctual_missing = (
        affected.loc[
            affected["pct_missing_anatom_site"] < 97,
            "patient_id",
        ]
        .dropna()
        .tolist()
    )

    return patients_systematic_missing, patients_punctual_missing


def build_exclusion_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Build final patient-level and lesion-level exclusion lists.

    Returns
    -------
    tuple[list[str], list[str]]
        First element: patient IDs to remove completely.
        Second element: lesion IDs to remove individually.
    """

    patients_sex_missing = get_patients_with_missing(df, "sex")
    patients_age_missing = get_patients_with_missing(df, "age_approx")
    patients_under_18 = get_patients_under_18(df)

    (
        patients_anatom_systematic_missing,
        patients_anatom_punctual_missing,
    ) = get_anatom_site_exclusions(df)

    patients_to_remove = sorted(
        set(patients_sex_missing)
        | set(patients_age_missing)
        | set(patients_under_18)
        | set(patients_anatom_systematic_missing)
    )

    lesions_to_remove = (
        df.loc[
            df["patient_id"].isin(patients_anatom_punctual_missing)
            & ~df["patient_id"].isin(patients_to_remove)
            & df["anatom_site_general"].isna(),
            "isic_id",
        ]
        .dropna()
        .sort_values()
        .tolist()
    )

    print(f"Patients with sex missing: {len(set(patients_sex_missing))}")
    print(f"Patients with age_approx missing: {len(set(patients_age_missing))}")
    print(f"Patients younger than 18 years: {len(set(patients_under_18))}")
    print(
        "Patients with systematic/quasi-systematic anatomical site missing: "
        f"{len(set(patients_anatom_systematic_missing))}"
    )
    print(f"Final unique patients to remove: {len(patients_to_remove)}")
    print(
        f"Punctual anatomical-site missing lesions to remove: {len(lesions_to_remove)}"
    )

    return patients_to_remove, lesions_to_remove


def apply_exclusions(
    df: pd.DataFrame,
    patients_to_remove: list[str],
    lesions_to_remove: list[str],
) -> pd.DataFrame:
    """Apply final patient-level and lesion-level exclusions."""

    return df.loc[
        ~df["patient_id"].isin(patients_to_remove)
        & ~df["isic_id"].isin(lesions_to_remove)
    ].copy()


def add_target_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary target variables for the two modelling hypotheses."""

    df = df.copy()

    df["target_biopsy"] = np.where(
        df["diagnostic_group"].eq("benign_non_biopsied"),
        0,
        1,
    ).astype("int8")

    df["target_malignant"] = pd.Series(pd.NA, index=df.index, dtype="Int8")

    df.loc[
        df["diagnostic_group"].isin(["benign_biopsied", "indeterminate_biopsied"]),
        "target_malignant",
    ] = 0

    df.loc[
        df["diagnostic_group"].eq("malignant_biopsied"),
        "target_malignant",
    ] = 1

    return df


def add_sex_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary sex encoding while preserving the original sex column."""

    df = df.copy()

    sex_mapping = {
        "female": 0,
        "male": 1,
    }

    unexpected_values = sorted(
        set(df["sex"].dropna().unique()) - set(sex_mapping.keys())
    )

    if unexpected_values:
        raise ValueError(f"Unexpected values in sex column: {unexpected_values}")

    df["sex_male"] = df["sex"].map(sex_mapping).astype("int8")

    return df


def clean_anatom_site_value(value: str) -> str:
    """Convert anatomical site values into safe column-name suffixes."""

    return value.replace("/", "_").replace(" ", "_")


def add_anatom_site_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ordinal-like and one-hot encodings for anatom_site_general.

    The integer code is provided as an auxiliary representation and should
    be used with caution because anatomical site is nominal.
    """

    df = df.copy()

    unexpected_values = sorted(
        set(df["anatom_site_general"].dropna().unique())
        - set(ANATOM_SITE_MAPPING.keys())
    )

    if unexpected_values:
        raise ValueError(
            f"Unexpected values in anatom_site_general column: {unexpected_values}"
        )

    if df["anatom_site_general"].isna().any():
        raise ValueError(
            "Missing values remain in anatom_site_general after preprocessing."
        )

    df["anatom_site_general_code"] = (
        df["anatom_site_general"].map(ANATOM_SITE_MAPPING).astype("int8")
    )

    dummy_prefix = "anatom_site"
    dummies = pd.get_dummies(
        df["anatom_site_general"],
        prefix=dummy_prefix,
        prefix_sep="__",
        dtype="int8",
    )

    dummies = dummies.rename(
        columns={
            f"{dummy_prefix}__{site}": (
                f"{dummy_prefix}__{clean_anatom_site_value(site)}"
            )
            for site in ANATOM_SITE_MAPPING
        }
    )

    for column in ANATOM_SITE_DUMMY_COLUMNS:
        if column not in dummies.columns:
            dummies[column] = np.int8(0)

    dummies = dummies[ANATOM_SITE_DUMMY_COLUMNS]

    return pd.concat([df, dummies], axis=1)


def add_size_transformation(df: pd.DataFrame) -> pd.DataFrame:
    """Add log1p transformation of lesion diameter."""

    df = df.copy()

    if df["clin_size_long_diam_mm"].isna().any():
        raise ValueError("Missing values found in clin_size_long_diam_mm.")

    if (df["clin_size_long_diam_mm"] < 0).any():
        raise ValueError("Negative values found in clin_size_long_diam_mm.")

    df["clin_size_long_diam_mm_log1p"] = np.log1p(df["clin_size_long_diam_mm"])

    return df


def validate_final_dataframe(df: pd.DataFrame) -> None:
    """Run basic validation checks on the final dataframe."""

    missing_columns = [column for column in FINAL_COLUMNS if column not in df.columns]

    if missing_columns:
        raise KeyError(f"Missing final columns: {missing_columns}")

    if df["isic_id"].duplicated().any():
        duplicated_ids = (
            df.loc[df["isic_id"].duplicated(), "isic_id"].dropna().unique().tolist()
        )
        raise ValueError(f"Duplicated isic_id values found: {duplicated_ids}")

    mandatory_non_missing_columns = [
        "isic_id",
        "patient_id",
        "diagnostic_group",
        "target_biopsy",
        "sex",
        "sex_male",
        "age_approx",
        "anatom_site_general",
        "anatom_site_general_code",
        "clin_size_long_diam_mm",
        "clin_size_long_diam_mm_log1p",
    ]

    missing_counts = df[mandatory_non_missing_columns].isna().sum()
    missing_counts = missing_counts.loc[missing_counts > 0]

    if not missing_counts.empty:
        raise ValueError(
            f"Missing values remain in mandatory columns:\n{missing_counts}"
        )


def print_preprocessing_summary(
    original_df: pd.DataFrame,
    final_df: pd.DataFrame,
    patients_to_remove: list[str],
    lesions_to_remove: list[str],
) -> None:
    """Print a compact summary of the final preprocessing impact."""

    original_patients = original_df["patient_id"].nunique()
    final_patients = final_df["patient_id"].nunique()

    original_lesions = len(original_df)
    final_lesions = len(final_df)

    print("\nFinal preprocessing summary")
    print("---------------------------")
    print(f"Original patients: {original_patients}")
    print(f"Final patients: {final_patients}")
    print(f"Removed patients: {original_patients - final_patients}")
    print(f"Original lesions: {original_lesions}")
    print(f"Final lesions: {final_lesions}")
    print(f"Removed lesions: {original_lesions - final_lesions}")
    print(f"Patient-level exclusions: {len(patients_to_remove)}")
    print(f"Lesion-level punctual exclusions: {len(lesions_to_remove)}")


def main() -> None:
    """Run final preprocessing and save processed metadata parquet."""

    df = load_or_create_interim_metadata()

    required_columns = [
        "isic_id",
        "patient_id",
        "diagnostic_group",
        "sex",
        "age_approx",
        "anatom_site_general",
        "clin_size_long_diam_mm",
    ]

    missing_columns = [
        column for column in required_columns if column not in df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns in input dataframe: {missing_columns}"
        )

    patients_to_remove, lesions_to_remove = build_exclusion_lists(df)

    final_df = apply_exclusions(
        df=df,
        patients_to_remove=patients_to_remove,
        lesions_to_remove=lesions_to_remove,
    )

    final_df = add_target_variables(final_df)
    final_df = add_sex_encoding(final_df)
    final_df = add_anatom_site_encodings(final_df)
    final_df = add_size_transformation(final_df)

    final_df = final_df.loc[:, FINAL_COLUMNS].copy()

    validate_final_dataframe(final_df)

    output_path = save_metadata_parquet(
        final_df,
        stage="processed",
        name="final_preprocessed_from_raw",
        timestamp=True,
    )

    print_preprocessing_summary(
        original_df=df,
        final_df=final_df,
        patients_to_remove=patients_to_remove,
        lesions_to_remove=lesions_to_remove,
    )

    print(f"\nSaved final preprocessed metadata to: {output_path}")


if __name__ == "__main__":
    main()
