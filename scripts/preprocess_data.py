from __future__ import annotations

import numpy as np
import pandas as pd

from skin_lesion_ai.utils.data_utils import (
    load_raw_metadata,
    save_metadata_parquet,
)

"""Load raw dfs and generate a final dataframe of preprocessed data after EDA_01.ipynb.

Persist the final dataframe to `data/interim/metadata` as
`preprocessed_from_raw_{timestamp}.parquet` via `save_metadata_parquet()`.
"""


def assign_diagnostic_group(df: pd.DataFrame) -> pd.Series:
    """Create the four-category diagnostic label used for modelling."""

    conditions = [
        (df["iddx_1"] == "Benign") & df["iddx_2"].isna(),
        (df["iddx_1"] == "Benign") & df["iddx_2"].notna(),
        df["iddx_1"] == "Indeterminate",
        df["iddx_1"] == "Malignant",
    ]

    choices = [
        "benign_non_biopsied",
        "benign_biopsied",
        "indeterminate_biopsied",
        "malignant_biopsied",
    ]

    return pd.Series(
        np.select(conditions, choices, default=pd.NA),
        index=df.index,
        dtype="string",
    )


def main() -> None:
    _, df2, df3 = load_raw_metadata()

    for name, df in (("df2", df2), ("df3", df3)):
        if df["isic_id"].duplicated().any():
            duplicates = df.loc[df["isic_id"].duplicated(), "isic_id"].unique().tolist()
            raise ValueError(
                f"{name} contains duplicate isic_id values. Duplicates: {duplicates}"
            )

    merged = df2.merge(df3, how="inner", on="isic_id", validate="one_to_one")

    keep_columns = [
        "isic_id",
        "patient_id",
        "attribution",
        "copyright_license",
        "iddx_1",
        "iddx_2",
        "age_approx",
        "sex",
        "anatom_site_general",
        "clin_size_long_diam_mm",
        "tbp_lv_nevi_confidence",
        "tbp_lv_dnn_lesion_confidence",
        "tbp_lv_location",
        "tbp_lv_location_simple",
    ]

    missing_columns = [col for col in keep_columns if col not in merged.columns]
    if missing_columns:
        raise KeyError(f"Missing expected columns after merge: {missing_columns}")

    final_df = merged.loc[:, keep_columns].copy()
    final_df["diagnostic_group"] = assign_diagnostic_group(final_df)
    final_df = final_df.drop(columns=["iddx_1", "iddx_2"])

    output_path = save_metadata_parquet(
        final_df,
        stage="interim",
        name="preprocessed_from_raw",
        timestamp=True,
    )
    print(f"Saved preprocessed interim data to: {output_path}")


if __name__ == "__main__":
    main()
