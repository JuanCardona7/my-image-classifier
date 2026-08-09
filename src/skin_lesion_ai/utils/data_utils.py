from pathlib import Path
from datetime import datetime
import re

import pandas as pd
import yaml
import json

# -----------------------------------
# CONFIG PATH AND LOADING
# -----------------------------------


def get_project_root() -> Path:
    current = Path(__file__).resolve()

    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent

    raise FileNotFoundError("Could not find project root.")


def path(relative_path: str) -> Path:
    return get_project_root() / relative_path


def load_yaml_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


# -----------------------------------
# REPRODUCIBLE SUBSAMPLING
# -----------------------------------


def subsample_training_split(
    df: pd.DataFrame,
    n_samples: int | None,
    target_column: str,
    id_column: str = "isic_id",
    group_column: str = "patient_id",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Deterministically subsample a training dataset at patient level while
    approximately preserving the original positive-class prevalence.

    All observations from a selected patient are retained together.

    Patient ordering is based on a SHA-256 hash of (random_state, patient_id),
    making the selection independent of dataframe row order and reproducible
    across runs and machines.

    The requested n_samples is treated as a maximum number of observations.
    Because patient groups are never split, the returned dataset may contain
    slightly fewer than n_samples observations.

    Parameters
    ----------
    df:
        Full training dataframe.

    n_samples:
        Maximum number of observations to retain.
        If None or greater than or equal to len(df), the full dataframe
        is returned.

    target_column:
        Binary target column encoded as 0/1.

    id_column:
        Unique observation identifier. Default: "isic_id".

    group_column:
        Patient/group identifier. Default: "patient_id".

    random_state:
        Seed incorporated into the deterministic patient hash.
        The same input data, n_samples and random_state always produce
        the same subset. Default: 42.

    Returns
    -------
    pd.DataFrame
        Deterministically selected training subset containing complete
        patient groups and approximately preserving the original prevalence.
    """
    import hashlib

    # ---------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------

    required_columns = {
        id_column,
        group_column,
        target_column,
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")

    if df[id_column].isna().any():
        raise ValueError(f"Column '{id_column}' contains missing values.")

    if df[group_column].isna().any():
        raise ValueError(f"Column '{group_column}' contains missing values.")

    if df[target_column].isna().any():
        raise ValueError(f"Column '{target_column}' contains missing values.")

    if df[id_column].duplicated().any():
        duplicates = (
            df.loc[df[id_column].duplicated(), id_column].astype(str).head(10).tolist()
        )

        raise ValueError(
            f"Duplicated values found in '{id_column}'. Examples: {duplicates}"
        )

    target_values = set(df[target_column].astype(int).unique())

    if not target_values.issubset({0, 1}):
        raise ValueError(f"Column '{target_column}' must be binary and encoded as 0/1.")

    # ---------------------------------------------------------
    # Return full dataset when no subsampling is required
    # ---------------------------------------------------------

    if n_samples is None:
        return df.sort_values(id_column).reset_index(drop=True).copy()

    if not isinstance(n_samples, int) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer or None.")

    if n_samples >= len(df):
        return df.sort_values(id_column).reset_index(drop=True).copy()

    # ---------------------------------------------------------
    # Original training prevalence
    # ---------------------------------------------------------

    target_prevalence = float(df[target_column].mean())

    # ---------------------------------------------------------
    # Build patient-level summary
    # ---------------------------------------------------------

    patient_table = df.groupby(group_column, as_index=False).agg(
        n_observations=(id_column, "size"),
        n_positive=(target_column, "sum"),
    )

    patient_table["positive_rate"] = (
        patient_table["n_positive"] / patient_table["n_observations"]
    )

    # ---------------------------------------------------------
    # Deterministic patient ordering
    # ---------------------------------------------------------

    def deterministic_hash(patient_id: object) -> str:
        value = f"{random_state}|{str(patient_id)}"

        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    patient_table["_hash"] = patient_table[group_column].map(deterministic_hash)

    # Patients are divided according to whether their own
    # positive rate is above or below the population prevalence.
    #
    # This allows the algorithm to alternate between both groups
    # depending on the prevalence of the subset constructed so far.

    high_prevalence = (
        patient_table.loc[patient_table["positive_rate"] >= target_prevalence]
        .sort_values(
            ["_hash", group_column],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    low_prevalence = (
        patient_table.loc[patient_table["positive_rate"] < target_prevalence]
        .sort_values(
            ["_hash", group_column],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )

    # Convert to records for deterministic sequential access
    high_records = high_prevalence.to_dict("records")
    low_records = low_prevalence.to_dict("records")

    high_index = 0
    low_index = 0

    selected_patients: list[object] = []

    current_n = 0
    current_positive = 0

    # ---------------------------------------------------------
    # Helper to retrieve the next patient that still fits
    # ---------------------------------------------------------

    def get_next_fitting_patient(
        records: list[dict],
        start_index: int,
        remaining_capacity: int,
    ) -> tuple[dict | None, int]:
        index = start_index

        while index < len(records):
            candidate = records[index]
            index += 1

            if candidate["n_observations"] <= remaining_capacity:
                return candidate, index

        return None, index

    # ---------------------------------------------------------
    # Deterministic prevalence-aware patient selection
    # ---------------------------------------------------------

    while current_n < n_samples:
        remaining_capacity = n_samples - current_n

        if remaining_capacity <= 0:
            break

        if current_n == 0:
            current_prevalence = target_prevalence
        else:
            current_prevalence = current_positive / current_n

        # If the current subset has too few positives, preferentially
        # draw from patients with a relatively high positive rate.
        # Otherwise, draw from the lower-prevalence group.

        if current_prevalence < target_prevalence:
            preferred = "high"
        elif current_prevalence > target_prevalence:
            preferred = "low"
        else:
            # Deterministic tie-breaking:
            # choose the group whose next available hash comes first.

            high_hash = (
                high_records[high_index]["_hash"]
                if high_index < len(high_records)
                else None
            )

            low_hash = (
                low_records[low_index]["_hash"]
                if low_index < len(low_records)
                else None
            )

            if high_hash is None and low_hash is None:
                break

            if low_hash is None:
                preferred = "high"
            elif high_hash is None:
                preferred = "low"
            elif high_hash <= low_hash:
                preferred = "high"
            else:
                preferred = "low"

        candidate = None

        # Try preferred group first
        if preferred == "high":
            candidate, high_index = get_next_fitting_patient(
                high_records,
                high_index,
                remaining_capacity,
            )

            # Fall back to the other group if necessary
            if candidate is None:
                candidate, low_index = get_next_fitting_patient(
                    low_records,
                    low_index,
                    remaining_capacity,
                )

        else:
            candidate, low_index = get_next_fitting_patient(
                low_records,
                low_index,
                remaining_capacity,
            )

            if candidate is None:
                candidate, high_index = get_next_fitting_patient(
                    high_records,
                    high_index,
                    remaining_capacity,
                )

        # No complete remaining patient fits
        if candidate is None:
            break

        selected_patients.append(candidate[group_column])

        current_n += int(candidate["n_observations"])

        current_positive += int(candidate["n_positive"])

    if not selected_patients:
        raise ValueError(
            "No complete patient group fits within n_samples. Increase n_samples."
        )

    # ---------------------------------------------------------
    # Recover all observations from selected patients
    # ---------------------------------------------------------

    sampled_df = df.loc[df[group_column].isin(selected_patients)].copy()

    sampled_df = sampled_df.sort_values(id_column).reset_index(drop=True)

    return sampled_df


# -----------------------------------
# LOADING DATA
# -----------------------------------


def load_raw_metadata(
    config_path: str | Path = "configs/data_config.yaml",
):
    config = load_yaml_config(path(config_path))

    raw = config["paths"]["raw"]

    ground_truth = pd.read_csv(path(raw["ground_truth_csv"]))
    supplement = pd.read_csv(path(raw["supplement_csv"]))
    metadata = pd.read_csv(path(raw["metadata_csv"]))

    return ground_truth, supplement, metadata


def load_metadata_parquet(
    stage: str,
    filename: str,
    config_path: str | Path = "configs/data_config.yaml",
    timestamp_flag: bool = False,
    timestamp_value: str | None = None,
) -> pd.DataFrame:
    """
    Load a metadata Parquet file from the interim or processed data folder.

    Parameters
    ----------
    stage:
        Data stage from which to load the file. Must be either "interim"
        or "processed".

    filename:
        Base filename or full Parquet filename. If timestamp_flag=False,
        the function loads <stage>/metadata/<filename>. If the filename
        does not end with ".parquet", the extension is added automatically.

        If timestamp_flag=True, filename is interpreted as the base filename.
        If ".parquet" is included, it is removed before appending the
        timestamp.

    config_path:
        Path to the YAML configuration file containing data paths.

    timestamp_flag:
        Whether to load a timestamped Parquet file.

    timestamp_value:
        Optional timestamp in the format YYYYmmdd_HHMMSS. If provided,
        the function loads <base_filename>_<timestamp_value>.parquet.
        If None and timestamp_flag=True, the function searches for matching
        timestamped files and loads the one with the latest valid timestamp.

    Returns
    -------
    pd.DataFrame
        Loaded metadata dataframe.
    """
    if stage not in {"interim", "processed"}:
        raise ValueError("stage must be either 'interim' or 'processed'.")

    config = load_yaml_config(path(config_path))
    output_dir = path(config["paths"][stage]["metadata"])

    if not output_dir.exists():
        raise FileNotFoundError(f"Metadata directory not found: {output_dir}")

    base_filename = Path(filename).stem

    if not timestamp_flag:
        parquet_path = output_dir / f"{base_filename}.parquet"

        if not parquet_path.exists():
            available = sorted(p.name for p in output_dir.glob("*.parquet"))
            raise FileNotFoundError(
                f"File '{parquet_path.name}' not found in '{output_dir}'. "
                f"Available files: {available}"
            )

        return pd.read_parquet(parquet_path)

    if timestamp_value is not None:
        try:
            datetime.strptime(timestamp_value, "%Y%m%d_%H%M%S")
        except ValueError as exc:
            raise ValueError(
                "timestamp_value must be in the format YYYYmmdd_HHMMSS."
            ) from exc

        parquet_path = output_dir / f"{base_filename}_{timestamp_value}.parquet"

        if not parquet_path.exists():
            available = sorted(p.name for p in output_dir.glob("*.parquet"))
            raise FileNotFoundError(
                f"File '{parquet_path.name}' not found in '{output_dir}'. "
                f"Available files: {available}"
            )

        return pd.read_parquet(parquet_path)

    pattern = re.compile(rf"^{re.escape(base_filename)}_(\d{{8}}_\d{{6}})\.parquet$")

    candidates: list[tuple[datetime, Path]] = []

    for parquet_file in output_dir.glob(f"{base_filename}_*.parquet"):
        match = pattern.match(parquet_file.name)

        if match:
            timestamp_str = match.group(1)

            try:
                parsed_ts = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except ValueError:
                continue

            candidates.append((parsed_ts, parquet_file))

    if not candidates:
        available = sorted(p.name for p in output_dir.glob("*.parquet"))
        raise FileNotFoundError(
            f"No files found matching "
            f"'{base_filename}_<timestamp>.parquet' in '{output_dir}'. "
            f"Available files: {available}"
        )

    latest_path = max(candidates, key=lambda entry: entry[0])[1]

    return pd.read_parquet(latest_path)


def load_image_parquet(
    image_size: int,
    config_path: str | Path = "configs/data_config.yaml",
    base_filename: str = "raw_images_preprocessed",
    timestamp_value: str | None = None,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load preprocessed image Parquet file(s) from the interim images folder.

    The function supports both:
    - a single Parquet file:
        raw_images_preprocessed_<image_size>_<timestamp>.parquet

    - multiple Parquet shards:
        raw_images_preprocessed_<image_size>_<timestamp>_shard_000.parquet
        raw_images_preprocessed_<image_size>_<timestamp>_shard_001.parquet
        ...

    If timestamp_value is not provided, the latest valid timestamp is loaded.
    If a manifest JSON exists, it is used to identify the files.
    Otherwise, the function falls back to detecting files by filename pattern.
    """

    config = load_yaml_config(path(config_path))
    image_dir = path(config["paths"]["interim"]["images"])

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    base_filename = Path(base_filename).stem
    prefix = f"{base_filename}_{image_size}"

    if timestamp_value is not None:
        try:
            datetime.strptime(timestamp_value, "%Y%m%d_%H%M%S")
        except ValueError as exc:
            raise ValueError(
                "timestamp_value must be in the format YYYYmmdd_HHMMSS."
            ) from exc

        selected_timestamp = timestamp_value

    else:
        manifest_pattern = re.compile(
            rf"^{re.escape(prefix)}_(\d{{8}}_\d{{6}})_manifest\.json$"
        )
        single_file_pattern = re.compile(
            rf"^{re.escape(prefix)}_(\d{{8}}_\d{{6}})\.parquet$"
        )
        shard_pattern = re.compile(
            rf"^{re.escape(prefix)}_(\d{{8}}_\d{{6}})_shard_\d{{3}}\.parquet$"
        )

        candidates: list[tuple[datetime, str]] = []

        for file in image_dir.glob(f"{prefix}_*"):
            match = (
                manifest_pattern.match(file.name)
                or single_file_pattern.match(file.name)
                or shard_pattern.match(file.name)
            )

            if match:
                timestamp_str = match.group(1)

                try:
                    parsed_ts = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                except ValueError:
                    continue

                candidates.append((parsed_ts, timestamp_str))

        if not candidates:
            available = sorted(p.name for p in image_dir.glob("*"))
            raise FileNotFoundError(
                f"No image Parquet files found matching '{prefix}_<timestamp>' "
                f"in '{image_dir}'. Available files: {available}"
            )

        selected_timestamp = max(candidates, key=lambda entry: entry[0])[1]

    manifest_path = image_dir / f"{prefix}_{selected_timestamp}_manifest.json"

    parquet_paths: list[Path] = []

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        for file_path in manifest.get("files", []):
            parquet_path = Path(file_path)

            if parquet_path.exists():
                parquet_paths.append(parquet_path)
            else:
                fallback_path = image_dir / parquet_path.name

                if fallback_path.exists():
                    parquet_paths.append(fallback_path)

    if not parquet_paths:
        single_file_path = image_dir / f"{prefix}_{selected_timestamp}.parquet"

        if single_file_path.exists():
            parquet_paths = [single_file_path]
        else:
            parquet_paths = sorted(
                image_dir.glob(f"{prefix}_{selected_timestamp}_shard_*.parquet")
            )

    if not parquet_paths:
        available = sorted(p.name for p in image_dir.glob("*"))
        raise FileNotFoundError(
            f"No Parquet files found for '{prefix}_{selected_timestamp}'. "
            f"Available files: {available}"
        )

    if len(parquet_paths) == 1:
        return pd.read_parquet(parquet_paths[0], columns=columns)

    dataframes = [
        pd.read_parquet(parquet_path, columns=columns) for parquet_path in parquet_paths
    ]

    return pd.concat(dataframes, ignore_index=True)


def load_image_embeddings(
    model_name: str,
    image_size: int = 136,
    config_path: str | Path = "configs/data_config.yaml",
    base_filename: str = "image_embeddings",
    timestamp_value: str | None = None,
    columns: list[str] | None = None,
    lesion_ids: list[str] | set[str] | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Load CNN image-embedding Parquet shards from the processed images folder.

    The function selects one embedding-generation run using its JSON manifest.
    If timestamp_value is None, the latest valid timestamp for the selected
    model and image size is used.

    Parameters
    ----------
    model_name:
        Name of the pretrained feature extractor, for example
        "efficientnet_b0", "resnet50", "densenet121" or "convnext_tiny".

    image_size:
        Spatial image size used to generate the embeddings.

    config_path:
        Path to the YAML configuration file containing data paths.

    base_filename:
        Base filename used by the embedding-generation script.

    timestamp_value:
        Optional timestamp in YYYYmmdd_HHMMSS format. If None, the latest
        valid embedding manifest is selected automatically.

    columns:
        Optional subset of columns to return. If lesion_ids is provided,
        lesion_id is read internally even when it is not requested.

    lesion_ids:
        Optional lesion identifiers to retain. This allows downstream models
        to load only the required train/validation/test observations instead
        of loading the complete embedding dataset into memory.

    Returns
    -------
    pd.DataFrame
        Embedding dataframe assembled from all shards in the selected run.
    """

    config = load_yaml_config(path(config_path))
    image_dir = path(config["paths"]["processed"]["images"])

    if not image_dir.exists():
        raise FileNotFoundError(f"Processed image directory not found: {image_dir}")

    base_filename = Path(base_filename).stem
    prefix = f"{base_filename}_{model_name}_{image_size}"

    if timestamp_value is not None:
        try:
            datetime.strptime(timestamp_value, "%Y%m%d_%H%M%S")
        except ValueError as exc:
            raise ValueError(
                "timestamp_value must be in the format YYYYmmdd_HHMMSS."
            ) from exc

        manifest_path = image_dir / f"{prefix}_{timestamp_value}_manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(f"Embedding manifest not found: {manifest_path}")

    else:
        manifest_pattern = re.compile(
            rf"^{re.escape(prefix)}_(\d{{8}}_\d{{6}})_manifest\.json$"
        )

        candidates: list[tuple[datetime, Path]] = []

        for candidate in image_dir.glob(f"{prefix}_*_manifest.json"):
            match = manifest_pattern.match(candidate.name)

            if not match:
                continue

            try:
                parsed_timestamp = datetime.strptime(
                    match.group(1),
                    "%Y%m%d_%H%M%S",
                )
            except ValueError:
                continue

            candidates.append((parsed_timestamp, candidate))

        if not candidates:
            available = sorted(p.name for p in image_dir.glob("*_manifest.json"))
            raise FileNotFoundError(
                f"No embedding manifests found matching "
                f"'{prefix}_<timestamp>_manifest.json' in '{image_dir}'. "
                f"Available manifests: {available}"
            )

        manifest_path = max(
            candidates,
            key=lambda item: item[0],
        )[1]

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("model") != model_name:
        raise ValueError(
            f"Manifest model mismatch: expected '{model_name}', "
            f"found '{manifest.get('model')}'."
        )

    if int(manifest.get("image_size", -1)) != image_size:
        raise ValueError(
            f"Manifest image-size mismatch: expected {image_size}, "
            f"found {manifest.get('image_size')}."
        )

    parquet_paths: list[Path] = []

    for stored_path in manifest.get("files", []):
        candidate = Path(stored_path)

        if candidate.exists():
            parquet_paths.append(candidate)
            continue

        fallback = image_dir / candidate.name

        if fallback.exists():
            parquet_paths.append(fallback)
            continue

        raise FileNotFoundError(
            f"Embedding shard listed in manifest was not found: {stored_path}"
        )

    if not parquet_paths:
        raise ValueError(
            f"Embedding manifest contains no valid Parquet files: {manifest_path}"
        )

    requested_ids: set[str] | None = None

    if lesion_ids is not None:
        requested_ids = {str(value) for value in lesion_ids}

        if not requested_ids:
            raise ValueError("lesion_ids is empty.")

    read_columns = columns

    if columns is not None and "lesion_id" not in columns:
        read_columns = ["lesion_id", *columns]

    dataframes: list[pd.DataFrame] = []

    for parquet_path in parquet_paths:
        shard_df = pd.read_parquet(
            parquet_path,
            columns=read_columns,
        )

        if "lesion_id" not in shard_df.columns:
            raise KeyError(f"Expected 'lesion_id' in embedding shard: {parquet_path}")

        shard_df["lesion_id"] = shard_df["lesion_id"].astype(str)

        if requested_ids is not None:
            shard_df = shard_df.loc[shard_df["lesion_id"].isin(requested_ids)].copy()

        if not shard_df.empty:
            dataframes.append(shard_df)

    if not dataframes:
        raise ValueError("No embedding rows matched the requested data.")

    embeddings = pd.concat(
        dataframes,
        ignore_index=True,
    )

    if embeddings["lesion_id"].duplicated().any():
        duplicates = (
            embeddings.loc[
                embeddings["lesion_id"].duplicated(),
                "lesion_id",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Duplicated lesion_id values found in embeddings. Examples: {duplicates}"
        )

    if requested_ids is not None:
        loaded_ids = set(embeddings["lesion_id"])

        missing_ids = requested_ids.difference(loaded_ids)

        if missing_ids:
            examples = sorted(missing_ids)[:10]

            raise ValueError(
                f"{len(missing_ids):,} requested lesion_ids were not found "
                f"in the selected embedding run. Examples: {examples}"
            )

    elif len(embeddings) != int(manifest.get("n_rows", len(embeddings))):
        raise ValueError(
            f"Embedding row count does not match the manifest: "
            f"loaded {len(embeddings):,}, expected "
            f"{int(manifest['n_rows']):,}."
        )

    if columns is not None:
        embeddings = embeddings.loc[:, columns]

    return embeddings


def load_image_pca(
    hypothesis: int,
    split: str,
    image_size: int = 136,
    n_components: int = 128,
    config_path: str | Path = "configs/data_config.yaml",
    timestamp_value: str | None = None,
    columns: list[str] | None = None,
    lesion_ids: list[str] | set[str] | pd.Series | None = None,
) -> pd.DataFrame:
    """
    Load PCA image features from the processed images folder.

    Parameters
    ----------
    hypothesis:
        Modelling hypothesis. Must be 1 or 2.

    split:
        Dataset split to load. Must be "train", "val" or "test".

    image_size:
        Image size used before PCA. Default: 136.

    n_components:
        Number of PCA components. Default: 128.

    config_path:
        Path to the YAML configuration file.

    timestamp_value:
        Optional exact PCA generation timestamp in YYYYmmdd_HHMMSS format.
        If omitted, the latest matching PCA manifest is used.

    columns:
        Optional subset of columns to return. If lesion_ids is provided,
        lesion_id is read internally even when it is not requested.

    lesion_ids:
        Optional lesion identifiers to retain. This is useful when the
        training metadata have been reproducibly subsampled, because only
        the selected PCA rows are retained while reading the shards.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the requested PCA rows and columns.
    """

    if hypothesis not in {1, 2}:
        raise ValueError("hypothesis must be either 1 or 2.")

    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val' or 'test'.")

    config = load_yaml_config(path(config_path))
    image_dir = path(config["paths"]["processed"]["images"])

    if not image_dir.exists():
        raise FileNotFoundError(f"Processed image directory not found: {image_dir}")

    prefix = f"image_pca_h{hypothesis}_{image_size}px_{n_components}c"

    if timestamp_value is not None:
        try:
            datetime.strptime(timestamp_value, "%Y%m%d_%H%M%S")
        except ValueError as exc:
            raise ValueError(
                "timestamp_value must be in the format YYYYmmdd_HHMMSS."
            ) from exc

        manifest_path = image_dir / (f"{prefix}_{timestamp_value}_manifest.json")

        if not manifest_path.exists():
            raise FileNotFoundError(f"PCA manifest not found: {manifest_path}")

    else:
        pattern = re.compile(
            rf"^{re.escape(prefix)}_"
            rf"(\d{{8}}_\d{{6}})_manifest\.json$"
        )

        candidates: list[tuple[datetime, Path]] = []

        for manifest_file in image_dir.glob(f"{prefix}_*_manifest.json"):
            match = pattern.match(manifest_file.name)

            if match:
                timestamp_str = match.group(1)

                try:
                    parsed_ts = datetime.strptime(
                        timestamp_str,
                        "%Y%m%d_%H%M%S",
                    )
                except ValueError:
                    continue

                candidates.append((parsed_ts, manifest_file))

        if not candidates:
            raise FileNotFoundError(
                f"No PCA manifest found matching '{prefix}' in '{image_dir}'."
            )

        manifest_path = max(
            candidates,
            key=lambda entry: entry[0],
        )[1]

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    files_by_split = manifest.get("files_by_split", {})

    if split not in files_by_split:
        raise ValueError(
            f"Split '{split}' not found in PCA manifest: {manifest_path.name}"
        )

    parquet_paths: list[Path] = []

    for file_path in files_by_split[split]:
        parquet_path = Path(file_path)

        if not parquet_path.exists():
            parquet_path = image_dir / parquet_path.name

        if not parquet_path.exists():
            raise FileNotFoundError(f"PCA Parquet file not found: {file_path}")

        parquet_paths.append(parquet_path)

    if not parquet_paths:
        raise FileNotFoundError(f"No PCA Parquet files found for split '{split}'.")

    requested_ids: set[str] | None = None

    if lesion_ids is not None:
        requested_ids = {str(value) for value in lesion_ids}

        if not requested_ids:
            raise ValueError("lesion_ids is empty.")

    read_columns = columns

    if columns is not None and "lesion_id" not in columns:
        read_columns = ["lesion_id", *columns]

    dataframes: list[pd.DataFrame] = []

    for parquet_path in parquet_paths:
        shard_df = pd.read_parquet(
            parquet_path,
            columns=read_columns,
        )

        if "lesion_id" not in shard_df.columns:
            raise KeyError(f"Expected 'lesion_id' in PCA shard: {parquet_path}")

        shard_df["lesion_id"] = shard_df["lesion_id"].astype(str)

        if requested_ids is not None:
            shard_df = shard_df.loc[shard_df["lesion_id"].isin(requested_ids)].copy()

        if not shard_df.empty:
            dataframes.append(shard_df)

    if not dataframes:
        raise ValueError("No PCA rows matched the requested data.")

    pca_features = pd.concat(
        dataframes,
        ignore_index=True,
    )

    if pca_features["lesion_id"].duplicated().any():
        duplicates = (
            pca_features.loc[
                pca_features["lesion_id"].duplicated(),
                "lesion_id",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(
            f"Duplicated lesion_id values found in PCA features. Examples: {duplicates}"
        )

    if requested_ids is not None:
        loaded_ids = set(pca_features["lesion_id"])
        missing_ids = requested_ids.difference(loaded_ids)

        if missing_ids:
            examples = sorted(missing_ids)[:10]

            raise ValueError(
                f"{len(missing_ids):,} requested lesion_ids were not found "
                f"in the selected PCA run. Examples: {examples}"
            )

    expected_rows = manifest.get("n_rows_by_split", {}).get(split)

    if requested_ids is None and expected_rows is not None:
        if len(pca_features) != int(expected_rows):
            raise ValueError(
                f"PCA row count does not match the manifest for '{split}': "
                f"loaded {len(pca_features):,}, expected "
                f"{int(expected_rows):,}."
            )

    if columns is not None:
        pca_features = pca_features.loc[:, columns]

    return pca_features


# -----------------------------------
# PERSISTING DATA TO PARQUET
# -----------------------------------


def save_metadata_parquet(
    df: pd.DataFrame,
    stage: str,
    config_path: str | Path = "configs/data_config.yaml",
    name: str | None = None,
    timestamp: bool = True,
) -> Path:
    """
    Save a DataFrame as a parquet file inside the metadata folder
    of either interim or processed data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    stage : str
        Either 'interim' or 'processed'.
    config_path : str | Path
        Path to the YAML config file.
    name : str | None
        Optional base name for the output file. If provided, the
        filename will use this name.
    timestamp : bool
        If True and a name is provided, append a timestamp to the
        filename. If name is not provided, the default filename
        convention is used and timestamp is always included.

    Returns
    -------
    Path
        Path where the parquet file has been saved.
    """
    if stage not in {"interim", "processed"}:
        raise ValueError("stage must be either 'interim' or 'processed'.")

    config = load_yaml_config(path(config_path))

    output_dir = path(config["paths"][stage]["metadata"])
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    if name:
        base_name = Path(name).stem
        if timestamp:
            filename = f"{base_name}_{timestamp_str}.parquet"
        else:
            filename = f"{base_name}.parquet"
    else:
        if stage == "interim":
            filename = f"df_interim_{timestamp_str}.parquet"
        else:
            filename = f"df_preprocessed_{timestamp_str}.parquet"

    output_path = output_dir / filename

    df.to_parquet(output_path, index=False)

    return output_path
