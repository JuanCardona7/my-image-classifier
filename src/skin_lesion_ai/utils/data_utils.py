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
