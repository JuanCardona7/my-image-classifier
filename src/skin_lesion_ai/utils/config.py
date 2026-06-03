from pathlib import Path
from datetime import datetime

import pandas as pd
import yaml

# CONFIG PATH AND LOADING


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


# LOADING DATA


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
) -> pd.DataFrame:
    """
    Load a parquet file from the metadata folder of either
    interim or processed data.
    """
    if stage not in {"interim", "processed"}:
        raise ValueError("stage must be either 'interim' or 'processed'.")

    config = load_yaml_config(path(config_path))

    parquet_path = path(config["paths"][stage]["metadata"]) / filename

    if not parquet_path.exists():
        available = sorted(p.name for p in parquet_path.parent.glob("*.parquet"))

    raise FileNotFoundError(
        f"File '{filename}' not found in "
        f"'{parquet_path.parent}'. "
        f"Available files: {available}"
    )

    return pd.read_parquet(parquet_path)


# PERSISTING DATA TO PARQUET


def save_metadata_parquet(
    df: pd.DataFrame,
    stage: str,
    config_path: str | Path = "configs/data_config.yaml",
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if stage == "interim":
        filename = f"df_interim_{timestamp}.parquet"
    else:
        filename = f"df_preprocessed_{timestamp}.parquet"

    output_path = output_dir / filename

    df.to_parquet(output_path, index=False)

    return output_path
