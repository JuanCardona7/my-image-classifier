"""Utilities for final model selection, multimodal fusion and test evaluation.

This module is designed to be used after the base models have already been
trained and evaluated on train/validation data.

Key principles
--------------
1. Base models are NEVER retrained.
2. Validation/test prediction Parquets are loaded from cache when available.
3. If the Parquet is missing, inference is rerun from the saved model only.
4. Multimodal fusion is selected exclusively with patient-grouped CV on the
   validation split.
5. Fusion selection hierarchy:
      PR-AUC (higher) -> ROC-AUC (higher) -> Brier score (lower).
6. The selected fusion method is refitted on the complete validation split.
7. Its clinical threshold is selected on validation and then frozen.
8. The test split is used only for the final evaluation.

Supported automatic inference
-----------------------------
- scikit-learn/joblib metadata models, when fitted feature names are recoverable;
- XGBoost / scikit-learn models on CNN embeddings;
- XGBoost / scikit-learn models on image PCA features;
- direct PyTorch TorchVision image classifiers when enough architecture
  information is available in model_metadata.json, inference_config.json,
  training_configuration.json, or notebook overrides.

Legacy CSV prediction files are intentionally ignored: only the canonical
Parquet cache is reused. For old models whose preprocessing cannot be
reconstructed safely, the module raises an explicit error instead of guessing.
Optional MODEL_OVERRIDES can provide missing inference information.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from skin_lesion_ai.inference import evaluation as ev
from skin_lesion_ai.utils.data_utils import (
    get_project_root,
    load_image_embeddings,
    load_image_pca,
    load_yaml_config,
)


DEFAULT_CONFIG_PATH = "configs/data_config.yaml"
VALIDATION_PREDICTIONS_FILENAME = "validation_predictions.parquet"
TEST_PREDICTIONS_FILENAME = "test_predictions.parquet"

BASE_CANDIDATES = (
    "metadata_only",
    "image_only",
)

FUSION_METHODS = (
    "simple_average",
    "weighted_average",
    "logistic_stacking",
    "interaction_stacking",
)

ALL_CANDIDATES = BASE_CANDIDATES + FUSION_METHODS

IMAGE_EMBEDDING_MODELS = (
    "efficientnet_b0",
    "resnet50",
    "densenet121",
    "convnext_tiny",
)

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


# ---------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _project_path(value: str | Path) -> Path:
    value = Path(value)
    if value.is_absolute():
        return value
    return get_project_root() / value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        try:
            return str(value.relative_to(get_project_root()))
        except ValueError:
            return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _write_json(data: dict, path: Path) -> Path:
    with path.open("w", encoding="utf-8") as file:
        json.dump(_jsonable(data), file, indent=2, ensure_ascii=False)
    return path


def _read_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in: {path}")
    return data


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name).strip()).strip("._-")
    if not safe:
        raise ValueError("A non-empty model name is required.")
    return safe


def _load_model_metadata(model_directory: str | Path) -> tuple[Path, dict]:
    model_directory = _project_path(model_directory)
    metadata_path = model_directory / "model_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"model_metadata.json not found in: {model_directory}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if "model" not in metadata or "model_name" not in metadata["model"]:
        raise ValueError(f"Invalid model_metadata.json in: {model_directory}")

    return metadata_path, metadata


def infer_hypothesis(model_directory: str | Path) -> int:
    """Infer H1/H2 from model metadata, falling back to the model name."""
    _, metadata = _load_model_metadata(model_directory)

    hypothesis = (metadata.get("evaluation") or {}).get("hypothesis")
    if hypothesis in (1, 2):
        return int(hypothesis)

    model_name = metadata["model"]["model_name"].lower()
    match = re.search(r"(?:^|_)h([12])(?:_|$)", model_name)
    if match:
        return int(match.group(1))

    raise ValueError(
        f"Could not infer hypothesis for '{model_name}'. "
        "The model must have evaluation.hypothesis or contain '_h1_'/'_h2_'."
    )


def _load_split(
    hypothesis: int,
    split: str,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
) -> pd.DataFrame:
    """Load the latest timestamped validation or test split."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")

    config_path = _project_path(config_path)
    config = load_yaml_config(config_path)

    try:
        metadata_dir = _project_path(config["paths"]["processed"]["metadata"])
    except KeyError as exc:
        raise KeyError(
            "configs/data_config.yaml must contain paths.processed.metadata."
        ) from exc

    split_prefix = "val" if split == "validation" else "test"
    base_name = f"{split_prefix}_split_h{hypothesis}"

    pattern = re.compile(rf"^{re.escape(base_name)}_(\d{{8}}_\d{{6}})\.parquet$")

    candidates: list[tuple[str, Path]] = []
    for path in metadata_dir.glob(f"{base_name}_*.parquet"):
        match = pattern.match(path.name)
        if match:
            candidates.append((match.group(1), path))

    # Allow a non-timestamped split as a conservative fallback.
    plain_path = metadata_dir / f"{base_name}.parquet"

    if candidates:
        split_path = max(candidates, key=lambda item: item[0])[1]
    elif plain_path.exists():
        split_path = plain_path
    else:
        available = sorted(p.name for p in metadata_dir.glob("*split_h*.parquet"))
        raise FileNotFoundError(
            f"No {split} split found for H{hypothesis} in {metadata_dir}. "
            f"Available split files: {available}"
        )

    df = pd.read_parquet(split_path)

    required = {
        "isic_id",
        "patient_id",
        ev.DEFAULT_LABEL_COLUMNS[hypothesis],
    }
    missing = required.difference(df.columns)

    if missing:
        raise KeyError(
            f"{split_path.name} is missing required columns: {sorted(missing)}"
        )

    if df["isic_id"].duplicated().any():
        raise ValueError(f"{split_path.name} contains duplicated isic_id values.")

    df = df.copy()
    df["isic_id"] = df["isic_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    df.attrs["source_path"] = str(split_path)

    return df


def _canonicalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Return prediction data with exactly isic_id + probability."""
    aliases = {
        "lesion_id": "isic_id",
        "y_prob": "probability",
    }

    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})

    required = {"isic_id", "probability"}
    missing = required.difference(df.columns)

    if missing:
        raise KeyError(f"Prediction dataframe is missing: {sorted(missing)}")

    out = df.loc[:, ["isic_id", "probability"]].copy()
    out["isic_id"] = out["isic_id"].astype(str)
    out["probability"] = pd.to_numeric(out["probability"], errors="raise")

    if out["isic_id"].duplicated().any():
        raise ValueError("Prediction dataframe contains duplicated isic_id values.")

    if out["probability"].isna().any():
        raise ValueError("Prediction dataframe contains missing probabilities.")

    if not np.isfinite(out["probability"]).all():
        raise ValueError("Prediction dataframe contains non-finite probabilities.")

    if not out["probability"].between(0.0, 1.0).all():
        raise ValueError("Predicted probabilities must lie in [0, 1].")

    return out


def _validate_prediction_coverage(
    predictions: pd.DataFrame,
    split_df: pd.DataFrame,
    split_name: str,
) -> None:
    prediction_ids = set(predictions["isic_id"].astype(str))
    split_ids = set(split_df["isic_id"].astype(str))

    missing = split_ids.difference(prediction_ids)
    extra = prediction_ids.difference(split_ids)

    if missing or extra:
        raise ValueError(
            f"{split_name} prediction IDs do not match the split. "
            f"Missing predictions: {len(missing)}; extra predictions: {len(extra)}."
        )


# ---------------------------------------------------------------------
# Inference configuration and model-type detection
# ---------------------------------------------------------------------


def _merged_inference_config(
    model_directory: Path,
    model_metadata: dict,
    override: dict | None,
) -> dict:
    """Merge inference information from all available model artefacts.

    Precedence, from lowest to highest:
    1. model_metadata.json -> inference
    2. training_configuration.json
    3. inference_config.json
    4. notebook MODEL_OVERRIDES

    The training configuration parser understands the structure used by the
    direct-image EfficientNet notebook, including nested training and
    augmentation sections.
    """
    config: dict[str, Any] = {}

    metadata_inference = model_metadata.get("inference")
    if isinstance(metadata_inference, dict):
        config.update(metadata_inference)

    training = _read_json_if_exists(model_directory / "training_configuration.json")

    if training:
        # Top-level information saved by the image-model notebook.
        top_level_keys = (
            "architecture",
            "image_size",
            "image_manifest",
            "image_timestamp_setting",
            "weights",
            "target",
            "hypothesis",
        )

        for key in top_level_keys:
            if key in training and key not in config:
                config[key] = training[key]

        # Training information relevant for model reconstruction.
        training_section = training.get("training", {})

        if isinstance(training_section, dict):
            if "dropout" in training_section and "dropout" not in config:
                config["dropout"] = training_section["dropout"]

            if "batch_size" in training_section:
                config.setdefault(
                    "training_batch_size",
                    training_section["batch_size"],
                )

            if "num_workers" in training_section:
                config.setdefault(
                    "training_num_workers",
                    training_section["num_workers"],
                )

            if "mixed_precision" in training_section:
                config.setdefault(
                    "training_mixed_precision",
                    training_section["mixed_precision"],
                )

        # Deterministic evaluation normalization.
        augmentation = training.get("augmentation", {})

        if isinstance(augmentation, dict):
            if "imagenet_mean" in augmentation:
                config.setdefault(
                    "imagenet_mean",
                    augmentation["imagenet_mean"],
                )

            if "imagenet_std" in augmentation:
                config.setdefault(
                    "imagenet_std",
                    augmentation["imagenet_std"],
                )

    # Explicit sidecar inference configuration overrides training metadata.
    sidecar = _read_json_if_exists(model_directory / "inference_config.json")
    config.update(sidecar)

    # Notebook override has highest priority.
    if override:
        config.update(override)

    return config


def detect_input_kind(
    model_directory: str | Path,
    override: dict | None = None,
) -> str:
    """Detect metadata, image_embedding, image_pca or direct_image."""
    model_directory = _project_path(model_directory)
    _, metadata = _load_model_metadata(model_directory)
    config = _merged_inference_config(model_directory, metadata, override)

    explicit = config.get("input_kind")
    if explicit is not None:
        allowed = {"metadata", "image_embedding", "image_pca", "direct_image"}
        if explicit not in allowed:
            raise ValueError(
                f"Unsupported input_kind '{explicit}'. Choose one of {sorted(allowed)}."
            )
        return explicit

    name = metadata["model"]["model_name"].lower()
    framework = str(metadata["model"].get("framework", "")).lower()

    if "image_embedding" in name or "_embedding_" in name:
        return "image_embedding"

    if "image_pca" in name or "_pca_" in name:
        return "image_pca"

    if framework in {"pytorch", "keras", "tensorflow"}:
        return "direct_image"

    if "_image_" in name and framework != "scikit-learn":
        return "direct_image"

    return "metadata"


def _extract_embedding_architecture(
    model_name: str,
    inference_config: dict,
) -> str:
    explicit = inference_config.get("embedding_model")
    if explicit:
        return str(explicit)

    lower = model_name.lower()
    for candidate in IMAGE_EMBEDDING_MODELS:
        if candidate in lower:
            return candidate

    raise ValueError(
        f"Could not infer the CNN embedding extractor from '{model_name}'. "
        "Set MODEL_OVERRIDES[model_dir]['embedding_model']."
    )


def _model_feature_names(model: Any) -> list[str] | None:
    names = getattr(model, "feature_names_in_", None)

    if names is not None:
        return [str(name) for name in names]

    if hasattr(model, "get_booster"):
        try:
            booster_names = model.get_booster().feature_names
        except Exception:
            booster_names = None

        if booster_names:
            # Ignore generic f0/f1/... names because they do not encode source columns.
            generic = all(re.fullmatch(r"f\d+", str(name)) for name in booster_names)
            if not generic:
                return [str(name) for name in booster_names]

    return None


def _model_n_features(model: Any) -> int | None:
    value = getattr(model, "n_features_in_", None)
    if value is not None:
        return int(value)

    if hasattr(model, "get_booster"):
        try:
            return int(model.get_booster().num_features())
        except Exception:
            return None

    return None


# ---------------------------------------------------------------------
# scikit-learn / XGBoost inference
# ---------------------------------------------------------------------


def _prepare_sklearn_features(
    model: Any,
    model_directory: Path,
    model_metadata: dict,
    split_df: pd.DataFrame,
    split: str,
    hypothesis: int,
    inference_config: dict,
    config_path: str | Path,
) -> pd.DataFrame:
    input_kind = detect_input_kind(
        model_directory,
        override=inference_config,
    )

    feature_names = inference_config.get("feature_columns")
    if feature_names is not None:
        feature_names = [str(col) for col in feature_names]
    else:
        feature_names = _model_feature_names(model)

    n_features = _model_n_features(model)
    lesion_ids = split_df["isic_id"].astype(str)

    if input_kind == "metadata":
        if feature_names is None:
            raise ValueError(
                f"Cannot safely reconstruct metadata features for "
                f"'{model_metadata['model']['model_name']}'. "
                "The saved model does not expose fitted feature names. "
                "Add feature_columns to inference_config.json or MODEL_OVERRIDES."
            )

        missing = set(feature_names).difference(split_df.columns)
        if missing:
            raise KeyError(
                f"Metadata split is missing model features: {sorted(missing)}"
            )

        X = split_df.loc[:, feature_names].copy()

    elif input_kind == "image_embedding":
        architecture = _extract_embedding_architecture(
            model_metadata["model"]["model_name"],
            inference_config,
        )

        image_size = int(inference_config.get("image_size", 136))
        embedding_timestamp = inference_config.get("embedding_timestamp")

        embeddings = load_image_embeddings(
            model_name=architecture,
            image_size=image_size,
            config_path=config_path,
            timestamp_value=embedding_timestamp,
            lesion_ids=lesion_ids,
        )

        embeddings = embeddings.copy()
        embeddings["lesion_id"] = embeddings["lesion_id"].astype(str)

        if feature_names is None:
            feature_names = sorted(
                col for col in embeddings.columns if col.startswith("feature_")
            )

        missing = set(feature_names).difference(embeddings.columns)
        if missing:
            raise KeyError(
                f"Embedding data are missing model features: {sorted(missing)}"
            )

        X = (
            split_df.loc[:, ["isic_id"]]
            .rename(columns={"isic_id": "lesion_id"})
            .merge(
                embeddings.loc[:, ["lesion_id", *feature_names]],
                on="lesion_id",
                how="left",
                validate="one_to_one",
            )
            .loc[:, feature_names]
        )

    elif input_kind == "image_pca":
        image_size = int(inference_config.get("image_size", 136))

        n_components = inference_config.get("n_components")
        if n_components is None:
            n_components = n_features

        if n_components is None:
            raise ValueError(
                "Could not infer PCA component count. Set n_components explicitly."
            )

        pca_split = "val" if split == "validation" else "test"

        pca_df = load_image_pca(
            hypothesis=hypothesis,
            split=pca_split,
            image_size=image_size,
            n_components=int(n_components),
            config_path=config_path,
            timestamp_value=inference_config.get("pca_timestamp"),
            lesion_ids=lesion_ids,
        )

        pca_df = pca_df.copy()
        pca_df["lesion_id"] = pca_df["lesion_id"].astype(str)

        if feature_names is None:
            feature_names = sorted(
                col for col in pca_df.columns if col.startswith("feature_")
            )

        missing = set(feature_names).difference(pca_df.columns)
        if missing:
            raise KeyError(f"PCA data are missing model features: {sorted(missing)}")

        X = (
            split_df.loc[:, ["isic_id"]]
            .rename(columns={"isic_id": "lesion_id"})
            .merge(
                pca_df.loc[:, ["lesion_id", *feature_names]],
                on="lesion_id",
                how="left",
                validate="one_to_one",
            )
            .loc[:, feature_names]
        )

    else:
        raise ValueError(
            f"scikit-learn automatic inference does not support input_kind "
            f"'{input_kind}'."
        )

    if X.isna().any().any():
        missing_count = int(X.isna().any(axis=1).sum())
        raise ValueError(
            f"Prepared inference matrix contains missing values in "
            f"{missing_count:,} rows."
        )

    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(
            f"Feature-count mismatch: model expects {n_features}, "
            f"but inference prepared {X.shape[1]}."
        )

    return X


def _positive_probability_from_sklearn(model: Any, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"{model.__class__.__name__} does not expose predict_proba(). "
            "Because the original evaluation used probabilities, the final "
            "pipeline will not invent a conversion from decision_function()."
        )

    proba = np.asarray(model.predict_proba(X))

    if proba.ndim == 1:
        positive = proba
    elif proba.ndim == 2 and proba.shape[1] == 2:
        classes = getattr(model, "classes_", None)
        if classes is not None:
            classes = list(classes)
            if 1 in classes:
                positive = proba[:, classes.index(1)]
            else:
                positive = proba[:, -1]
        else:
            positive = proba[:, 1]
    elif proba.ndim == 2 and proba.shape[1] == 1:
        positive = proba[:, 0]
    else:
        raise ValueError(f"Unexpected predict_proba output shape: {proba.shape}")

    positive = positive.astype(float)

    if not np.isfinite(positive).all() or np.any((positive < 0) | (positive > 1)):
        raise ValueError("Model returned invalid positive-class probabilities.")

    return positive


# ---------------------------------------------------------------------
# Direct image inference
# ---------------------------------------------------------------------


def _select_torch_device(requested: str):
    """Select CUDA, Apple MPS or CPU without retraining anything."""
    import torch

    if requested != "auto":
        device = torch.device(requested)

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")

        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")

        return device

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def _resolve_image_manifest(
    image_size: int,
    config_path: str | Path,
    inference_config: dict,
) -> tuple[Path, list[Path], dict]:
    """Resolve the exact preprocessing manifest whenever possible.

    Preference:
    1. exact image_manifest saved with model training;
    2. explicit image_timestamp / image_timestamp_setting;
    3. latest compatible preprocessing manifest.
    """
    from datetime import datetime

    config = load_yaml_config(_project_path(config_path))
    image_dir = _project_path(config["paths"]["interim"]["images"])

    if not image_dir.exists():
        raise FileNotFoundError(f"Interim image directory not found: {image_dir}")

    prefix = f"raw_images_preprocessed_{image_size}"

    manifest_path: Path | None = None

    # --------------------------------------------------------------
    # 1. Exact training manifest
    # --------------------------------------------------------------
    saved_manifest = inference_config.get("image_manifest")

    if saved_manifest:
        candidate = Path(str(saved_manifest))

        if candidate.exists():
            manifest_path = candidate
        else:
            # Absolute paths saved on another machine are common.
            fallback = image_dir / candidate.name

            if fallback.exists():
                manifest_path = fallback

    # --------------------------------------------------------------
    # 2. Explicit timestamp
    # --------------------------------------------------------------
    timestamp_value = inference_config.get("image_timestamp")

    if timestamp_value is None:
        timestamp_value = inference_config.get("image_timestamp_setting")

    if manifest_path is None and timestamp_value:
        try:
            datetime.strptime(
                str(timestamp_value),
                "%Y%m%d_%H%M%S",
            )
        except ValueError as exc:
            raise ValueError("Image timestamp must use YYYYmmdd_HHMMSS.") from exc

        candidate = image_dir / f"{prefix}_{timestamp_value}_manifest.json"

        if not candidate.exists():
            raise FileNotFoundError(
                f"Preprocessed-image manifest not found: {candidate}"
            )

        manifest_path = candidate

    # --------------------------------------------------------------
    # 3. Latest compatible manifest
    # --------------------------------------------------------------
    if manifest_path is None:
        pattern = re.compile(
            rf"^{re.escape(prefix)}_"
            r"(\d{8}_\d{6})_manifest\.json$"
        )

        candidates: list[tuple[str, Path]] = []

        for candidate in image_dir.glob(f"{prefix}_*_manifest.json"):
            match = pattern.match(candidate.name)

            if match:
                candidates.append((match.group(1), candidate))

        if not candidates:
            raise FileNotFoundError(
                f"No preprocessed-image manifest found for "
                f"{image_size}x{image_size} in {image_dir}."
            )

        manifest_path = max(
            candidates,
            key=lambda item: item[0],
        )[1]

    manifest = _read_json_if_exists(manifest_path)

    required_keys = {
        "image_size",
        "files",
    }
    missing_keys = required_keys.difference(manifest)

    if missing_keys:
        raise ValueError(f"Image manifest is missing keys: {sorted(missing_keys)}")

    if int(manifest["image_size"]) != int(image_size):
        raise ValueError(
            f"Manifest image size ({manifest['image_size']}) "
            f"does not match expected size ({image_size})."
        )

    parquet_paths: list[Path] = []

    for raw_path in manifest["files"]:
        candidate = Path(raw_path)

        if candidate.exists():
            parquet_paths.append(candidate)
            continue

        fallback = image_dir / candidate.name

        if fallback.exists():
            parquet_paths.append(fallback)
            continue

        raise FileNotFoundError(
            f"Image Parquet listed in the manifest was not found: {raw_path}"
        )

    if not parquet_paths:
        raise ValueError(f"No image Parquet files listed in {manifest_path.name}.")

    return (
        manifest_path,
        parquet_paths,
        manifest,
    )


def _load_selected_image_bytes(
    parquet_paths: list[Path],
    lesion_ids: Iterable[str],
    io_batch_size: int,
) -> pd.DataFrame:
    """Load only requested JPEG bytes from the image Parquet collection.

    This follows the original EfficientNet notebook strategy, but avoids
    converting non-selected image bytes to Python objects whenever possible.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm

    requested_ids = {str(lesion_id) for lesion_id in lesion_ids}

    if not requested_ids:
        raise ValueError("No lesion identifiers were requested.")

    remaining_ids = set(requested_ids)
    chunks: list[pd.DataFrame] = []

    for parquet_path in tqdm(
        parquet_paths,
        desc="Scanning image shards",
        unit="shard",
    ):
        parquet_file = pq.ParquetFile(parquet_path)
        available_columns = set(parquet_file.schema.names)

        if "lesion_id" in available_columns:
            id_col = "lesion_id"
        elif "isic_id" in available_columns:
            id_col = "isic_id"
        else:
            raise KeyError(
                f"{parquet_path.name} contains neither 'lesion_id' nor 'isic_id'."
            )

        required_columns = {
            id_col,
            "image",
        }
        missing_columns = required_columns - available_columns

        if missing_columns:
            raise KeyError(
                f"{parquet_path.name} is missing columns: {sorted(missing_columns)}"
            )

        for record_batch in parquet_file.iter_batches(
            batch_size=io_batch_size,
            columns=[id_col, "image"],
        ):
            if not remaining_ids:
                break

            id_array = record_batch.column(record_batch.schema.get_field_index(id_col))

            # Compare IDs inside Arrow before converting JPEG bytes to
            # Python objects. This substantially reduces pandas/Python
            # overhead when only a subset of lesions is requested.
            id_as_string = pc.cast(
                id_array,
                pa.string(),
            )

            value_set = pa.array(
                list(remaining_ids),
                type=pa.string(),
            )

            mask = pc.is_in(
                id_as_string,
                value_set=value_set,
            )

            if pc.sum(mask).as_py() == 0:
                continue

            selected_batch = record_batch.filter(mask)
            selected = selected_batch.to_pandas()

            selected[id_col] = selected[id_col].astype(str)

            selected = selected.rename(
                columns={
                    id_col: "isic_id",
                }
            )

            selected = selected[["isic_id", "image"]].copy()

            selected_ids = set(selected["isic_id"])

            chunks.append(selected)
            remaining_ids.difference_update(selected_ids)

        if not remaining_ids:
            break

    if not chunks:
        raise ValueError(
            "None of the requested lesions were found in the image Parquet files."
        )

    image_df = pd.concat(
        chunks,
        ignore_index=True,
    )

    if image_df["isic_id"].duplicated().any():
        duplicated = (
            image_df.loc[
                image_df["isic_id"].duplicated(),
                "isic_id",
            ]
            .head(10)
            .tolist()
        )

        raise ValueError(f"Duplicated lesion images were found. Examples: {duplicated}")

    loaded_ids = set(image_df["isic_id"])
    missing_ids = requested_ids - loaded_ids

    if missing_ids:
        raise ValueError(
            f"{len(missing_ids):,} requested lesions were "
            "not found in the image Parquets. "
            f"Examples: {sorted(missing_ids)[:10]}"
        )

    return image_df


class _InferenceImageDataset:
    """Fast Dataset for deterministic inference from JPEG bytes.

    The class deliberately stores lists instead of performing dataframe.iloc
    on every sample, which removes substantial pandas overhead.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_size: int,
        transform,
    ):
        self.image_bytes = dataframe["image"].tolist()

        self.lesion_ids = dataframe["isic_id"].astype(str).tolist()

        self.image_size = int(image_size)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.lesion_ids)

    def __getitem__(self, index: int):
        image_bytes = self.image_bytes[index]
        lesion_id = self.lesion_ids[index]

        try:
            with Image.open(BytesIO(bytes(image_bytes))) as image:
                image = image.convert("RGB")

                if image.size != (
                    self.image_size,
                    self.image_size,
                ):
                    raise ValueError(
                        f"Lesion {lesion_id}: expected "
                        f"{self.image_size}x{self.image_size}, "
                        f"got {image.size}."
                    )

                tensor = self.transform(image)

        except UnidentifiedImageError as exc:
            raise UnidentifiedImageError(
                f"Could not decode lesion {lesion_id}."
            ) from exc

        return (
            tensor,
            lesion_id,
        )


def _make_evaluation_transform(
    inference_config: dict,
):
    """Build the deterministic transform used for image-model inference."""
    from torchvision import transforms

    normalization = str(
        inference_config.get(
            "normalization",
            "imagenet",
        )
    ).lower()

    transform_steps = [
        transforms.ToTensor(),
    ]

    if normalization == "imagenet":
        mean = inference_config.get(
            "imagenet_mean",
            [0.485, 0.456, 0.406],
        )

        std = inference_config.get(
            "imagenet_std",
            [0.229, 0.224, 0.225],
        )

        transform_steps.append(
            transforms.Normalize(
                mean=mean,
                std=std,
            )
        )

    elif normalization != "0_1":
        raise ValueError("normalization must be 'imagenet' or '0_1'.")

    return transforms.Compose(transform_steps)


def _build_torchvision_binary_model(
    architecture: str,
    dropout: float,
):
    """Reconstruct supported TorchVision binary classifiers."""
    import torch.nn as nn
    from torchvision.models import (
        convnext_tiny,
        densenet121,
        efficientnet_b0,
        resnet50,
    )

    architecture = architecture.lower()

    if architecture == "efficientnet_b0":
        # Match the original training notebook exactly.
        model = efficientnet_b0(weights=None)

        in_features = model.classifier[-1].in_features

        model.classifier[0].p = float(dropout)

        model.classifier[-1] = nn.Linear(
            in_features,
            1,
        )

    elif architecture == "resnet50":
        model = resnet50(weights=None)

        in_features = model.fc.in_features

        model.fc = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(
                in_features,
                1,
            ),
        )

    elif architecture == "densenet121":
        model = densenet121(weights=None)

        in_features = model.classifier.in_features

        model.classifier = nn.Sequential(
            nn.Dropout(p=float(dropout)),
            nn.Linear(
                in_features,
                1,
            ),
        )

    elif architecture == "convnext_tiny":
        model = convnext_tiny(weights=None)

        in_features = model.classifier[-1].in_features

        model.classifier[-1] = nn.Linear(
            in_features,
            1,
        )

    else:
        raise ValueError(
            "Unsupported automatic TorchVision architecture "
            f"'{architecture}'. Supported: "
            f"{IMAGE_EMBEDDING_MODELS}."
        )

    return model


def _load_pytorch_model(
    model_directory: Path,
    model_metadata: dict,
    inference_config: dict,
    device,
):
    """Reconstruct a saved PyTorch model and load its frozen state_dict."""
    import torch

    architecture = inference_config.get("architecture")

    if architecture is None:
        name = model_metadata["model"]["model_name"].lower()

        for candidate in IMAGE_EMBEDDING_MODELS:
            if candidate in name:
                architecture = candidate
                break

    if architecture is None:
        raise ValueError(
            "Could not infer PyTorch architecture. "
            "Add architecture to training_configuration.json, "
            "inference_config.json or MODEL_OVERRIDES."
        )

    dropout = float(
        inference_config.get(
            "dropout",
            0.2,
        )
    )

    model = _build_torchvision_binary_model(
        architecture=str(architecture),
        dropout=dropout,
    )

    model_file = model_metadata["model"].get(
        "model_file",
        "model_state_dict.pt",
    )

    state_path = model_directory / model_file

    if not state_path.exists():
        raise FileNotFoundError(f"PyTorch state_dict not found: {state_path}")

    # Load on CPU first. Moving one reconstructed model to the accelerator
    # is faster and more robust than deserializing the state directly there.
    try:
        state = torch.load(
            state_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        state = torch.load(
            state_path,
            map_location="cpu",
        )

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if (
        isinstance(state, dict)
        and state
        and all(str(key).startswith("module.") for key in state)
    ):
        state = {str(key)[7:]: value for key, value in state.items()}

    try:
        model.load_state_dict(
            state,
            strict=True,
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "The saved PyTorch state_dict does not match the "
            f"reconstructed '{architecture}' classifier. "
            "No model was retrained. Provide the exact architecture "
            "through inference_config.json or MODEL_OVERRIDES."
        ) from exc

    model = model.to(device)
    model.eval()

    return model


def _default_inference_batch_size(
    device_type: str,
) -> int:
    """Choose a conservative but inference-oriented default batch size."""
    if device_type == "cuda":
        return 256

    if device_type == "mps":
        return 128

    return 64


def _default_inference_num_workers(
    device_type: str,
) -> int:
    """Choose safe DataLoader defaults.

    CUDA benefits from parallel JPEG decoding. On macOS/MPS, multiprocessing
    can duplicate the in-memory JPEG-byte dataset and is therefore disabled by
    default; users can override it explicitly when appropriate.
    """
    if device_type == "cuda":
        return 4

    return 0


def _predict_direct_pytorch(
    model_directory: Path,
    model_metadata: dict,
    split_df: pd.DataFrame,
    inference_config: dict,
    config_path: str | Path,
    device_name: str,
    split_name: str,
) -> np.ndarray:
    """Generate direct-image probabilities efficiently with Dataset/DataLoader."""
    import time

    import torch
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    device = _select_torch_device(device_name)

    image_size = int(
        inference_config.get(
            "image_size",
            136,
        )
    )

    io_batch_size = int(
        inference_config.get(
            "io_batch_size",
            4096,
        )
    )

    batch_size = int(
        inference_config.get(
            "inference_batch_size",
            _default_inference_batch_size(device.type),
        )
    )

    num_workers = int(
        inference_config.get(
            "inference_num_workers",
            _default_inference_num_workers(device.type),
        )
    )

    if batch_size <= 0:
        raise ValueError("inference_batch_size must be positive.")

    if num_workers < 0:
        raise ValueError("inference_num_workers cannot be negative.")

    target_ids = split_df["isic_id"].astype(str).tolist()

    print("\n" + "-" * 72)
    print(f"DIRECT IMAGE INFERENCE | {model_metadata['model']['model_name']}")
    print(f"Split: {split_name} | Lesions: {len(target_ids):,}")
    print(f"Device: {device} | Batch size: {batch_size} | Workers: {num_workers}")

    # --------------------------------------------------------------
    # Reconstruct and load the frozen model
    # --------------------------------------------------------------
    print("Loading frozen PyTorch model...")

    model_load_start = time.perf_counter()

    model = _load_pytorch_model(
        model_directory=model_directory,
        model_metadata=model_metadata,
        inference_config=inference_config,
        device=device,
    )

    print(f"Model loaded in {time.perf_counter() - model_load_start:.1f} s.")

    # --------------------------------------------------------------
    # Resolve the exact image preprocessing run
    # --------------------------------------------------------------
    (
        manifest_path,
        parquet_paths,
        _,
    ) = _resolve_image_manifest(
        image_size=image_size,
        config_path=config_path,
        inference_config=inference_config,
    )

    print(f"Image manifest: {manifest_path.name}")
    print(f"Image shards: {len(parquet_paths):,}")

    # --------------------------------------------------------------
    # Load only requested JPEG bytes
    # --------------------------------------------------------------
    print(f"Loading {split_name} image bytes...")

    image_load_start = time.perf_counter()

    image_df = _load_selected_image_bytes(
        parquet_paths=parquet_paths,
        lesion_ids=target_ids,
        io_batch_size=io_batch_size,
    )

    # Restore exact split order.
    order = pd.DataFrame(
        {
            "isic_id": target_ids,
            "_inference_order": np.arange(len(target_ids)),
        }
    )

    image_df = (
        order.merge(
            image_df,
            on="isic_id",
            how="left",
            validate="one_to_one",
        )
        .sort_values("_inference_order")
        .drop(columns="_inference_order")
        .reset_index(drop=True)
    )

    if image_df["image"].isna().any():
        raise ValueError(f"Missing image bytes after aligning {split_name} images.")

    image_load_seconds = time.perf_counter() - image_load_start

    print(f"Loaded {len(image_df):,} images in {image_load_seconds:.1f} s.")

    # --------------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------------
    evaluation_transform = _make_evaluation_transform(inference_config)

    dataset = _InferenceImageDataset(
        dataframe=image_df,
        image_size=image_size,
        transform=evaluation_transform,
    )

    pin_memory = device.type == "cuda"

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

        loader_kwargs["prefetch_factor"] = int(
            inference_config.get(
                "prefetch_factor",
                2,
            )
        )

    loader = DataLoader(**loader_kwargs)

    # --------------------------------------------------------------
    # Batched forward pass
    # --------------------------------------------------------------
    print(f"Running {split_name} forward pass...")

    inference_start = time.perf_counter()

    probabilities: list[np.ndarray] = []
    returned_ids: list[str] = []

    output_type = str(
        inference_config.get(
            "output_type",
            "logit",
        )
    ).lower()

    use_amp = bool(
        inference_config.get(
            "inference_mixed_precision",
            device.type == "cuda",
        )
    )

    model.eval()

    with torch.inference_mode():
        for images, lesion_ids in tqdm(
            loader,
            total=len(loader),
            desc=f"{split_name.capitalize()} inference",
            unit="batch",
        ):
            images = images.to(
                device,
                non_blocking=pin_memory,
            )

            if device.type == "cuda" and use_amp:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):
                    output = model(images).reshape(-1)
            else:
                output = model(images).reshape(-1)

            if output_type == "logit":
                batch_probabilities = torch.sigmoid(output)
            elif output_type == "probability":
                batch_probabilities = output
            else:
                raise ValueError("output_type must be 'logit' or 'probability'.")

            probabilities.append(
                batch_probabilities.detach().cpu().numpy().astype(float)
            )

            returned_ids.extend(str(value) for value in lesion_ids)

    probabilities_array = np.concatenate(probabilities)

    inference_seconds = time.perf_counter() - inference_start

    if returned_ids != target_ids:
        raise ValueError("Direct-image inference changed lesion ordering.")

    if not np.isfinite(probabilities_array).all():
        raise ValueError("Direct-image model returned non-finite probabilities.")

    if np.any((probabilities_array < 0.0) | (probabilities_array > 1.0)):
        raise ValueError("Direct-image probabilities must lie in [0, 1].")

    images_per_second = (
        len(target_ids) / inference_seconds if inference_seconds > 0 else np.nan
    )

    print(
        f"Forward pass completed in "
        f"{inference_seconds / 60.0:.1f} min "
        f"({images_per_second:.1f} images/s)."
    )
    print("-" * 72)

    # Release the large in-memory JPEG table as soon as possible.
    del loader
    del dataset
    del image_df

    if device.type == "cuda":
        torch.cuda.empty_cache()

    elif device.type == "mps" and hasattr(
        torch,
        "mps",
    ):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    return probabilities_array


# ---------------------------------------------------------------------
# Cached inference
# ---------------------------------------------------------------------


def get_or_create_predictions(
    model_directory: str | Path,
    split_df: pd.DataFrame,
    split: str,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    override: dict | None = None,
    force_recompute: bool = False,
    device: str = "auto",
) -> pd.DataFrame:
    """Load cached probabilities or rerun inference without retraining."""
    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'.")

    model_directory = _project_path(model_directory)
    cache_filename = (
        VALIDATION_PREDICTIONS_FILENAME
        if split == "validation"
        else TEST_PREDICTIONS_FILENAME
    )
    cache_path = model_directory / cache_filename

    if cache_path.exists() and not force_recompute:
        predictions = _canonicalize_predictions(pd.read_parquet(cache_path))
        _validate_prediction_coverage(predictions, split_df, split)
        print(f"Using cached {split} predictions: {cache_path}")
        return predictions

    print(
        f"No cached {split} Parquet found for "
        f"{model_directory.name}. Recomputing probabilities only "
        "(the base model will NOT be retrained)."
    )

    _, model_metadata = _load_model_metadata(model_directory)
    inference_config = _merged_inference_config(
        model_directory,
        model_metadata,
        override,
    )

    framework = str(model_metadata["model"].get("framework", "")).lower()
    model_file = model_metadata["model"].get("model_file")

    if framework == "scikit-learn" or (
        model_file is not None and str(model_file).endswith(".joblib")
    ):
        path = model_directory / str(model_file or "model.joblib")
        if not path.exists():
            raise FileNotFoundError(f"Saved model not found: {path}")

        print(f"Loading saved model: {path.name}")
        model = joblib.load(path)

        hypothesis = infer_hypothesis(model_directory)

        print(f"Preparing {split} features...")
        X = _prepare_sklearn_features(
            model=model,
            model_directory=model_directory,
            model_metadata=model_metadata,
            split_df=split_df,
            split=split,
            hypothesis=hypothesis,
            inference_config=inference_config,
            config_path=config_path,
        )

        print(f"Running {split} predict_proba()...")
        probabilities = _positive_probability_from_sklearn(model, X)

    elif framework == "pytorch":
        probabilities = _predict_direct_pytorch(
            model_directory=model_directory,
            model_metadata=model_metadata,
            split_df=split_df,
            inference_config=inference_config,
            config_path=config_path,
            device_name=device,
            split_name=split,
        )

    elif framework in {"keras", "tensorflow"}:
        raise NotImplementedError(
            "Automatic fallback inference for legacy Keras direct-image models "
            "is intentionally not guessed. If such a model already has cached "
            "validation/test prediction Parquets it works directly; otherwise "
            "add a project-specific loader or inference_config implementation."
        )

    else:
        raise ValueError(f"Unsupported framework '{framework}' in {model_directory}.")

    predictions = pd.DataFrame(
        {
            "isic_id": split_df["isic_id"].astype(str).to_numpy(),
            "probability": np.asarray(probabilities, dtype=float),
        }
    )

    predictions = _canonicalize_predictions(predictions)
    _validate_prediction_coverage(predictions, split_df, split)

    predictions.to_parquet(cache_path, index=False)

    print(f"Calculated and cached {split} predictions: {cache_path}")

    return predictions


# ---------------------------------------------------------------------
# Fusion model
# ---------------------------------------------------------------------


def _interaction_features(
    metadata_probability: np.ndarray,
    image_probability: np.ndarray,
) -> np.ndarray:
    """Build a small nonlinear feature set for interaction stacking.

    The fourth fusion remains deliberately low-dimensional because the
    validation set contains relatively few independent patients. The features
    allow the contribution of the image model to depend on agreement or
    disagreement with the metadata model without introducing a high-capacity
    meta-learner.
    """
    p_meta = np.asarray(metadata_probability, dtype=float)
    p_image = np.asarray(image_probability, dtype=float)

    if p_meta.shape != p_image.shape:
        raise ValueError("Base probability arrays must have the same shape.")

    return np.column_stack(
        [
            p_meta,
            p_image,
            p_meta * p_image,
            np.abs(p_meta - p_image),
        ]
    )


@dataclass
class ProbabilityFusionModel:
    """Serializable late-fusion model operating on two base probabilities."""

    method: str
    metadata_weight: float | None = None
    logistic_model: LogisticRegression | None = None

    def predict_probability(
        self,
        metadata_probability: np.ndarray,
        image_probability: np.ndarray,
    ) -> np.ndarray:
        metadata_probability = np.asarray(
            metadata_probability,
            dtype=float,
        )
        image_probability = np.asarray(
            image_probability,
            dtype=float,
        )

        if metadata_probability.shape != image_probability.shape:
            raise ValueError("Base probability arrays must have the same shape.")

        if self.method == "simple_average":
            return 0.5 * metadata_probability + 0.5 * image_probability

        if self.method == "weighted_average":
            if self.metadata_weight is None:
                raise ValueError("metadata_weight is missing.")

            weight = float(self.metadata_weight)

            return weight * metadata_probability + (1.0 - weight) * image_probability

        if self.method == "logistic_stacking":
            if self.logistic_model is None:
                raise ValueError("logistic_model is missing.")

            features = np.column_stack(
                [
                    metadata_probability,
                    image_probability,
                ]
            )

            return self.logistic_model.predict_proba(features)[:, 1]

        if self.method == "interaction_stacking":
            if self.logistic_model is None:
                raise ValueError("logistic_model is missing.")

            features = _interaction_features(
                metadata_probability,
                image_probability,
            )

            return self.logistic_model.predict_proba(features)[:, 1]

        raise ValueError(f"Unsupported fusion method: {self.method}")


def _metric_key(
    metrics: dict[str, float],
) -> tuple[float, float, float]:
    """Higher tuple is better: PR-AUC, ROC-AUC, then negative Brier."""
    return (
        float(metrics["pr_auc"]),
        float(metrics["roc_auc"]),
        -float(metrics["brier_score"]),
    )


def _fusion_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    return ev.compute_threshold_free_metrics(
        y_true=np.asarray(
            y_true,
            dtype=int,
        ),
        y_prob=np.asarray(
            y_prob,
            dtype=float,
        ),
    )


def _select_weight(
    y_true: np.ndarray,
    metadata_prob: np.ndarray,
    image_prob: np.ndarray,
    weight_grid: np.ndarray,
) -> tuple[float, dict[str, float]]:
    best_weight: float | None = None
    best_metrics: dict[str, float] | None = None

    for weight in weight_grid:
        probability = float(weight) * metadata_prob + (1.0 - float(weight)) * image_prob

        metrics = _fusion_metrics(
            y_true,
            probability,
        )

        if best_metrics is None or _metric_key(metrics) > _metric_key(best_metrics):
            best_weight = float(weight)
            best_metrics = metrics

    assert best_weight is not None
    assert best_metrics is not None

    return best_weight, best_metrics


def _merge_fusion_validation_data(
    validation_split: pd.DataFrame,
    metadata_predictions: pd.DataFrame,
    image_predictions: pd.DataFrame,
    hypothesis: int,
) -> pd.DataFrame:
    target = ev.DEFAULT_LABEL_COLUMNS[hypothesis]

    base = validation_split.loc[
        :,
        [
            "isic_id",
            "patient_id",
            target,
        ],
    ].copy()

    base["isic_id"] = base["isic_id"].astype(str)
    base["patient_id"] = base["patient_id"].astype(str)

    metadata_predictions = _canonicalize_predictions(metadata_predictions).rename(
        columns={"probability": "metadata_probability"}
    )

    image_predictions = _canonicalize_predictions(image_predictions).rename(
        columns={"probability": "image_probability"}
    )

    merged = base.merge(
        metadata_predictions,
        on="isic_id",
        how="inner",
        validate="one_to_one",
    ).merge(
        image_predictions,
        on="isic_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != len(base):
        raise ValueError(
            "Fusion validation merge changed the number of validation lesions."
        )

    return merged


def _fit_logistic_fusion(
    features: np.ndarray,
    y_true: np.ndarray,
    random_state: int,
) -> LogisticRegression:
    """Fit a deliberately regularised low-capacity stacking model."""
    model = LogisticRegression(
        solver="lbfgs",
        C=1.0,
        l1_ratio=0,
        max_iter=2000,
        random_state=random_state,
    )

    model.fit(
        features,
        y_true,
    )

    return model


def select_fusion_with_grouped_cv(
    validation_split: pd.DataFrame,
    metadata_predictions: pd.DataFrame,
    image_predictions: pd.DataFrame,
    hypothesis: int,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    weight_grid_step: float = 0.01,
    min_pr_auc_gain: float = 0.01,
) -> dict[str, Any]:
    """Select the best candidate using validation only.

    The two already-trained base models are explicit candidates together with
    four fusion methods. The fusion methods that require fitting are evaluated
    with patient-grouped out-of-fold predictions. Base models are fixed before
    validation and therefore their validation probabilities can be evaluated
    directly without refitting.

    Selection hierarchy:
    1. PR-AUC (higher)
    2. ROC-AUC (higher)
    3. Brier score (lower)
    """
    if not 0 < weight_grid_step <= 1:
        raise ValueError("weight_grid_step must lie in (0, 1].")

    if min_pr_auc_gain < 0:
        raise ValueError("min_pr_auc_gain must be >= 0.")

    data = _merge_fusion_validation_data(
        validation_split=validation_split,
        metadata_predictions=metadata_predictions,
        image_predictions=image_predictions,
        hypothesis=hypothesis,
    )

    target = ev.DEFAULT_LABEL_COLUMNS[hypothesis]

    y = data[target].to_numpy(dtype=int)
    groups = data["patient_id"].astype(str).to_numpy()
    p_meta = data["metadata_probability"].to_numpy(dtype=float)
    p_image = data["image_probability"].to_numpy(dtype=float)

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    # Base candidates are already frozen. Keeping them in the same OOF
    # dictionary makes the final comparison completely explicit.
    oof = {
        "metadata_only": p_meta.copy(),
        "image_only": p_image.copy(),
    }

    for method in FUSION_METHODS:
        oof[method] = np.full(
            len(data),
            np.nan,
            dtype=float,
        )

    weight_grid = np.arange(
        0.0,
        1.0 + weight_grid_step / 2.0,
        weight_grid_step,
        dtype=float,
    )
    weight_grid = np.unique(
        np.clip(
            weight_grid,
            0.0,
            1.0,
        )
    )

    fold_details: list[dict[str, Any]] = []

    for fold, (
        fit_idx,
        holdout_idx,
    ) in enumerate(
        splitter.split(
            X=np.zeros((len(data), 1)),
            y=y,
            groups=groups,
        )
    ):
        y_fit = y[fit_idx]
        y_holdout = y[holdout_idx]

        if set(np.unique(y_fit)) != {0, 1}:
            raise ValueError(f"CV fold {fold}: fit partition lacks one target class.")

        if set(np.unique(y_holdout)) != {0, 1}:
            raise ValueError(
                f"CV fold {fold}: holdout partition "
                "lacks one target class. Reduce n_splits."
            )

        # ----------------------------------------------------------
        # 1. Simple average
        # ----------------------------------------------------------
        oof["simple_average"][holdout_idx] = (
            0.5 * p_meta[holdout_idx] + 0.5 * p_image[holdout_idx]
        )

        # ----------------------------------------------------------
        # 2. Weighted average
        # ----------------------------------------------------------
        fold_weight, _ = _select_weight(
            y_true=y_fit,
            metadata_prob=p_meta[fit_idx],
            image_prob=p_image[fit_idx],
            weight_grid=weight_grid,
        )

        oof["weighted_average"][holdout_idx] = (
            fold_weight * p_meta[holdout_idx]
            + (1.0 - fold_weight) * p_image[holdout_idx]
        )

        # ----------------------------------------------------------
        # 3. Linear logistic stacking
        # ----------------------------------------------------------
        logistic_features_fit = np.column_stack(
            [
                p_meta[fit_idx],
                p_image[fit_idx],
            ]
        )

        logistic_features_holdout = np.column_stack(
            [
                p_meta[holdout_idx],
                p_image[holdout_idx],
            ]
        )

        logistic = _fit_logistic_fusion(
            logistic_features_fit,
            y_fit,
            random_state,
        )

        oof["logistic_stacking"][holdout_idx] = logistic.predict_proba(
            logistic_features_holdout
        )[:, 1]

        # ----------------------------------------------------------
        # 4. Interaction stacking
        # ----------------------------------------------------------
        interaction_fit = _interaction_features(
            p_meta[fit_idx],
            p_image[fit_idx],
        )

        interaction_holdout = _interaction_features(
            p_meta[holdout_idx],
            p_image[holdout_idx],
        )

        interaction_model = _fit_logistic_fusion(
            interaction_fit,
            y_fit,
            random_state,
        )

        oof["interaction_stacking"][holdout_idx] = interaction_model.predict_proba(
            interaction_holdout
        )[:, 1]

        fold_details.append(
            {
                "fold": fold,
                "fit_lesions": len(fit_idx),
                "holdout_lesions": len(holdout_idx),
                "fit_patients": int(pd.Series(groups[fit_idx]).nunique()),
                "holdout_patients": int(pd.Series(groups[holdout_idx]).nunique()),
                "weighted_average_metadata_weight": fold_weight,
            }
        )

    for method in FUSION_METHODS:
        if np.isnan(oof[method]).any():
            raise RuntimeError(f"OOF predictions are incomplete for {method}.")

    candidate_metrics = {
        method: _fusion_metrics(
            y,
            oof[method],
        )
        for method in ALL_CANDIDATES
    }

    best_base_method = max(
        BASE_CANDIDATES,
        key=lambda method: _metric_key(candidate_metrics[method]),
    )

    best_fusion_method = max(
        FUSION_METHODS,
        key=lambda method: _metric_key(candidate_metrics[method]),
    )

    best_base_pr_auc = float(candidate_metrics[best_base_method]["pr_auc"])

    best_fusion_pr_auc = float(candidate_metrics[best_fusion_method]["pr_auc"])

    fusion_pr_auc_gain = best_fusion_pr_auc - best_base_pr_auc

    # A fusion replaces the best individual model only if it improves
    # validation PR-AUC by at least the predefined absolute margin.
    if fusion_pr_auc_gain >= min_pr_auc_gain:
        selected_method = best_fusion_method
    else:
        selected_method = best_base_method

    # --------------------------------------------------------------
    # Fit the best fusion on complete validation only if a fusion
    # candidate wins the overall validation comparison.
    # --------------------------------------------------------------
    fusion_model: ProbabilityFusionModel | None = None

    final_validation_probability: np.ndarray | None = None

    if selected_method in FUSION_METHODS:
        if selected_method == "simple_average":
            fusion_model = ProbabilityFusionModel(
                method="simple_average",
            )

        elif selected_method == "weighted_average":
            final_weight, _ = _select_weight(
                y_true=y,
                metadata_prob=p_meta,
                image_prob=p_image,
                weight_grid=weight_grid,
            )

            fusion_model = ProbabilityFusionModel(
                method="weighted_average",
                metadata_weight=final_weight,
            )

        elif selected_method == "logistic_stacking":
            final_logistic = _fit_logistic_fusion(
                np.column_stack(
                    [
                        p_meta,
                        p_image,
                    ]
                ),
                y,
                random_state,
            )

            fusion_model = ProbabilityFusionModel(
                method="logistic_stacking",
                logistic_model=final_logistic,
            )

        elif selected_method == "interaction_stacking":
            final_interaction = _fit_logistic_fusion(
                _interaction_features(
                    p_meta,
                    p_image,
                ),
                y,
                random_state,
            )

            fusion_model = ProbabilityFusionModel(
                method="interaction_stacking",
                logistic_model=final_interaction,
            )

        final_validation_probability = fusion_model.predict_probability(
            metadata_probability=p_meta,
            image_probability=p_image,
        )

    return {
        "data": data,
        "y_true": y,
        "oof_predictions": oof,
        "candidate_metrics": candidate_metrics,
        # Backwards-compatible alias used by downstream code.
        "method_metrics": candidate_metrics,
        "best_base_method": best_base_method,
        "best_fusion_method": best_fusion_method,
        "fusion_pr_auc_gain": fusion_pr_auc_gain,
        "min_pr_auc_gain": float(min_pr_auc_gain),
        "selected_method": selected_method,
        "selected_is_fusion": selected_method in FUSION_METHODS,
        "fusion_model": fusion_model,
        "final_validation_probability": final_validation_probability,
        "fold_details": fold_details,
        "weight_grid": weight_grid,
    }


def _create_fusion_directory(
    metadata_model_directory: Path,
    image_model_directory: Path,
    config_path: str | Path,
) -> Path:
    config = load_yaml_config(_project_path(config_path))

    models_dir = _project_path(config["paths"]["models"])
    models_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _, meta_metadata = _load_model_metadata(metadata_model_directory)
    _, image_metadata = _load_model_metadata(image_model_directory)

    base_name = _sanitize_name(
        "fusion_"
        + meta_metadata["model"]["model_name"]
        + "__"
        + image_metadata["model"]["model_name"]
    )

    run_directory = models_dir / base_name

    if not run_directory.exists():
        run_directory.mkdir()
        return run_directory

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_directory = models_dir / f"{base_name}_{timestamp}"

    counter = 1

    while run_directory.exists():
        run_directory = models_dir / (f"{base_name}_{timestamp}_{counter:02d}")
        counter += 1

    run_directory.mkdir()

    return run_directory


def _fusion_model_parameters(
    fusion_model: ProbabilityFusionModel,
) -> dict[str, Any]:
    """Return JSON-serialisable parameters for the selected fusion."""
    parameters: dict[
        str,
        Any,
    ] = {}

    if fusion_model.method == "weighted_average":
        parameters["metadata_weight"] = float(fusion_model.metadata_weight)
        parameters["image_weight"] = 1.0 - float(fusion_model.metadata_weight)

    elif fusion_model.method in {
        "logistic_stacking",
        "interaction_stacking",
    }:
        logistic = fusion_model.logistic_model

        parameters["intercept"] = float(logistic.intercept_[0])

        if fusion_model.method == "logistic_stacking":
            feature_names = [
                "metadata_probability",
                "image_probability",
            ]
        else:
            feature_names = [
                "metadata_probability",
                "image_probability",
                "probability_interaction",
                "absolute_probability_difference",
            ]

        parameters["coefficients"] = {
            feature_name: float(coefficient)
            for (
                feature_name,
                coefficient,
            ) in zip(
                feature_names,
                logistic.coef_[0],
                strict=True,
            )
        }

    return parameters


def save_selected_fusion_model(
    selection: dict[str, Any],
    metadata_model_directory: str | Path,
    image_model_directory: str | Path,
    hypothesis: int,
    target_sensitivity: float,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    n_splits: int = 5,
    random_state: int = 42,
    weight_grid_step: float = 0.01,
) -> dict[str, Any]:
    """Save a fusion artefact only when a fusion candidate wins validation."""
    if not selection["selected_is_fusion"]:
        raise ValueError(
            "save_selected_fusion_model() was called "
            "although an individual base model won validation."
        )

    metadata_model_directory = _project_path(metadata_model_directory)
    image_model_directory = _project_path(image_model_directory)

    run_directory = _create_fusion_directory(
        metadata_model_directory=metadata_model_directory,
        image_model_directory=image_model_directory,
        config_path=config_path,
    )

    fusion_model = selection["fusion_model"]

    if fusion_model is None:
        raise RuntimeError("Selected fusion model is missing.")

    fusion_model_path = run_directory / "fusion_model.joblib"

    joblib.dump(
        fusion_model,
        fusion_model_path,
    )

    data = selection["data"]

    validation_probability = np.asarray(
        selection["final_validation_probability"],
        dtype=float,
    )

    validation_predictions = pd.DataFrame(
        {
            "isic_id": data["isic_id"].astype(str).to_numpy(),
            "probability": validation_probability,
        }
    )

    validation_predictions.to_parquet(
        run_directory / VALIDATION_PREDICTIONS_FILENAME,
        index=False,
    )

    selected_threshold = ev.select_clinical_threshold(
        y_true=np.asarray(
            selection["y_true"],
            dtype=int,
        ),
        y_prob=validation_probability,
        target_sensitivity=target_sensitivity,
    )

    _, meta_metadata = _load_model_metadata(metadata_model_directory)
    _, image_metadata = _load_model_metadata(image_model_directory)

    model_name = run_directory.name

    model_metadata = {
        "created_at": _now(),
        "model": {
            "model_name": model_name,
            "model_type": 4,
            "framework": "fusion",
            "class_name": "ProbabilityFusionModel",
            "class_module": "skin_lesion_ai.inference.final_models",
            "model_file": fusion_model_path.name,
        },
        "inference": {
            "input_kind": "multimodal_probability_fusion",
            "metadata_model_directory": metadata_model_directory,
            "image_model_directory": image_model_directory,
        },
        "evaluation": {
            "evaluated_at": _now(),
            "hypothesis": hypothesis,
            "hypothesis_description": ev.HYPOTHESIS_DESCRIPTIONS[hypothesis],
            "label_column": ev.DEFAULT_LABEL_COLUMNS[hypothesis],
            "threshold_selection_split": "validation",
            "threshold_applied_to": [
                "validation",
                "test",
            ],
            "target_sensitivity": target_sensitivity,
            "selected_threshold": selected_threshold,
        },
    }

    _write_json(
        model_metadata,
        run_directory / "model_metadata.json",
    )

    rows = []

    for method in ALL_CANDIDATES:
        metrics = selection["candidate_metrics"][method]

        rows.append(
            {
                "method": method,
                "candidate_type": (
                    "base_model" if method in BASE_CANDIDATES else "fusion"
                ),
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "best_fusion": (method == selection["best_fusion_method"]),
                "selected_overall": (method == selection["selected_method"]),
            }
        )

    cv_results = pd.DataFrame(rows)

    cv_results.to_csv(
        run_directory / "fusion_cv_results.csv",
        index=False,
        float_format="%.10g",
    )

    final_parameters = _fusion_model_parameters(fusion_model)

    fusion_metadata = {
        "created_at": _now(),
        "base_models": {
            "metadata": {
                "directory": metadata_model_directory,
                "model_name": meta_metadata["model"]["model_name"],
            },
            "image": {
                "directory": image_model_directory,
                "model_name": image_metadata["model"]["model_name"],
            },
        },
        "selection": {
            "dataset": "validation",
            "cv": "StratifiedGroupKFold",
            "group": "patient_id",
            "n_splits": n_splits,
            "shuffle": True,
            "random_state": random_state,
            "selection_hierarchy": [
                "pr_auc_max",
                "roc_auc_max",
                "brier_score_min",
            ],
            "weight_grid_step": weight_grid_step,
            "min_pr_auc_gain": selection["min_pr_auc_gain"],
            "fusion_pr_auc_gain": selection["fusion_pr_auc_gain"],
            "candidate_metrics": selection["candidate_metrics"],
            "best_base_method": selection["best_base_method"],
            "best_fusion_method": selection["best_fusion_method"],
            "selected_method": selection["selected_method"],
            "fold_details": selection["fold_details"],
        },
        "final_fit": {
            "dataset": "complete_validation",
            "method": fusion_model.method,
            "parameters": final_parameters,
            "selected_threshold": selected_threshold,
            "target_sensitivity": target_sensitivity,
        },
    }

    _write_json(
        fusion_metadata,
        run_directory / "fusion_metadata.json",
    )

    return {
        "model_directory": run_directory,
        "fusion_model": fusion_model,
        "selected_threshold": selected_threshold,
        "validation_predictions": validation_predictions,
        "cv_results": cv_results,
        "fusion_metadata": fusion_metadata,
    }


# ---------------------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------------------


def _plot_test_precision_recall(
    merged: pd.DataFrame,
    output_directory: Path,
) -> Path:
    y_true = merged["y_true"].to_numpy(dtype=int)
    y_prob = merged["y_prob"].to_numpy(dtype=float)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    prevalence = float(y_true.mean())

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, linewidth=2, label=f"Test (PR-AUC = {pr_auc:.3f})")
    ax.axhline(
        prevalence,
        linestyle=":",
        linewidth=1.5,
        label=f"Test prevalence ({prevalence:.3f})",
    )
    ax.set_xlabel("Recall / Sensitivity")
    ax.set_ylabel("Precision / PPV")
    ax.set_title("Test Precision-Recall curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    path = output_directory / "precision_recall_curve.jpg"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_test_roc(
    merged: pd.DataFrame,
    output_directory: Path,
) -> Path:
    y_true = merged["y_true"].to_numpy(dtype=int)
    y_prob = merged["y_prob"].to_numpy(dtype=float)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, linewidth=2, label=f"Test (ROC-AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.5, label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("Sensitivity / True positive rate")
    ax.set_title("Test ROC curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    path = output_directory / "roc_curve.jpg"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_test_combined(
    merged: pd.DataFrame,
    output_directory: Path,
) -> Path:
    y_true = merged["y_true"].to_numpy(dtype=int)
    y_prob = merged["y_prob"].to_numpy(dtype=float)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    prevalence = float(y_true.mean())

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(
        recall,
        precision,
        linewidth=2,
        label=f"Test (PR-AUC = {pr_auc:.3f})",
    )
    axes[0].axhline(
        prevalence,
        linestyle=":",
        linewidth=1.5,
        label=f"Prevalence ({prevalence:.3f})",
    )
    axes[0].set_xlabel("Recall / Sensitivity")
    axes[0].set_ylabel("Precision / PPV")
    axes[0].set_title("Precision-Recall")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].legend()

    axes[1].plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"Test (ROC-AUC = {roc_auc:.3f})",
    )
    axes[1].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Random classifier",
    )
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("Sensitivity / True positive rate")
    axes[1].set_title("ROC")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.suptitle("Final test discrimination curves")
    fig.tight_layout()

    path = output_directory / "pr_roc_curves.jpg"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_test_clinical_summary(
    merged: pd.DataFrame,
    metrics: dict[str, float | int],
    threshold: float,
    hypothesis: int,
    output_directory: Path,
) -> Path:
    y_true = merged["y_true"].to_numpy(dtype=int)
    y_prob = merged["y_prob"].to_numpy(dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    class_labels = (
        ["No biopsy", "Biopsy"] if hypothesis == 1 else ["Not malignant", "Malignant"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=",d",
        cmap="Blues",
        cbar=True,
        ax=axes[0],
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    axes[0].set_xlabel("Predicted class")
    axes[0].set_ylabel("True class")
    axes[0].set_title("Test confusion matrix")

    metric_names = ["Sensitivity", "Specificity", "PPV", "NPV"]
    metric_values = [
        metrics["sensitivity"],
        metrics["specificity"],
        metrics["ppv"],
        metrics["npv"],
    ]

    bars = axes[1].bar(metric_names, metric_values)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Metric value")
    axes[1].set_title("Test clinical metrics")
    axes[1].tick_params(axis="x", labelrotation=20)

    for bar, value in zip(bars, metric_values, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
        )

    fig.suptitle(f"Frozen validation threshold = {threshold:.6f}")
    fig.tight_layout()

    path = output_directory / "clinical_summary.jpg"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def evaluate_test(
    df_test_predictions: pd.DataFrame,
    df_test: pd.DataFrame,
    hypothesis: int,
    model_directory: str | Path,
    selected_threshold: float,
    *,
    prediction_id_col: str = "isic_id",
    probability_col: str = "probability",
    split_id_col: str = "isic_id",
    patient_id_col: str = "patient_id",
    label_col: str | None = None,
) -> dict[str, Any]:
    """Evaluate the final frozen model on test and save outputs under /test."""
    model_directory = _project_path(model_directory)
    test_directory = model_directory / "test"
    test_directory.mkdir(parents=True, exist_ok=True)

    resolved_label = (
        label_col if label_col is not None else ev.DEFAULT_LABEL_COLUMNS[hypothesis]
    )

    merged = ev._validate_and_merge_split(
        df_predictions=df_test_predictions,
        df_split=df_test,
        label_col=resolved_label,
        prediction_id_col=prediction_id_col,
        probability_col=probability_col,
        split_id_col=split_id_col,
        patient_id_col=patient_id_col,
        split_name="test",
    )

    y_true = merged["y_true"].to_numpy(dtype=int)
    y_prob = merged["y_prob"].to_numpy(dtype=float)

    dataset_summary = ev.summarize_split_data(merged)
    threshold_free = ev.compute_threshold_free_metrics(y_true, y_prob)
    threshold_metrics = ev.compute_metrics_at_threshold(
        merged=merged,
        threshold=selected_threshold,
    )

    predictions_path = test_directory / "test_predictions.parquet"
    pd.DataFrame(
        {
            "isic_id": merged["lesion_id"].astype(str),
            "probability": merged["y_prob"].astype(float),
        }
    ).to_parquet(predictions_path, index=False)

    metrics_row = {
        **dataset_summary,
        **threshold_free,
        "selected_threshold": selected_threshold,
        **threshold_metrics,
    }

    metrics_csv = test_directory / "test_metrics.csv"
    pd.DataFrame([metrics_row]).to_csv(
        metrics_csv,
        index=False,
        float_format="%.10g",
    )

    figure_paths = {
        "precision_recall_curve": _plot_test_precision_recall(
            merged,
            test_directory,
        ),
        "roc_curve": _plot_test_roc(
            merged,
            test_directory,
        ),
        "pr_roc_curves": _plot_test_combined(
            merged,
            test_directory,
        ),
        "clinical_summary": _plot_test_clinical_summary(
            merged,
            threshold_metrics,
            selected_threshold,
            hypothesis,
            test_directory,
        ),
    }

    _, model_metadata = _load_model_metadata(model_directory)

    final_json = {
        "created_at": _now(),
        "model": {
            "model_name": model_metadata["model"]["model_name"],
            "model_directory": model_directory,
            "hypothesis": hypothesis,
            "hypothesis_description": ev.HYPOTHESIS_DESCRIPTIONS[hypothesis],
            "label_column": resolved_label,
        },
        "evaluation": {
            "split": "test",
            "threshold_source": "validation",
            "selected_threshold": selected_threshold,
            "test_used_for_model_selection": False,
        },
        "test": {
            "dataset_summary": dataset_summary,
            "threshold_free_metrics": threshold_free,
            "threshold_metrics": threshold_metrics,
        },
        "outputs": {
            "test_directory": test_directory,
            "test_predictions": predictions_path,
            "test_metrics_csv": metrics_csv,
            "figures": figure_paths,
        },
    }

    final_json_path = _write_json(
        final_json,
        test_directory / "metrica_test_final.json",
    )

    # Add a pointer without changing the already frozen validation threshold.
    model_metadata["test_evaluation"] = {
        "evaluated_at": _now(),
        "metrics_file": str((Path("test") / final_json_path.name)),
        "selected_threshold": selected_threshold,
        "threshold_source": "validation",
    }

    _write_json(
        model_metadata,
        model_directory / "model_metadata.json",
    )

    return {
        "summary": pd.DataFrame([metrics_row]),
        "output_directory": test_directory,
        "metrics_json_path": final_json_path,
        "metrics_csv_path": metrics_csv,
        "predictions_path": predictions_path,
        "figure_paths": figure_paths,
    }


# ---------------------------------------------------------------------
# Existing test evaluation helpers
# ---------------------------------------------------------------------


def _test_evaluation_is_complete(
    model_directory: str | Path,
) -> bool:
    """Return True only when the canonical test artefacts are complete."""
    model_directory = _project_path(model_directory)

    test_directory = model_directory / "test"

    required_files = (
        "metrica_test_final.json",
        "test_metrics.csv",
        "test_predictions.parquet",
    )

    return test_directory.is_dir() and all(
        (test_directory / filename).is_file() for filename in required_files
    )


def _load_existing_test_results(
    model_directory: str | Path,
) -> dict[str, Any]:
    """Load an already-computed canonical test evaluation without rerunning it."""
    model_directory = _project_path(model_directory)

    test_directory = model_directory / "test"

    if not _test_evaluation_is_complete(model_directory):
        raise FileNotFoundError(
            f"Canonical test evaluation is incomplete in {test_directory}."
        )

    metrics_json_path = test_directory / "metrica_test_final.json"

    metrics_csv_path = test_directory / "test_metrics.csv"

    predictions_path = test_directory / "test_predictions.parquet"

    with metrics_json_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metrics_json = json.load(handle)

    summary = pd.read_csv(metrics_csv_path)

    print(f"Using existing test evaluation: {test_directory}")

    return {
        "summary": summary,
        "output_directory": test_directory,
        "metrics_json_path": metrics_json_path,
        "metrics_csv_path": metrics_csv_path,
        "predictions_path": predictions_path,
        "figure_paths": {
            key: test_directory / filename
            for key, filename in {
                "precision_recall_curve": "precision_recall_curve.jpg",
                "roc_curve": "roc_curve.jpg",
                "pr_roc_curves": "pr_roc_curves.jpg",
                "clinical_summary": "clinical_summary.jpg",
            }.items()
            if (test_directory / filename).exists()
        },
        "metrics_json": metrics_json,
        "reused_existing_test": True,
    }


def _ensure_base_test_evaluation(
    model_directory: Path,
    test_split: pd.DataFrame,
    validation_split: pd.DataFrame,
    hypothesis: int,
    *,
    config_path: str | Path,
    target_sensitivity: float,
    override: dict | None,
    force_recompute_predictions: bool,
    device: str,
) -> dict[str, Any]:
    """Load an existing test evaluation or create it once from the saved model."""
    if _test_evaluation_is_complete(model_directory):
        return _load_existing_test_results(model_directory)

    selected_threshold = _frozen_threshold_from_metadata(model_directory)

    if selected_threshold is None:
        validation_predictions = get_or_create_predictions(
            model_directory=model_directory,
            split_df=validation_split,
            split="validation",
            config_path=config_path,
            override=override,
            force_recompute=force_recompute_predictions,
            device=device,
        )

        merged_validation = ev._validate_and_merge_split(
            df_predictions=validation_predictions,
            df_split=validation_split,
            label_col=ev.DEFAULT_LABEL_COLUMNS[hypothesis],
            prediction_id_col="isic_id",
            probability_col="probability",
            split_id_col="isic_id",
            patient_id_col="patient_id",
            split_name="validation",
        )

        selected_threshold = ev.select_clinical_threshold(
            y_true=merged_validation["y_true"].to_numpy(dtype=int),
            y_prob=merged_validation["y_prob"].to_numpy(dtype=float),
            target_sensitivity=target_sensitivity,
        )

    test_predictions = get_or_create_predictions(
        model_directory=model_directory,
        split_df=test_split,
        split="test",
        config_path=config_path,
        override=override,
        force_recompute=force_recompute_predictions,
        device=device,
    )

    results = evaluate_test(
        df_test_predictions=test_predictions,
        df_test=test_split,
        hypothesis=hypothesis,
        model_directory=model_directory,
        selected_threshold=selected_threshold,
    )

    results["reused_existing_test"] = False

    return results


def _validation_evaluation_row(
    predictions: pd.DataFrame,
    validation_split: pd.DataFrame,
    hypothesis: int,
    threshold: float,
) -> dict[str, float]:
    """Compute validation metrics using an already-frozen threshold."""
    merged = ev._validate_and_merge_split(
        df_predictions=predictions,
        df_split=validation_split,
        label_col=ev.DEFAULT_LABEL_COLUMNS[hypothesis],
        prediction_id_col="isic_id",
        probability_col="probability",
        split_id_col="isic_id",
        patient_id_col="patient_id",
        split_name="validation",
    )

    y_true = merged["y_true"].to_numpy(dtype=int)
    y_prob = merged["y_prob"].to_numpy(dtype=float)

    return {
        **ev.compute_threshold_free_metrics(
            y_true,
            y_prob,
        ),
        **ev.compute_metrics_at_threshold(
            merged,
            threshold,
        ),
        "selected_threshold": float(threshold),
    }


def _test_summary_row(
    test_results: dict[str, Any],
) -> dict[str, Any]:
    """Extract one canonical test metric row."""
    summary = test_results["summary"]

    if summary.empty:
        raise ValueError("Test summary is empty.")

    return summary.iloc[0].to_dict()


def _comparison_row(
    role: str,
    model_name: str,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    *,
    selected: bool,
) -> dict[str, Any]:
    return {
        "role": role,
        "model": model_name,
        "selected": selected,
        "validation_pr_auc": validation_metrics["pr_auc"],
        "test_pr_auc": test_metrics["pr_auc"],
        "validation_roc_auc": validation_metrics["roc_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "validation_brier": validation_metrics["brier_score"],
        "test_brier": test_metrics["brier_score"],
        "threshold": validation_metrics["selected_threshold"],
        "validation_sensitivity": validation_metrics["sensitivity"],
        "test_sensitivity": test_metrics["sensitivity"],
        "validation_specificity": validation_metrics["specificity"],
        "test_specificity": test_metrics["specificity"],
    }


# ---------------------------------------------------------------------
# End-to-end orchestration
# ---------------------------------------------------------------------


def _resolve_model_override(
    model_directory: Path,
    model_overrides: dict[str, dict] | None,
) -> dict | None:
    if not model_overrides:
        return None

    candidates = [
        str(model_directory),
        model_directory.name,
    ]

    try:
        candidates.append(str(model_directory.relative_to(get_project_root())))
    except ValueError:
        pass

    for key in candidates:
        if key in model_overrides:
            return model_overrides[key]

    return None


def _frozen_threshold_from_metadata(
    model_directory: Path,
) -> float | None:
    _, metadata = _load_model_metadata(model_directory)
    threshold = (metadata.get("evaluation") or {}).get("selected_threshold")
    if threshold is None:
        return None
    return float(threshold)


def run_final_model_pipeline(
    model_directories: list[str | Path],
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    target_sensitivity: float = 0.95,
    n_splits: int = 5,
    random_state: int = 42,
    weight_grid_step: float = 0.01,
    min_pr_auc_gain: float = 0.01,
    model_overrides: dict[str, dict] | None = None,
    force_recompute_predictions: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    """Run final evaluation for one model or selection among two base models.

    One model
    ---------
    If the canonical /test evaluation already exists, it is loaded and no
    inference or evaluation is repeated. Otherwise it is created once.

    Two models
    ----------
    1. Validation probabilities are obtained without using test.
    2. The two individual models and four fusion candidates are compared using
       validation only.
    3. If an individual model wins, no fusion directory is created.
    4. If a fusion wins, it is fitted on complete validation, saved, and its
       validation threshold is frozen.
    5. Only after selection is frozen are test evaluations loaded or generated.
    6. A validation-vs-test comparison is printed for the two base models and
       the selected candidate.
    """
    if len(model_directories) not in {1, 2}:
        raise ValueError(
            "Provide exactly one model directory, "
            "or two directories "
            "(one metadata model + one image model)."
        )

    model_directories = [_project_path(path) for path in model_directories]

    for path in model_directories:
        if not path.is_dir():
            raise NotADirectoryError(f"Model directory not found: {path}")

    hypotheses = [infer_hypothesis(path) for path in model_directories]

    if len(set(hypotheses)) != 1:
        raise ValueError(
            f"All input models must belong to the same hypothesis: {hypotheses}"
        )

    hypothesis = hypotheses[0]

    validation_split = _load_split(
        hypothesis=hypothesis,
        split="validation",
        config_path=config_path,
    )

    test_split = _load_split(
        hypothesis=hypothesis,
        split="test",
        config_path=config_path,
    )

    # ============================================================
    # ONE BASE MODEL
    # ============================================================

    if len(model_directories) == 1:
        model_directory = model_directories[0]

        override = _resolve_model_override(
            model_directory,
            model_overrides,
        )

        test_results = _ensure_base_test_evaluation(
            model_directory=model_directory,
            test_split=test_split,
            validation_split=validation_split,
            hypothesis=hypothesis,
            config_path=config_path,
            target_sensitivity=target_sensitivity,
            override=override,
            force_recompute_predictions=force_recompute_predictions,
            device=device,
        )

        selected_threshold = _frozen_threshold_from_metadata(model_directory)

        return {
            "mode": "single_model",
            "hypothesis": hypothesis,
            "final_model_directory": model_directory,
            "selected_threshold": selected_threshold,
            "test_results": test_results,
            "fusion_cv_results": None,
            "comparison": None,
        }

    # ============================================================
    # TWO BASE MODELS
    # ============================================================

    kinds = []

    for directory in model_directories:
        override = _resolve_model_override(
            directory,
            model_overrides,
        )

        kinds.append(
            detect_input_kind(
                directory,
                override=override,
            )
        )

    metadata_indices = [index for index, kind in enumerate(kinds) if kind == "metadata"]

    image_indices = [index for index, kind in enumerate(kinds) if kind != "metadata"]

    if len(metadata_indices) != 1 or len(image_indices) != 1:
        raise ValueError(
            "Two-model fusion requires exactly one metadata "
            "model and one image model. "
            f"Detected input kinds: {kinds}. "
            "Use MODEL_OVERRIDES if an old model name is ambiguous."
        )

    metadata_model_directory = model_directories[metadata_indices[0]]

    image_model_directory = model_directories[image_indices[0]]

    metadata_override = _resolve_model_override(
        metadata_model_directory,
        model_overrides,
    )

    image_override = _resolve_model_override(
        image_model_directory,
        model_overrides,
    )

    # ------------------------------------------------------------
    # VALIDATION ONLY: obtain base predictions
    # ------------------------------------------------------------

    metadata_validation_predictions = get_or_create_predictions(
        model_directory=metadata_model_directory,
        split_df=validation_split,
        split="validation",
        config_path=config_path,
        override=metadata_override,
        force_recompute=force_recompute_predictions,
        device=device,
    )

    image_validation_predictions = get_or_create_predictions(
        model_directory=image_model_directory,
        split_df=validation_split,
        split="validation",
        config_path=config_path,
        override=image_override,
        force_recompute=force_recompute_predictions,
        device=device,
    )

    # ------------------------------------------------------------
    # VALIDATION ONLY: select among 2 base + 4 fusion candidates
    # ------------------------------------------------------------

    selection = select_fusion_with_grouped_cv(
        validation_split=validation_split,
        metadata_predictions=metadata_validation_predictions,
        image_predictions=image_validation_predictions,
        hypothesis=hypothesis,
        n_splits=n_splits,
        random_state=random_state,
        weight_grid_step=weight_grid_step,
        min_pr_auc_gain=min_pr_auc_gain,
    )

    candidate_table = (
        pd.DataFrame(
            [
                {
                    "method": method,
                    "candidate_type": (
                        "base_model" if method in BASE_CANDIDATES else "fusion"
                    ),
                    "pr_auc": selection["candidate_metrics"][method]["pr_auc"],
                    "roc_auc": selection["candidate_metrics"][method]["roc_auc"],
                    "brier_score": selection["candidate_metrics"][method][
                        "brier_score"
                    ],
                    "best_fusion": (method == selection["best_fusion_method"]),
                    "selected_overall": (method == selection["selected_method"]),
                }
                for method in ALL_CANDIDATES
            ]
        )
        .sort_values(
            by=[
                "pr_auc",
                "roc_auc",
                "brier_score",
            ],
            ascending=[
                False,
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    print("\nValidation candidate comparison:")
    print(candidate_table.to_string(index=False))

    print(
        "\nFusion replacement rule: "
        f"best base = {selection['best_base_method']} | "
        f"best fusion = {selection['best_fusion_method']} | "
        f"PR-AUC gain = {selection['fusion_pr_auc_gain']:.6f} | "
        f"minimum required = {selection['min_pr_auc_gain']:.6f}"
    )

    selected_method = selection["selected_method"]

    selected_is_fusion = selection["selected_is_fusion"]

    # ------------------------------------------------------------
    # Freeze the selected candidate BEFORE touching test
    # ------------------------------------------------------------

    saved_fusion: dict[str, Any] | None = None

    if selected_is_fusion:
        saved_fusion = save_selected_fusion_model(
            selection=selection,
            metadata_model_directory=metadata_model_directory,
            image_model_directory=image_model_directory,
            hypothesis=hypothesis,
            target_sensitivity=target_sensitivity,
            config_path=config_path,
            n_splits=n_splits,
            random_state=random_state,
            weight_grid_step=weight_grid_step,
        )

        final_model_directory = saved_fusion["model_directory"]

        selected_threshold = saved_fusion["selected_threshold"]

        selected_validation_predictions = saved_fusion["validation_predictions"]

    elif selected_method == "metadata_only":
        final_model_directory = metadata_model_directory

        selected_threshold = _frozen_threshold_from_metadata(metadata_model_directory)

        selected_validation_predictions = metadata_validation_predictions

    elif selected_method == "image_only":
        final_model_directory = image_model_directory

        selected_threshold = _frozen_threshold_from_metadata(image_model_directory)

        selected_validation_predictions = image_validation_predictions

    else:
        raise RuntimeError(f"Unsupported selected candidate: {selected_method}")

    if selected_threshold is None:
        # This should be unusual for the already-evaluated base models.
        merged_selected_validation = ev._validate_and_merge_split(
            df_predictions=selected_validation_predictions,
            df_split=validation_split,
            label_col=ev.DEFAULT_LABEL_COLUMNS[hypothesis],
            prediction_id_col="isic_id",
            probability_col="probability",
            split_id_col="isic_id",
            patient_id_col="patient_id",
            split_name="validation",
        )

        selected_threshold = ev.select_clinical_threshold(
            y_true=merged_selected_validation["y_true"].to_numpy(dtype=int),
            y_prob=merged_selected_validation["y_prob"].to_numpy(dtype=float),
            target_sensitivity=target_sensitivity,
        )

    # ============================================================
    # TEST: only now, after validation selection is frozen
    # ============================================================

    metadata_test_results = _ensure_base_test_evaluation(
        model_directory=metadata_model_directory,
        test_split=test_split,
        validation_split=validation_split,
        hypothesis=hypothesis,
        config_path=config_path,
        target_sensitivity=target_sensitivity,
        override=metadata_override,
        force_recompute_predictions=force_recompute_predictions,
        device=device,
    )

    image_test_results = _ensure_base_test_evaluation(
        model_directory=image_model_directory,
        test_split=test_split,
        validation_split=validation_split,
        hypothesis=hypothesis,
        config_path=config_path,
        target_sensitivity=target_sensitivity,
        override=image_override,
        force_recompute_predictions=force_recompute_predictions,
        device=device,
    )

    # ------------------------------------------------------------
    # Selected fusion test, only if a fusion actually won
    # ------------------------------------------------------------

    if selected_is_fusion:
        if _test_evaluation_is_complete(final_model_directory):
            selected_test_results = _load_existing_test_results(final_model_directory)

        else:
            metadata_test_predictions = pd.read_parquet(
                metadata_test_results["predictions_path"]
            )

            image_test_predictions = pd.read_parquet(
                image_test_results["predictions_path"]
            )

            test_base = test_split.loc[
                :,
                ["isic_id"],
            ].copy()

            test_base = test_base.merge(
                _canonicalize_predictions(metadata_test_predictions).rename(
                    columns={"probability": "metadata_probability"}
                ),
                on="isic_id",
                how="inner",
                validate="one_to_one",
            ).merge(
                _canonicalize_predictions(image_test_predictions).rename(
                    columns={"probability": "image_probability"}
                ),
                on="isic_id",
                how="inner",
                validate="one_to_one",
            )

            fusion_model = saved_fusion["fusion_model"]

            fused_test_probability = fusion_model.predict_probability(
                metadata_probability=test_base["metadata_probability"].to_numpy(
                    dtype=float
                ),
                image_probability=test_base["image_probability"].to_numpy(dtype=float),
            )

            fused_test_predictions = pd.DataFrame(
                {
                    "isic_id": test_base["isic_id"].astype(str),
                    "probability": fused_test_probability,
                }
            )

            # Root-level cache for the frozen fusion.
            fused_test_predictions.to_parquet(
                final_model_directory / TEST_PREDICTIONS_FILENAME,
                index=False,
            )

            selected_test_results = evaluate_test(
                df_test_predictions=fused_test_predictions,
                df_test=test_split,
                hypothesis=hypothesis,
                model_directory=final_model_directory,
                selected_threshold=selected_threshold,
            )

    elif selected_method == "metadata_only":
        selected_test_results = metadata_test_results

    else:
        selected_test_results = image_test_results

    # ============================================================
    # VALIDATION vs TEST COMPARISON
    # ============================================================

    metadata_threshold = _frozen_threshold_from_metadata(metadata_model_directory)

    image_threshold = _frozen_threshold_from_metadata(image_model_directory)

    if metadata_threshold is None:
        raise ValueError("Metadata base model has no frozen validation threshold.")

    if image_threshold is None:
        raise ValueError("Image base model has no frozen validation threshold.")

    metadata_validation_metrics = _validation_evaluation_row(
        predictions=metadata_validation_predictions,
        validation_split=validation_split,
        hypothesis=hypothesis,
        threshold=metadata_threshold,
    )

    image_validation_metrics = _validation_evaluation_row(
        predictions=image_validation_predictions,
        validation_split=validation_split,
        hypothesis=hypothesis,
        threshold=image_threshold,
    )

    selected_validation_metrics = _validation_evaluation_row(
        predictions=selected_validation_predictions,
        validation_split=validation_split,
        hypothesis=hypothesis,
        threshold=selected_threshold,
    )

    _, metadata_meta = _load_model_metadata(metadata_model_directory)
    _, image_meta = _load_model_metadata(image_model_directory)
    _, selected_meta = _load_model_metadata(final_model_directory)

    comparison_rows = [
        _comparison_row(
            role="metadata_base",
            model_name=metadata_meta["model"]["model_name"],
            validation_metrics=metadata_validation_metrics,
            test_metrics=_test_summary_row(metadata_test_results),
            selected=(selected_method == "metadata_only"),
        ),
        _comparison_row(
            role="image_base",
            model_name=image_meta["model"]["model_name"],
            validation_metrics=image_validation_metrics,
            test_metrics=_test_summary_row(image_test_results),
            selected=(selected_method == "image_only"),
        ),
        _comparison_row(
            role="selected_candidate",
            model_name=selected_meta["model"]["model_name"],
            validation_metrics=selected_validation_metrics,
            test_metrics=_test_summary_row(selected_test_results),
            selected=True,
        ),
    ]

    comparison = pd.DataFrame(comparison_rows)

    print("\nValidation vs test comparison:")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    return {
        "mode": ("fusion" if selected_is_fusion else "base_model_selected"),
        "hypothesis": hypothesis,
        "metadata_model_directory": metadata_model_directory,
        "image_model_directory": image_model_directory,
        "candidate_results": candidate_table,
        "best_base_method": selection["best_base_method"],
        "best_fusion_method": selection["best_fusion_method"],
        "fusion_pr_auc_gain": selection["fusion_pr_auc_gain"],
        "min_pr_auc_gain": selection["min_pr_auc_gain"],
        "selected_method": selected_method,
        "selected_fusion_method": (selected_method if selected_is_fusion else None),
        "selected_is_fusion": selected_is_fusion,
        "fusion_cv_results": (
            saved_fusion["cv_results"] if saved_fusion is not None else candidate_table
        ),
        "final_model_directory": final_model_directory,
        "selected_threshold": selected_threshold,
        "metadata_test_results": metadata_test_results,
        "image_test_results": image_test_results,
        "test_results": selected_test_results,
        "comparison": comparison,
    }
