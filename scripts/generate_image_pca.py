from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from PIL import Image, UnidentifiedImageError
from sklearn.decomposition import IncrementalPCA
from tqdm.auto import tqdm

from skin_lesion_ai.utils.data_utils import (
    load_yaml_config,
    path,
)

"""
Generate PCA features from the preprocessed 136x136 lesion images.

The PCA is fitted exclusively on the training split of the selected hypothesis
and the fitted transformation is then applied unchanged to train, validation
and test images. This prevents information leakage from validation or test data.

Image preprocessing follows the PCA exploratory notebook:
    1. decode JPEG bytes as RGB;
    2. convert pixels to float32;
    3. scale pixels from 0-255 to 0-1;
    4. flatten each image to one vector;
    5. apply IncrementalPCA.

No per-pixel StandardScaler is used. IncrementalPCA performs feature centering
internally, and the common division by 255 only changes the global numerical
scale of the raw pixels.

Outputs are written to data/processed/images as timestamped Parquet shards with
one row per lesion and columns:

    lesion_id, patient_id, feature_0000, ..., feature_0127

The fitted PCA object is saved with joblib, and a JSON manifest records the
hypothesis, preprocessing, source files, explained variance and generated files.
"""

CONFIG_PATH = "configs/data_config.yaml"
INPUT_BASENAME = "raw_images_preprocessed"
OUTPUT_BASENAME = "image_pca"

DEFAULT_IMAGE_SIZE = 136
DEFAULT_N_COMPONENTS = 128
DEFAULT_PCA_BATCH_SIZE = 512
DEFAULT_TRANSFORM_BATCH_SIZE = 512
DEFAULT_IO_BATCH_SIZE = 2048
DEFAULT_OUTPUT_SHARD_ROWS = 20_000

SPLIT_NAMES = ("train", "val", "test")

HYPOTHESIS_CONFIG = {
    1: {
        "target": "target_biopsy",
    },
    2: {
        "target": "target_malignant",
    },
}


# ---------------------------------------------------------------------
# Paths and manifests
# ---------------------------------------------------------------------


def get_directories(
    config_path: str | Path = CONFIG_PATH,
) -> tuple[Path, Path, Path]:
    """Return image input, image output and processed metadata directories."""

    config = load_yaml_config(path(config_path))

    input_dir = path(config["paths"]["interim"]["images"])
    output_dir = path(config["paths"]["processed"]["images"])
    metadata_dir = path(config["paths"]["processed"]["metadata"])

    if not input_dir.exists():
        raise FileNotFoundError(f"Interim image directory not found: {input_dir}")

    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"Processed metadata directory not found: {metadata_dir}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    return input_dir, output_dir, metadata_dir


def find_latest_image_manifest(
    input_dir: Path,
    image_size: int,
) -> Path:
    """Find the latest preprocessing manifest for the selected image size."""

    pattern = f"{INPUT_BASENAME}_{image_size}_*_manifest.json"
    manifests = sorted(input_dir.glob(pattern))

    if not manifests:
        raise FileNotFoundError(
            f"No image preprocessing manifest found for {image_size}x{image_size} "
            f"in {input_dir}. Expected pattern: {pattern}"
        )

    return manifests[-1]


def load_image_manifest(manifest_path: Path) -> dict:
    """Load and minimally validate an image preprocessing manifest."""

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    required_keys = {"image_size", "n_rows", "files"}
    missing_keys = required_keys.difference(manifest)

    if missing_keys:
        raise ValueError(
            f"Input manifest is missing required keys: {sorted(missing_keys)}"
        )

    if not manifest["files"]:
        raise ValueError("Input manifest does not contain any Parquet files.")

    return manifest


def resolve_input_files(
    manifest: dict,
    input_dir: Path,
) -> list[Path]:
    """Resolve Parquet paths stored in the preprocessing manifest."""

    resolved_files: list[Path] = []

    for raw_file in manifest["files"]:
        candidate = Path(raw_file)

        if candidate.exists():
            resolved_files.append(candidate)
            continue

        fallback = input_dir / candidate.name

        if fallback.exists():
            resolved_files.append(fallback)
            continue

        raise FileNotFoundError(
            f"Parquet file listed in manifest was not found: {raw_file}"
        )

    return resolved_files


def find_latest_split_file(
    metadata_dir: Path,
    split_name: str,
    hypothesis: int,
) -> Path:
    """Find the latest timestamped metadata file for one split."""

    base_name = f"{split_name}_split_h{hypothesis}"
    pattern = f"{base_name}_*.parquet"
    timestamp_regex = re.compile(
        rf"^{re.escape(base_name)}_(\d{{8}}_\d{{6}})\.parquet$"
    )

    candidates: list[tuple[str, Path]] = []

    for candidate in metadata_dir.glob(pattern):
        match = timestamp_regex.match(candidate.name)

        if match:
            candidates.append((match.group(1), candidate))

    if not candidates:
        raise FileNotFoundError(
            f"No metadata split found for H{hypothesis}/{split_name} in "
            f"{metadata_dir}. Expected pattern: {pattern}"
        )

    return max(candidates, key=lambda item: item[0])[1]


def load_split_metadata(
    metadata_dir: Path,
    hypothesis: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """Load the latest train, validation and test metadata for one hypothesis."""

    target = HYPOTHESIS_CONFIG[hypothesis]["target"]
    split_frames: dict[str, pd.DataFrame] = {}
    split_paths: dict[str, Path] = {}

    for split_name in SPLIT_NAMES:
        split_path = find_latest_split_file(
            metadata_dir=metadata_dir,
            split_name=split_name,
            hypothesis=hypothesis,
        )
        split_df = pd.read_parquet(split_path)

        required_columns = {"isic_id", "patient_id", target}
        missing_columns = required_columns.difference(split_df.columns)

        if missing_columns:
            raise ValueError(
                f"{split_path.name} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        split_df = (
            split_df.loc[:, ["isic_id", "patient_id", target]]
            .rename(columns={"isic_id": "lesion_id"})
            .copy()
        )

        split_df["lesion_id"] = split_df["lesion_id"].astype(str)
        split_df["patient_id"] = split_df["patient_id"].astype(str)

        if split_df["lesion_id"].isna().any():
            raise ValueError(f"{split_path.name} contains missing lesion_id values.")

        if not split_df["lesion_id"].is_unique:
            raise ValueError(f"{split_path.name} contains duplicated lesion_id values.")

        split_frames[split_name] = split_df
        split_paths[split_name] = split_path

    validate_split_separation(split_frames)

    return split_frames, split_paths


def validate_split_separation(
    split_frames: dict[str, pd.DataFrame],
) -> None:
    """Validate that lesion identifiers do not overlap across splits."""

    split_ids = {
        split_name: set(split_df["lesion_id"])
        for split_name, split_df in split_frames.items()
    }

    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_ids[first].intersection(split_ids[second])

        if overlap:
            examples = sorted(overlap)[:10]
            raise ValueError(
                f"Lesion overlap between {first} and {second}. Examples: {examples}"
            )


# ---------------------------------------------------------------------
# Image decoding and raw-pixel preparation
# ---------------------------------------------------------------------


def decode_image_to_flat_pixels(
    image_bytes: bytes,
    image_size: int,
) -> np.ndarray:
    """Decode one JPEG image and return flattened RGB pixels in the 0-1 range."""

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")

            if image.size != (image_size, image_size):
                raise ValueError(
                    f"Expected {image_size}x{image_size}, got "
                    f"{image.size[0]}x{image.size[1]}."
                )

            array = np.asarray(image, dtype=np.float32) / 255.0

    except UnidentifiedImageError as exc:
        raise UnidentifiedImageError(
            "Could not decode one of the JPEG byte sequences."
        ) from exc

    return array.reshape(-1)


def validate_image_columns(input_files: list[Path]) -> None:
    """Validate that every image Parquet file contains the required columns."""

    required_columns = {"lesion_id", "patient_id", "image"}

    for input_file in input_files:
        parquet_file = pq.ParquetFile(input_file)
        available_columns = set(parquet_file.schema.names)
        missing_columns = required_columns.difference(available_columns)

        if missing_columns:
            raise ValueError(
                f"{input_file.name} is missing required columns: "
                f"{sorted(missing_columns)}"
            )


# ---------------------------------------------------------------------
# PCA fitting
# ---------------------------------------------------------------------


def select_pca_fit_metadata(
    train_df: pd.DataFrame,
    n_components: int,
) -> tuple[pd.DataFrame, str]:
    """Use the complete training split to fit PCA."""

    if len(train_df) < n_components:
        raise ValueError(
            f"Training split contains {len(train_df):,} rows, but PCA requires "
            f"at least {n_components:,} rows."
        )

    return train_df.copy(), "full_train"


def build_pca_fit_batch_sizes(
    n_rows: int,
    target_batch_size: int,
    n_components: int,
) -> list[int]:
    """Build fit batch sizes while ensuring every batch has >= n_components rows."""

    if n_rows < n_components:
        raise ValueError("PCA fitting requires at least n_components rows.")

    if target_batch_size < n_components:
        raise ValueError("pca_batch_size must be >= n_components.")

    if n_rows <= target_batch_size:
        return [n_rows]

    n_full_batches, remainder = divmod(n_rows, target_batch_size)

    if remainder == 0:
        return [target_batch_size] * n_full_batches

    if remainder >= n_components:
        return [target_batch_size] * n_full_batches + [remainder]

    return [target_batch_size] * (n_full_batches - 1) + [target_batch_size + remainder]


def iter_selected_fit_batches(
    input_files: list[Path],
    selected_lesion_ids: set[str],
    image_size: int,
    pca_batch_size: int,
    n_components: int,
    io_batch_size: int,
) -> Iterator[np.ndarray]:
    """Yield raw-pixel batches for the lesions used to fit IncrementalPCA."""

    batch_sizes = build_pca_fit_batch_sizes(
        n_rows=len(selected_lesion_ids),
        target_batch_size=pca_batch_size,
        n_components=n_components,
    )

    batch_idx = 0
    target_size = batch_sizes[batch_idx]
    pixel_buffer: list[np.ndarray] = []
    found_ids: set[str] = set()

    for input_file in input_files:
        parquet_file = pq.ParquetFile(input_file)

        for record_batch in parquet_file.iter_batches(
            batch_size=io_batch_size,
            columns=["lesion_id", "image"],
        ):
            batch_df = record_batch.to_pandas()

            for lesion_id, image_bytes in batch_df.itertuples(index=False, name=None):
                lesion_id = str(lesion_id)

                if lesion_id not in selected_lesion_ids:
                    continue

                if lesion_id in found_ids:
                    raise ValueError(f"Duplicated lesion_id found: {lesion_id}")

                found_ids.add(lesion_id)
                pixel_buffer.append(
                    decode_image_to_flat_pixels(
                        image_bytes=image_bytes,
                        image_size=image_size,
                    )
                )

                if len(pixel_buffer) == target_size:
                    yield np.stack(pixel_buffer).astype(np.float32, copy=False)
                    pixel_buffer = []
                    batch_idx += 1

                    if batch_idx < len(batch_sizes):
                        target_size = batch_sizes[batch_idx]

    missing_ids = selected_lesion_ids.difference(found_ids)

    if missing_ids:
        examples = sorted(missing_ids)[:10]
        raise ValueError(
            f"{len(missing_ids):,} PCA-fit lesions were not found in the "
            f"preprocessed image data. Examples: {examples}"
        )

    if pixel_buffer:
        raise RuntimeError(
            f"Unexpected final PCA pixel buffer with {len(pixel_buffer)} rows."
        )

    if batch_idx != len(batch_sizes):
        raise RuntimeError(
            f"Expected {len(batch_sizes)} PCA batches, generated {batch_idx}."
        )


def fit_incremental_pca(
    input_files: list[Path],
    fit_df: pd.DataFrame,
    image_size: int,
    n_components: int,
    pca_batch_size: int,
    io_batch_size: int,
) -> IncrementalPCA:
    """Fit IncrementalPCA exclusively on the selected training images."""

    fit_ids = set(fit_df["lesion_id"].astype(str))

    pca = IncrementalPCA(
        n_components=n_components,
        batch_size=pca_batch_size,
    )

    progress = tqdm(
        total=len(fit_ids),
        desc="Fitting IncrementalPCA",
        unit="image",
    )

    for pixel_batch in iter_selected_fit_batches(
        input_files=input_files,
        selected_lesion_ids=fit_ids,
        image_size=image_size,
        pca_batch_size=pca_batch_size,
        n_components=n_components,
        io_batch_size=io_batch_size,
    ):
        pca.partial_fit(pixel_batch)
        progress.update(len(pixel_batch))

    progress.close()

    if int(pca.n_components_) != n_components:
        raise RuntimeError(
            f"Expected {n_components} PCA components, got {pca.n_components_}."
        )

    return pca


# ---------------------------------------------------------------------
# PCA transformation and output
# ---------------------------------------------------------------------


def build_feature_dataframe(
    lesion_ids: list[str],
    patient_ids: list[str],
    features: np.ndarray,
) -> pd.DataFrame:
    """Create one tabular dataframe with PCA feature columns."""

    feature_columns = [f"feature_{idx:04d}" for idx in range(features.shape[1])]

    feature_df = pd.DataFrame(
        features,
        columns=feature_columns,
        dtype=np.float32,
    )

    feature_df.insert(0, "patient_id", patient_ids)
    feature_df.insert(0, "lesion_id", lesion_ids)

    return feature_df


def save_pca_shard(
    lesion_ids: list[str],
    patient_ids: list[str],
    feature_chunks: list[np.ndarray],
    output_dir: Path,
    hypothesis: int,
    image_size: int,
    n_components: int,
    split_name: str,
    timestamp: str,
    shard_idx: int,
) -> Path:
    """Concatenate buffered PCA features and save one Parquet shard."""

    features = np.concatenate(feature_chunks, axis=0)

    feature_df = build_feature_dataframe(
        lesion_ids=lesion_ids,
        patient_ids=patient_ids,
        features=features,
    )

    filename = (
        f"{OUTPUT_BASENAME}_h{hypothesis}_{image_size}px_{n_components}c_"
        f"{timestamp}_{split_name}_shard_{shard_idx:03d}.parquet"
    )
    output_path = output_dir / filename

    feature_df.to_parquet(output_path, index=False)

    return output_path


def transform_and_save_splits(
    input_files: list[Path],
    split_frames: dict[str, pd.DataFrame],
    pca: IncrementalPCA,
    output_dir: Path,
    hypothesis: int,
    image_size: int,
    n_components: int,
    transform_batch_size: int,
    io_batch_size: int,
    output_shard_rows: int,
    timestamp: str,
) -> tuple[dict[str, list[Path]], dict[str, int]]:
    """Transform all split images in one scan and save split-specific shards."""

    split_lookup: dict[str, str] = {}
    expected_rows: dict[str, int] = {}

    for split_name, split_df in split_frames.items():
        expected_rows[split_name] = len(split_df)

        for lesion_id in split_df["lesion_id"].astype(str):
            split_lookup[lesion_id] = split_name

    total_expected = sum(expected_rows.values())

    output_paths: dict[str, list[Path]] = {split_name: [] for split_name in SPLIT_NAMES}
    processed_rows: dict[str, int] = {split_name: 0 for split_name in SPLIT_NAMES}
    shard_idx: dict[str, int] = {split_name: 0 for split_name in SPLIT_NAMES}

    buffered_lesion_ids: dict[str, list[str]] = {
        split_name: [] for split_name in SPLIT_NAMES
    }
    buffered_patient_ids: dict[str, list[str]] = {
        split_name: [] for split_name in SPLIT_NAMES
    }
    buffered_features: dict[str, list[np.ndarray]] = {
        split_name: [] for split_name in SPLIT_NAMES
    }
    buffered_rows: dict[str, int] = {split_name: 0 for split_name in SPLIT_NAMES}

    transform_lesion_ids: list[str] = []
    transform_patient_ids: list[str] = []
    transform_split_names: list[str] = []
    transform_pixels: list[np.ndarray] = []

    found_ids: set[str] = set()

    progress = tqdm(
        total=total_expected,
        desc=f"Transforming H{hypothesis} images with PCA",
        unit="image",
    )

    def flush_split(split_name: str) -> None:
        if buffered_rows[split_name] == 0:
            return

        output_path = save_pca_shard(
            lesion_ids=buffered_lesion_ids[split_name],
            patient_ids=buffered_patient_ids[split_name],
            feature_chunks=buffered_features[split_name],
            output_dir=output_dir,
            hypothesis=hypothesis,
            image_size=image_size,
            n_components=n_components,
            split_name=split_name,
            timestamp=timestamp,
            shard_idx=shard_idx[split_name],
        )
        output_paths[split_name].append(output_path)

        buffered_lesion_ids[split_name] = []
        buffered_patient_ids[split_name] = []
        buffered_features[split_name] = []
        buffered_rows[split_name] = 0
        shard_idx[split_name] += 1

    def process_transform_buffer() -> None:
        if not transform_pixels:
            return

        pixel_batch = np.stack(transform_pixels).astype(np.float32, copy=False)
        transformed = pca.transform(pixel_batch).astype(np.float32, copy=False)

        split_array = np.asarray(transform_split_names, dtype=object)
        lesion_array = np.asarray(transform_lesion_ids, dtype=object)
        patient_array = np.asarray(transform_patient_ids, dtype=object)

        for split_name in SPLIT_NAMES:
            mask = split_array == split_name

            if not np.any(mask):
                continue

            split_features = transformed[mask]
            split_lesion_ids = lesion_array[mask].tolist()
            split_patient_ids = patient_array[mask].tolist()

            buffered_lesion_ids[split_name].extend(split_lesion_ids)
            buffered_patient_ids[split_name].extend(split_patient_ids)
            buffered_features[split_name].append(split_features)

            current_rows = len(split_lesion_ids)
            buffered_rows[split_name] += current_rows
            processed_rows[split_name] += current_rows

            if buffered_rows[split_name] >= output_shard_rows:
                flush_split(split_name)

        progress.update(len(transform_pixels))

        transform_lesion_ids.clear()
        transform_patient_ids.clear()
        transform_split_names.clear()
        transform_pixels.clear()

    for input_file in input_files:
        parquet_file = pq.ParquetFile(input_file)

        for record_batch in parquet_file.iter_batches(
            batch_size=io_batch_size,
            columns=["lesion_id", "patient_id", "image"],
        ):
            batch_df = record_batch.to_pandas()

            for lesion_id, patient_id, image_bytes in batch_df.itertuples(
                index=False,
                name=None,
            ):
                lesion_id = str(lesion_id)
                split_name = split_lookup.get(lesion_id)

                if split_name is None:
                    continue

                if lesion_id in found_ids:
                    raise ValueError(f"Duplicated lesion_id found: {lesion_id}")

                found_ids.add(lesion_id)
                transform_lesion_ids.append(lesion_id)
                transform_patient_ids.append(str(patient_id))
                transform_split_names.append(split_name)
                transform_pixels.append(
                    decode_image_to_flat_pixels(
                        image_bytes=image_bytes,
                        image_size=image_size,
                    )
                )

                if len(transform_pixels) == transform_batch_size:
                    process_transform_buffer()

    process_transform_buffer()
    progress.close()

    for split_name in SPLIT_NAMES:
        flush_split(split_name)

    expected_ids = set(split_lookup)
    missing_ids = expected_ids.difference(found_ids)

    if missing_ids:
        examples = sorted(missing_ids)[:10]
        raise ValueError(
            f"{len(missing_ids):,} split lesions were not found in the "
            f"preprocessed image data. Examples: {examples}"
        )

    for split_name in SPLIT_NAMES:
        if processed_rows[split_name] != expected_rows[split_name]:
            raise RuntimeError(
                f"{split_name}: expected {expected_rows[split_name]:,} rows, "
                f"but transformed {processed_rows[split_name]:,}."
            )

    return output_paths, processed_rows


def save_pca_object(
    pca: IncrementalPCA,
    output_dir: Path,
    hypothesis: int,
    image_size: int,
    n_components: int,
    timestamp: str,
) -> Path:
    """Serialize the fitted PCA object with joblib."""

    pca_path = output_dir / (
        f"{OUTPUT_BASENAME}_h{hypothesis}_{image_size}px_{n_components}c_"
        f"{timestamp}.joblib"
    )

    joblib.dump(pca, pca_path)

    return pca_path


def save_pca_manifest(
    output_dir: Path,
    hypothesis: int,
    target: str,
    image_size: int,
    n_components: int,
    source_manifest_path: Path,
    split_paths: dict[str, Path],
    output_paths: dict[str, list[Path]],
    processed_rows: dict[str, int],
    pca_path: Path,
    pca: IncrementalPCA,
    pca_fit_rows: int,
    pca_fit_strategy: str,
    pca_batch_size: int,
    transform_batch_size: int,
    io_batch_size: int,
    output_shard_rows: int,
    timestamp: str,
) -> Path:
    """Save metadata describing one PCA feature-generation run."""

    all_output_paths = [
        str(output_path)
        for split_name in SPLIT_NAMES
        for output_path in output_paths[split_name]
    ]

    manifest = {
        "created_at": timestamp,
        "representation": "raw_pixels_pca",
        "hypothesis": f"H{hypothesis}",
        "target": target,
        "image_size": image_size,
        "raw_feature_dim": image_size * image_size * 3,
        "n_components": n_components,
        "input_normalization": {
            "type": "scale_0_1",
            "divisor": 255.0,
            "channel_order": "RGB",
            "flattened": True,
        },
        "pca_centering": True,
        "pca_fit_split": "train",
        "pca_fit_strategy": pca_fit_strategy,
        "pca_fit_rows": pca_fit_rows,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "source_image_manifest": str(source_manifest_path),
        "source_split_files": {
            split_name: str(split_paths[split_name]) for split_name in SPLIT_NAMES
        },
        "n_rows": int(sum(processed_rows.values())),
        "n_rows_by_split": {
            split_name: int(processed_rows[split_name]) for split_name in SPLIT_NAMES
        },
        "n_files": len(all_output_paths),
        "files": all_output_paths,
        "files_by_split": {
            split_name: [str(p) for p in output_paths[split_name]]
            for split_name in SPLIT_NAMES
        },
        "pca_object": str(pca_path),
        "pca_batch_size": pca_batch_size,
        "transform_batch_size": transform_batch_size,
        "io_batch_size": io_batch_size,
        "output_shard_rows": output_shard_rows,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }

    manifest_path = output_dir / (
        f"{OUTPUT_BASENAME}_h{hypothesis}_{image_size}px_{n_components}c_"
        f"{timestamp}_manifest.json"
    )

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return manifest_path


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Generate training-fitted PCA features from preprocessed lesion images."
        )
    )

    parser.add_argument(
        "--hypothesis",
        type=int,
        required=True,
        choices=(1, 2),
        help="Modelling hypothesis whose training split is used to fit PCA.",
    )

    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_N_COMPONENTS,
        help=f"Number of PCA components. Default: {DEFAULT_N_COMPONENTS}.",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help=f"Expected preprocessed image size. Default: {DEFAULT_IMAGE_SIZE}.",
    )

    parser.add_argument(
        "--pca-batch-size",
        type=int,
        default=DEFAULT_PCA_BATCH_SIZE,
        help=(
            "Number of images used in each IncrementalPCA partial_fit update. "
            f"Default: {DEFAULT_PCA_BATCH_SIZE}."
        ),
    )

    parser.add_argument(
        "--transform-batch-size",
        type=int,
        default=DEFAULT_TRANSFORM_BATCH_SIZE,
        help=(
            "Number of images transformed with the fitted PCA at once. "
            f"Default: {DEFAULT_TRANSFORM_BATCH_SIZE}."
        ),
    )

    parser.add_argument(
        "--io-batch-size",
        type=int,
        default=DEFAULT_IO_BATCH_SIZE,
        help=(
            "Rows read at once from the image Parquet files. "
            f"Default: {DEFAULT_IO_BATCH_SIZE}."
        ),
    )

    parser.add_argument(
        "--output-shard-rows",
        type=int,
        default=DEFAULT_OUTPUT_SHARD_ROWS,
        help=(
            "Approximate rows per output Parquet shard. "
            f"Default: {DEFAULT_OUTPUT_SHARD_ROWS}."
        ),
    )

    parser.add_argument(
        "--input-manifest",
        type=str,
        default=None,
        help=(
            "Optional exact image preprocessing manifest path. If omitted, the "
            "latest manifest matching the selected image size is used."
        ),
    )

    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate positive numeric CLI arguments."""

    for argument_name in (
        "image_size",
        "n_components",
        "pca_batch_size",
        "transform_batch_size",
        "io_batch_size",
        "output_shard_rows",
    ):
        value = getattr(args, argument_name)

        if value <= 0:
            raise ValueError(f"{argument_name} must be greater than 0.")

    if args.pca_batch_size < args.n_components:
        raise ValueError("pca_batch_size must be >= n_components.")


def main() -> None:
    """Run the complete PCA feature-generation pipeline."""

    args = parse_args()
    validate_args(args)

    target = HYPOTHESIS_CONFIG[args.hypothesis]["target"]
    input_dir, output_dir, metadata_dir = get_directories()

    if args.input_manifest is None:
        source_manifest_path = find_latest_image_manifest(
            input_dir=input_dir,
            image_size=args.image_size,
        )
    else:
        source_manifest_path = Path(args.input_manifest).expanduser().resolve()

        if not source_manifest_path.exists():
            raise FileNotFoundError(f"Input manifest not found: {source_manifest_path}")

    source_manifest = load_image_manifest(source_manifest_path)

    if int(source_manifest["image_size"]) != args.image_size:
        raise ValueError(
            f"Manifest image size ({source_manifest['image_size']}) does not "
            f"match --image-size ({args.image_size})."
        )

    input_files = resolve_input_files(
        manifest=source_manifest,
        input_dir=input_dir,
    )
    validate_image_columns(input_files)

    split_frames, split_paths = load_split_metadata(
        metadata_dir=metadata_dir,
        hypothesis=args.hypothesis,
    )

    fit_df, fit_strategy = select_pca_fit_metadata(
        train_df=split_frames["train"],
        n_components=args.n_components,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"Hypothesis: H{args.hypothesis}")
    print(f"Target: {target}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Raw pixel features: {args.image_size * args.image_size * 3:,}")
    print(f"PCA components: {args.n_components:,}")
    print(f"PCA fit strategy: {fit_strategy}")
    print(f"PCA fit rows: {len(fit_df):,}")
    print(f"Source image manifest: {source_manifest_path}")
    print(f"Input image Parquet files: {len(input_files)}")

    for split_name in SPLIT_NAMES:
        print(
            f"{split_name}: {len(split_frames[split_name]):,} rows | "
            f"{split_paths[split_name]}"
        )

    print(f"Output directory: {output_dir}")

    pca = fit_incremental_pca(
        input_files=input_files,
        fit_df=fit_df,
        image_size=args.image_size,
        n_components=args.n_components,
        pca_batch_size=args.pca_batch_size,
        io_batch_size=args.io_batch_size,
    )

    explained_variance = float(np.sum(pca.explained_variance_ratio_))

    print("\nPCA fitting completed.")
    print(
        f"Explained variance ({args.n_components} components): {explained_variance:.4%}"
    )

    pca_path = save_pca_object(
        pca=pca,
        output_dir=output_dir,
        hypothesis=args.hypothesis,
        image_size=args.image_size,
        n_components=args.n_components,
        timestamp=timestamp,
    )

    output_paths, processed_rows = transform_and_save_splits(
        input_files=input_files,
        split_frames=split_frames,
        pca=pca,
        output_dir=output_dir,
        hypothesis=args.hypothesis,
        image_size=args.image_size,
        n_components=args.n_components,
        transform_batch_size=args.transform_batch_size,
        io_batch_size=args.io_batch_size,
        output_shard_rows=args.output_shard_rows,
        timestamp=timestamp,
    )

    manifest_path = save_pca_manifest(
        output_dir=output_dir,
        hypothesis=args.hypothesis,
        target=target,
        image_size=args.image_size,
        n_components=args.n_components,
        source_manifest_path=source_manifest_path,
        split_paths=split_paths,
        output_paths=output_paths,
        processed_rows=processed_rows,
        pca_path=pca_path,
        pca=pca,
        pca_fit_rows=len(fit_df),
        pca_fit_strategy=fit_strategy,
        pca_batch_size=args.pca_batch_size,
        transform_batch_size=args.transform_batch_size,
        io_batch_size=args.io_batch_size,
        output_shard_rows=args.output_shard_rows,
        timestamp=timestamp,
    )

    print("\nPCA feature generation completed.")

    for split_name in SPLIT_NAMES:
        print(
            f"{split_name}: {processed_rows[split_name]:,} rows | "
            f"{len(output_paths[split_name])} shard(s)"
        )

    print(f"PCA object: {pca_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
