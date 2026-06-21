from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from io import BytesIO
from pathlib import Path
from subprocess import run
from tqdm.auto import tqdm

import pandas as pd
from PIL import Image, UnidentifiedImageError

from skin_lesion_ai.utils.data_utils import (
    get_project_root,
    load_metadata_parquet,
    load_yaml_config,
    path,
)

"""
Generate preprocessed image Parquet shards from raw JPG images.

This script implements the first image preprocessing step defined after
EDA_IMG_01.ipynb. It loads the final preprocessed metadata generated after
the metadata EDA_02, matches each lesion/image identifier with its corresponding
JPG file, resizes all images to a fixed square size, and stores the resized
images as JPEG bytes in Parquet format.

The resulting dataset should be understood as the base homogenized image
dataset, not as the final normalized training dataset. Images are resized to
IMAGE_SIZE x IMAGE_SIZE x 3 and kept in uint8-compatible JPEG encoding with
pixel values in the original 0-255 range. Conversion to float32 and pixel
normalization should be performed later inside the model training pipeline.

The output is written to data/interim/images as timestamped Parquet shards
using the naming pattern:

    raw_images_preprocessed_<image_size>_<timestamp>_shard_<n>.parquet
    or if single shard / no shards
    raw_images_preprocessed_<image_size>_<timestamp>.parque

A JSON manifest is also saved with basic information about the generated files.
"""

SUPPORTED_IMAGE_SIZES = (96, 128, 136, 144, 160, 224)

# Default image size selected after EDA_IMG_01.
# 128, 136 and 144 are the main recommended candidates.
# 136 is used by default because it offers the best balance between
# retained image detail, storage cost and limited extreme upsampling.
# 128 is more conservative and useful for prototyping or memory constraints.
# 144 may be tested if slightly higher resolution is desired.
# 96 is efficient but may compress visual information too much.
# 160 and especially 224 are not recommended as default sizes because they
# require more upsampling and higher computational cost.
DEFAULT_IMAGE_SIZE = 136

# Number of rows per Parquet file.
# Set to a value larger than the dataset size to generate a single Parquet.
# Reduce this value if memory usage becomes problematic.
DEFAULT_SHARD_ROWS = 500_000

# JPEG quality used when storing resized images as bytes.
# 95 keeps high visual quality while reducing storage size compared with
# raw uint8 arrays.
DEFAULT_JPEG_QUALITY = 95

CONFIG_PATH = "configs/data_config.yaml"

FINAL_METADATA_FILENAME = "final_preprocessed_from_raw"
FINAL_METADATA_SCRIPT = "scripts/final_preprocess_data.py"
OUTPUT_BASENAME = "raw_images_preprocessed"


def load_or_generate_final_metadata(
    config_path: str | Path = CONFIG_PATH,
) -> pd.DataFrame:
    """Load final metadata or generate it if no timestamped file is available."""

    repo_root = get_project_root()
    script_path = repo_root / FINAL_METADATA_SCRIPT

    try:
        return load_metadata_parquet(
            stage="processed",
            filename=FINAL_METADATA_FILENAME,
            config_path=config_path,
            timestamp_flag=True,
        )

    except FileNotFoundError:
        run(
            [sys.executable, str(script_path)],
            cwd=str(repo_root),
            check=True,
        )

        return load_metadata_parquet(
            stage="processed",
            filename=FINAL_METADATA_FILENAME,
            config_path=config_path,
            timestamp_flag=True,
        )


def prepare_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Select and standardize the metadata columns required for image modelling."""

    df = df.copy()

    if "lesion_id" not in df.columns:
        if "isic_id" not in df.columns:
            raise KeyError("Expected either 'lesion_id' or 'isic_id' in metadata.")

        df = df.rename(columns={"isic_id": "lesion_id"})

    required_columns = [
        "lesion_id",
        "patient_id",
        "diagnostic_group",
        "target_biopsy",
        "target_malignant",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise KeyError(f"Missing required metadata columns: {missing_columns}")

    if df["lesion_id"].duplicated().any():
        duplicates = df.loc[df["lesion_id"].duplicated(), "lesion_id"].head(10).tolist()
        raise ValueError(f"Duplicated lesion_id values found. Examples: {duplicates}")

    return df.loc[:, required_columns].reset_index(drop=True)


def get_data_paths(config_path: str | Path = CONFIG_PATH) -> tuple[Path, Path]:
    """Get raw image input directory and interim image output directory."""

    config = load_yaml_config(path(config_path))

    raw_image_dir = path(config["paths"]["raw"]["images"])
    output_dir = path(config["paths"]["interim"]["images"])

    if not raw_image_dir.exists():
        raise FileNotFoundError(f"Raw image directory not found: {raw_image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    return raw_image_dir, output_dir


def add_image_paths(metadata: pd.DataFrame, raw_image_dir: Path) -> pd.DataFrame:
    """Attach expected JPG image paths to metadata and validate missing files."""

    metadata = metadata.copy()
    metadata["image_path"] = metadata["lesion_id"].map(
        lambda lesion_id: raw_image_dir / f"{lesion_id}.jpg"
    )

    missing_mask = ~metadata["image_path"].map(Path.exists)

    if missing_mask.any():
        missing_examples = metadata.loc[missing_mask, "lesion_id"].head(10).tolist()
        n_missing = int(missing_mask.sum())

        raise FileNotFoundError(
            f"{n_missing} expected JPG files were not found. "
            f"Examples: {missing_examples}"
        )

    return metadata


def resize_image_to_jpeg_bytes(
    image_path: Path,
    image_size: int,
    jpeg_quality: int,
) -> tuple[bytes, float]:
    """Read one image, resize it to RGB JPEG bytes, and compute scale factor."""

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")

            original_width, original_height = img.size

            if original_width != original_height:
                raise ValueError(
                    f"Expected square image, got {original_width}x{original_height}: "
                    f"{image_path}"
                )

            scale_factor = image_size / original_width

            resized_img = img.resize(
                (image_size, image_size),
                resample=Image.Resampling.LANCZOS,
            )

            buffer = BytesIO()
            resized_img.save(
                buffer,
                format="JPEG",
                quality=jpeg_quality,
                optimize=True,
            )

            return buffer.getvalue(), scale_factor

    except UnidentifiedImageError as exc:
        raise UnidentifiedImageError(
            f"Could not read image file: {image_path}"
        ) from exc


def build_image_shard(
    metadata_shard: pd.DataFrame,
    image_size: int,
    jpeg_quality: int,
) -> pd.DataFrame:
    """Create one output dataframe shard with resized JPEG bytes."""

    records = []

    for row in tqdm(
        metadata_shard.itertuples(index=False),
        total=len(metadata_shard),
        desc="Resizing images",
    ):
        image_bytes, scale_factor = resize_image_to_jpeg_bytes(
            image_path=row.image_path,
            image_size=image_size,
            jpeg_quality=jpeg_quality,
        )

        records.append(
            {
                "lesion_id": row.lesion_id,
                "patient_id": row.patient_id,
                "diagnostic_group": row.diagnostic_group,
                "target_biopsy": row.target_biopsy,
                "target_malignant": row.target_malignant,
                "image": image_bytes,
                "scale_factor": scale_factor,
            }
        )

    return pd.DataFrame.from_records(records)


def save_image_parquet_shards(
    metadata: pd.DataFrame,
    output_dir: Path,
    image_size: int,
    shard_rows: int,
    jpeg_quality: int,
    single_file: bool = False,
) -> list[Path]:
    """Generate and save image Parquet files."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_paths: list[Path] = []

    n_rows = len(metadata)

    if n_rows == 0:
        raise ValueError("Metadata is empty. No images to process.")

    if single_file:
        shard_rows = n_rows

    if shard_rows <= 0:
        raise ValueError("shard_rows must be greater than 0.")

    is_final_single_file = single_file or n_rows <= shard_rows

    for shard_idx, start in enumerate(range(0, n_rows, shard_rows)):
        end = min(start + shard_rows, n_rows)
        metadata_shard = metadata.iloc[start:end].copy()

        print(f"Processing rows {start:,} to {end:,} of {n_rows:,}...")

        image_shard = build_image_shard(
            metadata_shard=metadata_shard,
            image_size=image_size,
            jpeg_quality=jpeg_quality,
        )

        if is_final_single_file:
            filename = f"{OUTPUT_BASENAME}_{image_size}_{timestamp}.parquet"
        else:
            filename = (
                f"{OUTPUT_BASENAME}_{image_size}_{timestamp}"
                f"_shard_{shard_idx:03d}.parquet"
            )

        output_path = output_dir / filename
        image_shard.to_parquet(output_path, index=False)

        output_paths.append(output_path)

        print(f"Saved: {output_path}")

    manifest = {
        "created_at": timestamp,
        "image_size": image_size,
        "jpeg_quality": jpeg_quality,
        "n_rows": n_rows,
        "single_file": is_final_single_file,
        "shard_rows": shard_rows,
        "n_files": len(output_paths),
        "files": [str(p) for p in output_paths],
    }

    manifest_path = (
        output_dir / f"{OUTPUT_BASENAME}_{image_size}_{timestamp}_manifest.json"
    )

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print(f"Saved manifest: {manifest_path}")

    return output_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate resized image Parquet files from raw JPG images."
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        choices=SUPPORTED_IMAGE_SIZES,
        help=f"Final square image size. Default: {DEFAULT_IMAGE_SIZE}.",
    )

    parser.add_argument(
        "--shard-rows",
        type=int,
        default=DEFAULT_SHARD_ROWS,
        help=f"Number of rows per Parquet shard. Default: {DEFAULT_SHARD_ROWS}.",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=DEFAULT_JPEG_QUALITY,
        help=f"JPEG quality for resized image bytes. Default: {DEFAULT_JPEG_QUALITY}.",
    )

    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Save all rows into a single Parquet file instead of shards.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for testing.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the full image preprocessing pipeline."""

    args = parse_args()

    metadata = load_or_generate_final_metadata()
    metadata = prepare_metadata(metadata)

    if args.limit is not None:
        metadata = metadata.head(args.limit).copy()

    raw_image_dir, output_dir = get_data_paths()
    metadata = add_image_paths(metadata, raw_image_dir)

    print(f"Rows to process: {len(metadata):,}")
    print(f"Raw image directory: {raw_image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"JPEG quality: {args.jpeg_quality}")
    print(f"Single file: {args.single_file}")

    save_image_parquet_shards(
        metadata=metadata,
        output_dir=output_dir,
        image_size=args.image_size,
        shard_rows=args.shard_rows,
        jpeg_quality=args.jpeg_quality,
        single_file=args.single_file,
    )


if __name__ == "__main__":
    main()
