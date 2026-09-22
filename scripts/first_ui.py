from __future__ import annotations
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from skin_lesion_ai.inference import final_models as fm
from skin_lesion_ai.utils.data_utils import get_project_root, load_yaml_config


# RUN SCRIPT USINNG ``` uv run streamlit run scripts/first_ui.py ````

# ============================================================
# Configuration
# ============================================================

CONFIG_PATH = "configs/data_config.yaml"
LOGO_PATH = "configs/logo.png"

LEGAL_NOTICE = "© Todos los derechos reservados."

PROJECT_TITLE = (
    "Clasificación de lesiones cutáneas mediante imágenes macroscópicas "
    "y metadatos clínicos"
)
PROJECT_SUBTITLE = (
    "Prueba de concepto para la priorización clínica y el estudio de "
    "malignidad entre lesiones biopsiadas."
)
PROJECT_COURSE = "Curso 2025–2026"
PROJECT_AUTHORS = (
    "Juan Guillermo Cardona Urrego, Pau Peracaula López-Amor y Carles Raich Bros"
)
PROJECT_TUTOR = "Miguel Ángel de la Llave Montiel"

# Only the final TFM artefact names are hardcoded.
MODEL_NAMES = {
    "h1_metadata": "xgboost_metadata_h1",
    "h1_image": "resnet50_image_h1_full_train_imagenet",
    "h1_fusion": ("fusion_svm_metadata_h1__resnet50_image_h1_full_train_imagenet"),
    "h2_metadata": "xgboost_metadata_h2",
    "h2_image": "efficientnet_b0_image_h2_full_train",
}

# Six possible user input combinations, but only five distinct artefacts.
MODEL_ROUTES = {
    (1, True, False): MODEL_NAMES["h1_metadata"],
    (1, False, True): MODEL_NAMES["h1_image"],
    (1, True, True): MODEL_NAMES["h1_fusion"],
    (2, True, False): MODEL_NAMES["h2_metadata"],
    (2, False, True): MODEL_NAMES["h2_image"],
    (2, True, True): MODEL_NAMES["h2_image"],
}

# Metadata preprocessing defined in the TFM preprocessing pipeline.
ANATOM_SITE_MAPPING = {
    "anterior torso": 1,
    "posterior torso": 2,
    "upper extremity": 3,
    "lower extremity": 4,
    "head/neck": 5,
}

ANATOM_SITE_DUMMY_COLUMNS = [
    "anatom_site__anterior_torso",
    "anatom_site__head_neck",
    "anatom_site__lower_extremity",
    "anatom_site__posterior_torso",
    "anatom_site__upper_extremity",
]


# ============================================================
# Path and metadata helpers
# ============================================================


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root."""
    value = Path(value)

    if value.is_absolute():
        return value

    return get_project_root() / value


@st.cache_resource
def get_models_directory() -> Path:
    """Read the models directory from the project YAML configuration."""
    config_path = resolve_project_path(CONFIG_PATH)
    config = load_yaml_config(config_path)

    models_dir = resolve_project_path(config["paths"]["models"])

    if not models_dir.is_dir():
        raise FileNotFoundError(f"Configured models directory not found: {models_dir}")

    return models_dir


def get_model_directory(model_name: str) -> Path:
    """Resolve one hardcoded final artefact name inside the models directory."""
    model_dir = get_models_directory() / model_name

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    return model_dir


@st.cache_data
def read_model_metadata(model_dir_str: str) -> dict[str, Any]:
    """Read model_metadata.json from a model artefact."""
    model_dir = Path(model_dir_str)
    metadata_path = model_dir / "model_metadata.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing model_metadata.json in: {model_dir}")

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if "model" not in metadata:
        raise KeyError(f"'model' section missing from: {metadata_path}")

    return metadata


def get_selected_threshold(metadata: dict[str, Any]) -> float:
    """
    Read the frozen validation threshold from the selected artefact.

    No fallback is used: a final inference artefact must contain its own
    threshold.
    """
    evaluation = metadata.get("evaluation") or {}

    if "selected_threshold" not in evaluation:
        raise KeyError(
            "The selected model artefact does not contain "
            "evaluation.selected_threshold."
        )

    threshold = float(evaluation["selected_threshold"])

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Invalid selected threshold: {threshold}")

    return threshold


def resolve_saved_model_reference(reference: str | Path) -> Path:
    """
    Resolve a base-model reference stored inside a fusion artefact.

    Saved references may use Windows or POSIX path separators. If the original
    path no longer exists, recover the model directory by basename inside the
    current configured models directory.
    """
    reference_str = str(reference).replace("\\", "/")
    reference = Path(reference_str)

    candidates = []

    if reference.is_absolute():
        candidates.append(reference)
    else:
        candidates.append(resolve_project_path(reference))

    candidates.append(get_models_directory() / reference.name)

    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve saved base-model reference: {reference}"
    )


# ============================================================
# Model routing
# ============================================================


def select_model_name(
    hypothesis: int,
    use_metadata: bool,
    use_image: bool,
) -> str:
    """Select the final TFM artefact for the requested input combination."""
    key = (
        int(hypothesis),
        bool(use_metadata),
        bool(use_image),
    )

    if key not in MODEL_ROUTES:
        raise ValueError("Select at least one input modality: metadata and/or image.")

    return MODEL_ROUTES[key]


# ============================================================
# Model loading
# ============================================================


def select_device() -> torch.device:
    """Select CUDA, Apple MPS or CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


@st.cache_resource
def load_joblib_model(model_dir_str: str):
    """Load a saved joblib artefact using the filename stored in metadata."""
    model_dir = Path(model_dir_str)
    metadata = read_model_metadata(str(model_dir))

    model_file = metadata["model"].get("model_file")

    if not model_file:
        raise KeyError(
            f"model.model_file missing in {model_dir / 'model_metadata.json'}"
        )

    model_path = model_dir / model_file

    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model file not found: {model_path}")

    model = joblib.load(model_path)

    return model


@st.cache_resource
def load_direct_image_model(model_dir_str: str):
    """
    Reconstruct a frozen direct-image PyTorch model from its own artefact.

    Architecture, dropout, image size and normalization are read through the
    same final_models.py helpers used by the TFM evaluation pipeline.
    """
    model_dir = Path(model_dir_str)
    _, metadata = fm._load_model_metadata(model_dir)

    inference_config = fm._merged_inference_config(
        model_directory=model_dir,
        model_metadata=metadata,
        override=None,
    )

    if "image_size" not in inference_config:
        raise KeyError(f"Image size is not recoverable from artefact: {model_dir}")

    image_size = int(inference_config["image_size"])
    device = select_device()

    model = fm._load_pytorch_model(
        model_directory=model_dir,
        model_metadata=metadata,
        inference_config=inference_config,
        device=device,
    )

    transform = fm._make_evaluation_transform(inference_config)

    model.eval()

    return (
        model,
        transform,
        device,
        image_size,
    )


# ============================================================
# Metadata preprocessing
# ============================================================


def build_raw_metadata_values(
    sex: str,
    age: float,
    anatomical_site: str,
    diameter_mm: float,
) -> dict[str, Any]:
    """
    Reproduce the deterministic metadata encodings used in the TFM.

    The final feature subset is not hardcoded here. It is selected later from
    the feature names expected by the saved model artefact.
    """
    if sex not in {"female", "male"}:
        raise ValueError("Sex must be female or male.")

    if age < 18:
        raise ValueError("The TFM modelling cohort excluded patients younger than 18.")

    if anatomical_site not in ANATOM_SITE_MAPPING:
        raise ValueError(f"Unsupported anatomical site: {anatomical_site}")

    if diameter_mm < 0:
        raise ValueError("Lesion diameter cannot be negative.")

    values: dict[str, Any] = {
        "sex": sex,
        "sex_male": int(sex == "male"),
        "age_approx": float(age),
        "anatom_site_general": anatomical_site,
        "anatom_site_general_code": ANATOM_SITE_MAPPING[anatomical_site],
        "clin_size_long_diam_mm": float(diameter_mm),
        "clin_size_long_diam_mm_log1p": float(np.log1p(diameter_mm)),
    }

    for column in ANATOM_SITE_DUMMY_COLUMNS:
        values[column] = np.int8(0)

    selected_dummy = "anatom_site__" + anatomical_site.replace("/", "_").replace(
        " ", "_"
    )

    values[selected_dummy] = np.int8(1)

    return values


def get_expected_metadata_features(
    model,
    model_dir: Path,
    metadata: dict[str, Any],
) -> list[str]:
    """Recover the exact fitted metadata feature names from the artefact."""
    inference_config = fm._merged_inference_config(
        model_directory=model_dir,
        model_metadata=metadata,
        override=None,
    )

    feature_names = inference_config.get("feature_columns")

    if feature_names is not None:
        return [str(column) for column in feature_names]

    feature_names = fm._model_feature_names(model)

    if feature_names is None:
        raise ValueError(
            "The saved metadata model does not expose fitted feature names, "
            "and no feature_columns were found in its inference configuration."
        )

    return feature_names


def build_metadata_dataframe(
    model,
    model_dir: Path,
    metadata: dict[str, Any],
    raw_values: dict[str, Any],
) -> pd.DataFrame:
    """Build one inference row in the exact feature order expected by the model."""
    feature_names = get_expected_metadata_features(
        model=model,
        model_dir=model_dir,
        metadata=metadata,
    )

    missing_features = [
        feature for feature in feature_names if feature not in raw_values
    ]

    if missing_features:
        raise ValueError(
            "The selected metadata model expects features that the UI cannot "
            f"reconstruct safely: {missing_features}"
        )

    return pd.DataFrame(
        [{feature: raw_values[feature] for feature in feature_names}],
        columns=feature_names,
    )


# ============================================================
# Image preprocessing
# ============================================================


def decode_uploaded_image(uploaded_file) -> Image.Image:
    """Validate the uploaded file and decode it as RGB."""
    if uploaded_file is None:
        raise ValueError("An image is required for the selected analysis.")

    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix not in {".jpg", ".jpeg", ".png"}:
        raise ValueError("Supported image formats are JPG, JPEG and PNG.")

    raw_bytes = uploaded_file.getvalue()

    if not raw_bytes:
        raise ValueError("The uploaded image file is empty.")

    try:
        with Image.open(BytesIO(raw_bytes)) as source:
            source.load()

            if source.width <= 0 or source.height <= 0:
                raise ValueError("The uploaded image has invalid dimensions.")

            image = source.convert("RGB")

    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file could not be decoded as an image.") from exc

    return image


def prepare_image_for_model(
    image: Image.Image,
    image_size: int,
) -> Image.Image:
    """
    Convert an uploaded image into the tensor-compatible spatial shape.

    The model artefact determines image_size. RGB conversion is performed when
    decoding the file. Resizing is deterministic and only provides technical
    compatibility; it does not validate lesion content or photographic quality.
    """
    if image_size <= 0:
        raise ValueError(f"Invalid model image size: {image_size}")

    if image.size != (image_size, image_size):
        image = image.resize(
            (image_size, image_size),
            resample=Image.Resampling.LANCZOS,
        )

    return image


# ============================================================
# Probability inference
# ============================================================


def predict_metadata_probability(
    model_dir: Path,
    raw_metadata_values: dict[str, Any] | None,
) -> float:
    """Run one saved metadata model and return its positive-class probability."""
    if raw_metadata_values is None:
        raise ValueError("Metadata values are required by the selected model.")

    metadata = read_model_metadata(str(model_dir))
    model = load_joblib_model(str(model_dir))

    inference_config = fm._merged_inference_config(
        model_directory=model_dir,
        model_metadata=metadata,
        override=None,
    )

    input_kind = fm.detect_input_kind(
        model_dir,
        override=inference_config,
    )

    if input_kind != "metadata":
        raise ValueError(
            f"Expected a metadata model, but artefact input_kind is '{input_kind}'."
        )

    features = build_metadata_dataframe(
        model=model,
        model_dir=model_dir,
        metadata=metadata,
        raw_values=raw_metadata_values,
    )

    if not hasattr(model, "predict_proba"):
        raise TypeError("The saved metadata model does not expose predict_proba().")

    probability = float(model.predict_proba(features)[0, 1])

    return validate_probability(probability)


def predict_image_probability(
    model_dir: Path,
    uploaded_image: Image.Image | None,
) -> float:
    """Run one frozen direct-image model."""
    if uploaded_image is None:
        raise ValueError("An image is required by the selected model.")

    metadata = read_model_metadata(str(model_dir))

    inference_config = fm._merged_inference_config(
        model_directory=model_dir,
        model_metadata=metadata,
        override=None,
    )

    input_kind = fm.detect_input_kind(
        model_dir,
        override=inference_config,
    )

    if input_kind != "direct_image":
        raise ValueError(
            f"Expected a direct-image model, but artefact input_kind is '{input_kind}'."
        )

    (
        model,
        transform,
        device,
        image_size,
    ) = load_direct_image_model(str(model_dir))

    image = prepare_image_for_model(
        uploaded_image,
        image_size=image_size,
    )

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(tensor)
        probability = torch.sigmoid(logits).reshape(-1)[0].item()

    return validate_probability(float(probability))


def predict_fusion_probability(
    model_dir: Path,
    raw_metadata_values: dict[str, Any] | None,
    uploaded_image: Image.Image | None,
) -> float:
    """
    Run a saved fusion artefact.

    The fusion artefact itself identifies its two base-model directories and
    the saved ProbabilityFusionModel determines the combination rule.
    """
    metadata = read_model_metadata(str(model_dir))
    inference = metadata.get("inference") or {}

    metadata_reference = inference.get("metadata_model_directory")
    image_reference = inference.get("image_model_directory")

    if metadata_reference is None or image_reference is None:
        raise KeyError("Fusion artefact is missing its saved base-model references.")

    metadata_model_dir = resolve_saved_model_reference(metadata_reference)
    image_model_dir = resolve_saved_model_reference(image_reference)

    fusion_model = load_joblib_model(str(model_dir))

    metadata_probability = predict_artifact_probability(
        model_dir=metadata_model_dir,
        raw_metadata_values=raw_metadata_values,
        uploaded_image=uploaded_image,
    )

    image_probability = predict_artifact_probability(
        model_dir=image_model_dir,
        raw_metadata_values=raw_metadata_values,
        uploaded_image=uploaded_image,
    )

    if not hasattr(
        fusion_model,
        "predict_probability",
    ):
        raise TypeError("Saved fusion model does not expose predict_probability().")

    fused_probability = fusion_model.predict_probability(
        metadata_probability=np.asarray(
            [metadata_probability],
            dtype=float,
        ),
        image_probability=np.asarray(
            [image_probability],
            dtype=float,
        ),
    )

    return validate_probability(float(np.asarray(fused_probability).reshape(-1)[0]))


def predict_artifact_probability(
    model_dir: Path,
    raw_metadata_values: dict[str, Any] | None,
    uploaded_image: Image.Image | None,
) -> float:
    """
    Generic inference dispatcher driven by the saved artefact.

    Only the top-level artefact name is selected by MODEL_ROUTES. Everything
    else is read from that artefact and, for fusion, from its saved dependencies.
    """
    metadata = read_model_metadata(str(model_dir))

    framework = str(metadata["model"].get("framework", "")).lower()

    if framework == "fusion":
        return predict_fusion_probability(
            model_dir=model_dir,
            raw_metadata_values=raw_metadata_values,
            uploaded_image=uploaded_image,
        )

    inference_config = fm._merged_inference_config(
        model_directory=model_dir,
        model_metadata=metadata,
        override=None,
    )

    input_kind = fm.detect_input_kind(
        model_dir,
        override=inference_config,
    )

    if input_kind == "metadata":
        return predict_metadata_probability(
            model_dir=model_dir,
            raw_metadata_values=raw_metadata_values,
        )

    if input_kind == "direct_image":
        return predict_image_probability(
            model_dir=model_dir,
            uploaded_image=uploaded_image,
        )

    raise ValueError(
        "This UI currently supports final metadata models, direct-image "
        f"models and saved fusion artefacts. Found input_kind='{input_kind}'."
    )


def validate_probability(probability: float) -> float:
    """Validate that an inference output is a finite probability."""
    if not np.isfinite(probability):
        raise ValueError("The model returned a non-finite probability.")

    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"The model returned an invalid probability: {probability}")

    return probability


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Clasificación de lesiones cutáneas · TFM",
    page_icon="🔬",
    layout="centered",
)


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

logo_path = resolve_project_path(LOGO_PATH)

header_logo, header_text = st.columns([1, 4])

with header_logo:
    if logo_path.is_file():
        st.image(str(logo_path), width=150)

with header_text:
    st.title(PROJECT_TITLE)
    st.markdown(f"*{PROJECT_SUBTITLE}*")
    st.caption(
        "Trabajo Final de Máster · Máster en Big Data & Data Science · Universitat de Barcelona"
    )

with st.expander("Información del proyecto"):
    st.markdown(
        f"""
**{PROJECT_COURSE}**

**Autores:** {PROJECT_AUTHORS}
**Tutor:** {PROJECT_TUTOR}

Esta aplicación es una prueba de concepto académica desarrollada en el marco
del Trabajo Final de Máster. No es un producto sanitario y no ha sido validada
para uso clínico.
"""
    )


# ------------------------------------------------------------
# Input selection
# ------------------------------------------------------------

st.subheader("Datos para el análisis")

input_mode = st.radio(
    "Información disponible",
    options=[
        "Imagen y metadatos clínicos",
        "Solo imagen",
        "Solo metadatos clínicos",
    ],
    horizontal=True,
)

use_image = input_mode != "Solo metadatos clínicos"
use_metadata = input_mode != "Solo imagen"

uploaded_file = None
decoded_image = None
raw_metadata_values = None


# ------------------------------------------------------------
# Image input
# ------------------------------------------------------------

if use_image:
    st.markdown("#### Imagen macroscópica")

    uploaded_file = st.file_uploader(
        "Sube una imagen de la lesión",
        type=["jpg", "jpeg", "png"],
        help="Formatos admitidos: JPG, JPEG y PNG.",
    )

    st.caption(
        "La validación del archivo es únicamente técnica: la aplicación no "
        "verifica que la imagen contenga una lesión cutánea ni que tenga "
        "calidad clínica suficiente."
    )

    if uploaded_file is not None:
        try:
            decoded_image = decode_uploaded_image(uploaded_file)

            st.image(
                decoded_image,
                caption=(
                    f"Imagen cargada · "
                    f"{decoded_image.width} × {decoded_image.height} px"
                ),
                width=280,
            )

        except Exception:
            st.error(
                "No se ha podido procesar la imagen. Comprueba el formato "
                "y vuelve a intentarlo."
            )


# ------------------------------------------------------------
# Metadata input
# ------------------------------------------------------------

if use_metadata:
    st.markdown("#### Metadatos clínicos")

    sex_labels = {
        "female": "Mujer",
        "male": "Hombre",
    }

    anatomical_site_labels = {
        "anterior torso": "Tronco anterior",
        "posterior torso": "Tronco posterior",
        "upper extremity": "Extremidad superior",
        "lower extremity": "Extremidad inferior",
        "head/neck": "Cabeza / cuello",
    }

    col1, col2 = st.columns(2)

    with col1:
        sex = st.selectbox(
            "Sexo",
            options=["female", "male"],
            format_func=lambda value: sex_labels[value],
        )

        age = st.number_input(
            "Edad (años)",
            min_value=18,
            max_value=120,
            value=50,
            step=1,
        )

    with col2:
        anatomical_site = st.selectbox(
            "Localización anatómica",
            options=list(ANATOM_SITE_MAPPING.keys()),
            format_func=lambda value: anatomical_site_labels[value],
        )

        diameter_mm = st.number_input(
            "Diámetro máximo de la lesión (mm)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.1,
            format="%.1f",
        )

    raw_metadata_values = build_raw_metadata_values(
        sex=sex,
        age=float(age),
        anatomical_site=anatomical_site,
        diameter_mm=float(diameter_mm),
    )


# ------------------------------------------------------------
# Sequential H1 -> H2 inference
# ------------------------------------------------------------

st.divider()

analyse = st.button(
    "Analizar lesión",
    type="primary",
    use_container_width=True,
)

if analyse:
    if use_image and decoded_image is None:
        st.error("Debes subir una imagen válida antes de realizar el análisis.")
        st.stop()

    try:
        with st.spinner("Analizando la lesión..."):
            # H1: prioritisation pattern based on biopsied vs non-biopsied lesions.
            h1_model_name = select_model_name(
                hypothesis=1,
                use_metadata=use_metadata,
                use_image=use_image,
            )
            h1_model_dir = get_model_directory(h1_model_name)
            h1_model_metadata = read_model_metadata(str(h1_model_dir))

            h1_probability = predict_artifact_probability(
                model_dir=h1_model_dir,
                raw_metadata_values=raw_metadata_values,
                uploaded_image=decoded_image,
            )
            h1_threshold = get_selected_threshold(h1_model_metadata)
            h1_positive = h1_probability >= h1_threshold

            # H2 is only applicable if H1 places the lesion on the biopsied side,
            # because H2 was developed among biopsied lesions.
            h2_positive = None

            if h1_positive:
                h2_model_name = select_model_name(
                    hypothesis=2,
                    use_metadata=use_metadata,
                    use_image=use_image,
                )
                h2_model_dir = get_model_directory(h2_model_name)
                h2_model_metadata = read_model_metadata(str(h2_model_dir))

                h2_probability = predict_artifact_probability(
                    model_dir=h2_model_dir,
                    raw_metadata_values=raw_metadata_values,
                    uploaded_image=decoded_image,
                )
                h2_threshold = get_selected_threshold(h2_model_metadata)
                h2_positive = h2_probability >= h2_threshold

        st.subheader("Valoración orientativa")

        if not h1_positive:
            st.info(
                "El modelo no identifica un patrón suficiente para priorizar "
                "esta lesión para una valoración dirigida a biopsia."
            )

            st.markdown(
                """
Este resultado **no permite descartar malignidad** ni implica que una biopsia
no pueda estar clínicamente indicada. La decisión sobre seguimiento,
derivación, exploración dermatoscópica o biopsia debe realizarla un profesional
médico teniendo en cuenta el conjunto de la valoración clínica.
"""
            )

        else:
            st.warning(
                "Los datos introducidos presentan un patrón que justifica la "
                "valoración de la lesión por un profesional médico para "
                "considerar la necesidad de biopsia."
            )

            st.markdown("#### Orientación adicional")

            if h2_positive:
                st.warning(
                    "El segundo análisis identifica un patrón con sospecha "
                    "de malignidad."
                )
            else:
                st.info(
                    "El segundo análisis identifica un patrón de mayor "
                    "compatibilidad con benignidad."
                )

            st.markdown(
                """
La orientación sobre benignidad o malignidad es **exclusivamente orientativa**
y no constituye un diagnóstico. La confirmación de una lesión sospechosa
requiere valoración médica y, cuando esté indicada, estudio histopatológico.
"""
            )

        st.warning(
            "Esta herramienta es una prueba de concepto académica y funciona "
            "únicamente como sistema de apoyo. Sus resultados no constituyen "
            "un diagnóstico ni una indicación médica. La valoración realizada "
            "por un profesional médico debe prevalecer siempre sobre el "
            "resultado del modelo."
        )

    except Exception as exc:
        st.error(
            "No se ha podido completar el análisis. Comprueba los datos "
            "introducidos y la configuración de los modelos."
        )
        with st.expander("Detalle técnico del error"):
            st.exception(exc)


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Prototipo de investigación · Uso exclusivamente académico · "
    "No validado para uso clínico"
)
st.caption(LEGAL_NOTICE)
