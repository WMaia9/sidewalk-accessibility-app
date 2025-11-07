import os
from pathlib import Path
from typing import Dict, List, Tuple

import hashlib
import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import tensorflow as tf


# =============================
# App config & constants
# =============================
st.set_page_config(page_title="Sidewalk Accessibility (Multi-Head)", layout="wide")

DEFAULT_IMG_SIZE = 512

# Updated: List of available models
AVAILABLE_MODELS = ["best_multitask.keras", "best_multitask2.keras"]

CLASS_NAMES = ["no", "unsure", "yes"]

# =============================
# Utilities
# =============================
@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str):
    """Load a multi-output Keras model once (cached by path)."""
    # Clears cache if a new model is selected to avoid memory bloat if switching often
    # (Optional depending on your server RAM)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def load_image_any(path: Path, img_size: int) -> Tuple[np.ndarray, Image.Image]:
    """Load an image; return (float32 [H,W,3] in [0, 255], PIL)."""
    pil = Image.open(path).convert("RGB")
    pil = pil.resize((img_size, img_size))
    # Keep in [0, 255] range for MobileNetV2
    arr = np.asarray(pil, dtype=np.float32)
    return arr, pil


def preprocess_for_mobilenet(arr: np.ndarray) -> np.ndarray:
    """Match training: [0, 255] float -> MobileNetV2 preprocess_input ([-1,1])."""
    x = arr.astype(np.float32)
    if x.ndim == 3 and x.shape[-1] not in (1, 3) and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    return tf.keras.applications.mobilenet_v2.preprocess_input(x)


def _is_prob_vector(a: np.ndarray, tol: float = 1e-3) -> bool:
    """Heuristic: already probs if sum≈1 and values in [0,1]."""
    s = float(np.sum(a))
    return (abs(s - 1.0) < tol) and np.all(a >= -tol) and np.all(a <= 1.0 + tol)


def probs_to_label_index(
    p: np.ndarray,
    tau: float = 0.50,
    ent_tau: float = 1.20,
    margin: float = 0.10,
) -> int:
    """
    Decision policy:
      - If max prob < tau  -> unsure
      - Or entropy too high -> unsure
      - Or (top - second) < margin -> unsure
      - else argmax
    Returns 0=no, 1=unsure, 2=yes.
    """
    p = np.clip(np.asarray(p, float), 1e-9, 1.0)
    p = p / p.sum()
    ent = float(-(p * np.log(p)).sum())
    top = float(p.max())
    second = float(np.partition(p, -2)[-2])
    if (top < tau) or (ent > ent_tau) or ((top - second) < margin):
        return 1
    return int(np.argmax(p))


def find_image_by_id(image_root: Path, image_id: str) -> Path | None:
    image_id = str(image_id)
    exts = (".jpg", ".jpeg", ".png", ".webp")
    if image_root and image_root.exists():
        for ext in exts:
            candidate = image_root / (image_id + ext)
            if candidate.exists():
                return candidate
        for p in image_root.rglob("*"):
            if p.suffix.lower() in exts and p.stem == image_id:
                return p
    return None


def predict_all_heads(model: tf.keras.Model, img_np: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Forward pass. Returns dict: head_name -> (3,) probabilities in CLASS_NAMES order.
    """
    x = preprocess_for_mobilenet(img_np)
    x = np.expand_dims(x, axis=0)

    outputs = model(x, training=False)
    out: Dict[str, np.ndarray] = {}

    if isinstance(outputs, (list, tuple)) and hasattr(model, "output_names"):
        outputs = {name: outputs[i] for i, name in enumerate(model.output_names)}
    elif not isinstance(outputs, dict):
         outputs = {model.output_names[0]: outputs}

    for head_name, vec in outputs.items():
        a = np.asarray(vec)[0].astype(np.float32)
        if _is_prob_vector(a):
            p = a / max(a.sum(), 1e-9)
        else:
            a = a - a.max()
            p = np.exp(a)
            p /= p.sum()
        out[head_name] = p
    return out


def make_results_table(
    probs_dict: Dict[str, np.ndarray],
    tau: float,
    ent_tau: float,
    margin: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for head, p in sorted(probs_dict.items()):
        pred_i = probs_to_label_index(p, tau=tau, ent_tau=ent_tau, margin=margin)
        arg_i = int(np.argmax(p))
        rows.append(
            {
                "MobilityAid": head.replace("head_", "").replace("_", " ").title(),
                "p_no": float(p[0]),
                "p_unsure": float(p[1]),
                "p_yes": float(p[2]),
                "argmax": CLASS_NAMES[arg_i],
                "decision": CLASS_NAMES[pred_i],
            }
        )
    # Swapped order: argmax before decision
    df = pd.DataFrame(rows, columns=["MobilityAid", "p_no", "p_unsure", "p_yes", "argmax", "decision"])
    return df


# =============================
# Sidebar
# =============================
st.sidebar.header("Settings")

# CHANGED: Replaced text_input with selectbox for the two models
model_path = st.sidebar.selectbox("Select Model", AVAILABLE_MODELS)

img_size = st.sidebar.number_input("Image size", min_value=96, max_value=1024, value=DEFAULT_IMG_SIZE, step=32)

st.sidebar.subheader("Decision policy")
tau = st.sidebar.slider("Confidence threshold τ", 0.30, 0.90, 0.50, 0.01)
margin = st.sidebar.slider("Top-2 margin", 0.00, 0.30, 0.10, 0.01)
ent_tau = st.sidebar.slider("Entropy threshold", 0.60, 2.00, 1.20, 0.05)

st.sidebar.subheader("Test image source")
mode = st.sidebar.radio("Select input mode", ["Upload", "Pick by ImageID (from a folder)"])

image_root: Path | None = None
image_id: str | None = None
uploaded_img = None

if mode == "Upload":
    uploaded_img = st.sidebar.file_uploader("Upload a test image", type=["jpg", "jpeg", "png", "webp"])
else:
    folder_str = st.sidebar.text_input("Image folder (e.g., /mount/data/sidewalk-images)", "")
    image_root = Path(folder_str) if folder_str else None
    image_id = st.sidebar.text_input("ImageID (e.g., 46864)", "")


# =============================
# Main
# =============================
st.title("Sidewalk Accessibility — Multi-Head Predictor")

# Load model
if not Path(model_path).exists():
    st.warning(f"Model not found at: {model_path}")
    st.stop()

with st.spinner(f"Loading {model_path}…"):
    model = load_model_cached(model_path)

# Get image tensor
chosen_img_path: Path | None = None
img_np = None
chosen_pil = None

if uploaded_img is not None:
    pil = Image.open(uploaded_img).convert("RGB")
    pil = pil.resize((img_size, img_size))
    img_np = np.asarray(pil, dtype=np.float32)
    chosen_pil = pil
elif mode == "Pick by ImageID (from a folder)":
    if image_root is None or not image_root.exists() or not image_id:
        st.info("Provide a valid image folder + ImageID in the sidebar.")
        st.stop()
    chosen_img_path = find_image_by_id(image_root, image_id)
    if chosen_img_path is None:
        st.error("Couldn’t find an image with that ImageID in the provided folder.")
        st.stop()
    img_np, chosen_pil = load_image_any(chosen_img_path, img_size)
else:
    st.info("Upload an image to begin.")
    st.stop()

# Layout: image | table
col_img, col_tbl = st.columns([1, 1.5])
with col_img:
    st.subheader("Input image")
    st.image(chosen_pil, caption=chosen_img_path.name if chosen_img_path else "uploaded", use_container_width=True)

# Predict & table
if img_np is not None:
    with st.spinner("Running inference…"):
        probs_dict = predict_all_heads(model, img_np)
    df_out = make_results_table(probs_dict, tau=tau, ent_tau=ent_tau, margin=margin)

    with col_tbl:
        st.subheader("Per-mobility-aid predictions")
        st.dataframe(
            df_out.style.format({"p_no": "{:.3f}", "p_unsure": "{:.3f}", "p_yes": "{:.3f}"}),
            use_container_width=True,
        )

    st.markdown("---")
    if st.checkbox("Show probability bars", value=False):
        st.subheader("Detailed breakdown")
        for _, row in df_out.iterrows():
            head = row["MobilityAid"]
            probs = [row["p_no"], row["p_unsure"], row["p_yes"]]
            chart_df = pd.DataFrame({"class": ["no", "unsure", "yes"], "probability": probs})
            st.markdown(f"**{head}** — Decision: `{row['decision']}`")
            st.bar_chart(chart_df.set_index("class"))
