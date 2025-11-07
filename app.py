import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st
import tensorflow as tf


# ----------- App config -----------
st.set_page_config(page_title="Sidewalk Accessibility (Multi-Head)", layout="wide")

DEFAULT_IMG_SIZE = 512
DEFAULT_MODEL_PATH = "best_multitask.keras"  # change if you saved elsewhere


# ----------- Utilities -----------

@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    """Load a Keras multi-output model once (cached)."""
    model = tf.keras.models.load_model(model_path, compile=False)
    return model


def load_image_any(path: Path, img_size: int) -> Tuple[np.ndarray, Image.Image]:
    """Load an image, return (np.float32 tensor [H,W,3] in [0,1], PIL image)."""
    pil = Image.open(path).convert("RGB")
    pil = pil.resize((img_size, img_size))
    arr = np.asarray(pil).astype(np.float32) / 255.0
    return arr, pil


def preprocess_for_mobilenet(arr: np.ndarray) -> np.ndarray:
    """Apply MobileNetV2 preprocessing on [0,1] float image."""
    x = arr.copy()
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # scales to [-1,1]
    return x


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def probs_to_label_index(
    p: np.ndarray,
    tau: float = 0.50,
    ent_tau: float = 1.20,
    margin: float = 0.10
) -> int:
    """
    Decision policy:
      - If max prob < tau  -> unsure
      - Or entropy too high -> unsure
      - Or (top - second) < margin -> unsure
      - else argmax
    Returns class index: 0=no, 1=unsure, 2=yes.
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
    """
    Try exact basename first (e.g., '46864.jpg'); otherwise contain substring.
    Supports .jpg/.jpeg/.png/.webp.
    """
    image_id = str(image_id)
    exts = (".jpg", ".jpeg", ".png", ".webp")
    # Exact base match
    for p in image_root.rglob("*"):
        if p.suffix.lower() in exts and p.stem == image_id:
            return p
    # Fallback: contains id
    for p in image_root.rglob("*"):
        if p.suffix.lower() in exts and image_id in p.name:
            return p
    return None


def predict_all_heads(
    model: tf.keras.Model,
    img_np01: np.ndarray,
    batch_size: int = 1
) -> Dict[str, np.ndarray]:
    """
    Run forward pass. Returns dict: head_name -> (3,) probabilities.
    """
    x = preprocess_for_mobilenet(img_np01)
    x = np.expand_dims(x, axis=0)  # (1,H,W,3)
    outputs = model(x, training=False)  # dict of logits OR probs (we used softmax in the head)
    out: Dict[str, np.ndarray] = {}
    # If your model heads already use softmax activation, skip softmax() below
    for head_name, logits_or_probs in outputs.items():
        arr = np.asarray(logits_or_probs)[0]  # (3,)
        # Detect if logits (no softmax) or already probabilities
        if np.any(arr < 0) and np.abs(arr).max() > 1.0:
            out[head_name] = softmax(arr)
        else:
            # Best-effort: assume already probs
            arr = np.clip(arr, 1e-9, 1.0)
            arr = arr / arr.sum()
            out[head_name] = arr
    return out


def make_results_table(
    probs_dict: Dict[str, np.ndarray],
    tau: float,
    ent_tau: float,
    margin: float
) -> pd.DataFrame:
    """
    Build a DataFrame with p_no, p_unsure, p_yes, decision, argmax for each head.
    """
    records: List[Dict[str, object]] = []
    for head, p in sorted(probs_dict.items()):
        pred_i = probs_to_label_index(p, tau=tau, ent_tau=ent_tau, margin=margin)
        arg_i = int(np.argmax(p))
        records.append({
            "MobilityAid": head.replace("head_", "").replace("_", " "),
            "p_no": float(p[0]),
            "p_unsure": float(p[1]),
            "p_yes": float(p[2]),
            "argmax": ["no", "unsure", "yes"][arg_i],
            "decision": ["no", "unsure", "yes"][pred_i],
        })
    df = pd.DataFrame.from_records(records)
    return df


# ----------- Sidebar -----------
st.sidebar.header("Settings")

model_path = st.sidebar.text_input("Model path (.keras / .h5)", DEFAULT_MODEL_PATH)
img_size = st.sidebar.number_input("Image size (training was 512)", min_value=96, max_value=1024, value=DEFAULT_IMG_SIZE, step=32)

st.sidebar.subheader("Decision policy")
tau = st.sidebar.slider("Confidence threshold τ", 0.30, 0.90, 0.50, 0.01)
margin = st.sidebar.slider("Top-2 margin", 0.00, 0.30, 0.10, 0.01)
ent_tau = st.sidebar.slider("Entropy threshold", 0.60, 2.00, 1.20, 0.05)

st.sidebar.subheader("Test image source")
mode = st.sidebar.radio("Select input mode", ["Upload", "Pick by ImageID (from a folder)"])

image_root = None
image_id = None
uploaded_img = None

if mode == "Upload":
    uploaded_img = st.sidebar.file_uploader("Upload a test image", type=["jpg", "jpeg", "png", "webp"])
else:
    folder_str = st.sidebar.text_input("Image folder (e.g., /content/Project_Sidewalk_Data/sidewalk-images)", "")
    image_root = Path(folder_str) if folder_str else None
    image_id = st.sidebar.text_input("ImageID (e.g., 46864)", "")


# ----------- Main area -----------

st.title("Sidewalk Accessibility — Multi-Head Predictor")

# Load model
if not Path(model_path).exists():
    st.warning(f"Model not found at: {model_path}")
    st.stop()

with st.spinner("Loading model…"):
    model = load_model(model_path)
st.success("Model loaded.")

# Discover heads from model outputs
if hasattr(model, "output_names"):
    heads = list(model.output_names)
else:
    # Fallback: keys from calling once (we’ll call after we have an image)
    heads = None

# Pick image
chosen_img_path: Path | None = None
if uploaded_img is not None:
    # Save to a tmp in memory for PIL; no disk write needed
    pil = Image.open(uploaded_img).convert("RGB")
    pil = pil.resize((img_size, img_size))
    img_np01 = np.asarray(pil).astype(np.float32) / 255.0
    chosen_pil = pil
else:
    if image_root is None or not image_root.exists() or not image_id:
        st.info("Upload an image or fill in a valid image folder + ImageID in the sidebar.")
        st.stop()
    chosen_img_path = find_image_by_id(image_root, image_id)
    if chosen_img_path is None:
        st.error("Couldn’t find an image containing this ImageID in the provided folder.")
        st.stop()
    img_np01, chosen_pil = load_image_any(chosen_img_path, img_size)

# Show image
col_img, col_tbl = st.columns([1, 1.2])
with col_img:
    st.subheader("Input image")
    st.image(chosen_pil, caption=str(chosen_img_path) if chosen_img_path else "uploaded", use_container_width=True)

# Predict
with st.spinner("Running inference…"):
    probs_dict = predict_all_heads(model, img_np01)
df_out = make_results_table(probs_dict, tau=tau, ent_tau=ent_tau, margin=margin)

with col_tbl:
    st.subheader("Per-mobility-aid predictions")
    st.dataframe(
        df_out.style.format({"p_no": "{:.3f}", "p_unsure": "{:.3f}", "p_yes": "{:.3f}"}),
        use_container_width=True
    )

st.markdown("---")
st.subheader("Per-head bars (optional)")
show_bars = st.checkbox("Show probability bars", value=False)
if show_bars:
    for _, row in df_out.iterrows():
        head = row["MobilityAid"]
        probs = [row["p_no"], row["p_unsure"], row["p_yes"]]
        chart_df = pd.DataFrame(
            {"class": ["no", "unsure", "yes"], "probability": probs}
        )
        st.markdown(f"**{head}** — decision: `{row['decision']}` (argmax: `{row['argmax']}`)")
        st.bar_chart(chart_df.set_index("class"))