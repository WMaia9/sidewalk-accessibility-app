import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf

# ===== Optional deps for VLM path =====
try:
    import torch
    import joblib
    import open_clip
except ImportError:
    torch = None
    joblib = None
    open_clip = None

# =============================
# App config & constants
# =============================
st.set_page_config(page_title="Sidewalk Accessibility — CNN vs VLM", layout="wide")

DEFAULT_IMG_SIZE = 512
DEFAULT_MODEL_PATH = "best_multitask.keras"
DEFAULT_PROBES_DIR = "sidewalk_probe_exports"
CLASS_NAMES = ["no", "unsure", "yes"]

# =============================
# Utilities (shared)
# =============================
def _is_prob_vector(a: np.ndarray, tol: float = 1e-3) -> bool:
    s = float(np.sum(a))
    return (abs(s - 1.0) < tol) and np.all(a >= -tol) and np.all(a <= 1.0 + tol)


def probs_to_label_index(
    p: np.ndarray, tau: float = 0.50, ent_tau: float = 1.20, margin: float = 0.10
) -> int:
    p = np.clip(np.asarray(p, float), 1e-9, 1.0)
    p = p / p.sum()
    ent = float(-(p * np.log(p)).sum())
    top = float(p.max())
    second = float(np.partition(p, -2)[-2])
    if (top < tau) or (ent > ent_tau) or ((top - second) < margin):
        return 1  # force 'unsure'
    return int(np.argmax(p))


def compute_diagnostics(p: np.ndarray) -> Tuple[float, float, float]:
    p = np.clip(np.asarray(p, float), 1e-12, 1.0)
    p = p / p.sum()
    top = float(p.max())
    second = float(np.partition(p, -2)[-2])
    ent = float(-(p * np.log(p)).sum())
    return top, top - second, ent


def find_image_by_id(image_root: Path, image_id: str) -> Optional[Path]:
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

# =============================
# CNN (Keras) backend
# =============================
@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def load_image_for_cnn(pil_img: Image.Image, target_size: int) -> Tuple[np.ndarray, Image.Image]:
    """Resize to square for CNN and normalize to [0, 255]."""
    pil_resized = pil_img.convert("RGB").resize((target_size, target_size))
    arr = np.asarray(pil_resized, dtype=np.float32)
    return arr, pil_resized


def preprocess_for_mobilenet(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32)
    if x.ndim == 3 and x.shape[-1] not in (1, 3) and x.shape[0] in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    return tf.keras.applications.mobilenet_v2.preprocess_input(x)


def predict_all_heads_cnn(model: tf.keras.Model, img_np: np.ndarray) -> Dict[str, np.ndarray]:
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

# =============================
# VLM (CLIP + linear probes) backend
# =============================
@st.cache_resource(show_spinner=False)
def load_clip_and_preprocess(model_name: str, pretrained: str, device_str: str = "auto"):
    assert open_clip is not None and torch is not None
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name=model_name, pretrained=pretrained)
    model.eval().to(device)
    return model, preprocess, device


@st.cache_resource(show_spinner=False)
def load_probes(probes_dir: str):
    assert joblib is not None
    root = Path(probes_dir)
    if not root.exists():
        raise FileNotFoundError(f"Probes folder not found: {probes_dir}")

    bundles = {}
    for sub in sorted([p for p in root.iterdir() if p.is_dir()]):
        meta_path = sub / "meta.json"
        scaler_path = sub / "scaler.joblib"
        clf_path = sub / "clf.joblib"
        if not (meta_path.exists() and scaler_path.exists() and clf_path.exists()):
            continue
        with open(meta_path, "r") as f:
            meta = json.load(f)
        bundles[sub.name] = {
            "scaler": joblib.load(scaler_path),
            "clf": joblib.load(clf_path),
            "meta": meta,
        }
    if not bundles:
        raise RuntimeError(f"No probes found under: {probes_dir}")
    return bundles


def sanitize_to_title(head: str) -> str:
    return head.replace("_", " ").title()


@st.cache_data(show_spinner=False)
def encode_clip_feature_once(_model, _preprocess, _device, pil_original: Image.Image):
    """Encodes the ORIGINAL aspect ratio image, letting CLIP handle resizing."""
    with torch.no_grad():
        x = _preprocess(pil_original.convert("RGB")).unsqueeze(0).to(_device)
        f = _model.encode_image(x)
        f = (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()
    return f


def predict_all_heads_vlm(
    model, preprocess, device, pil: Image.Image, bundles: Dict[str, dict]
) -> Dict[str, np.ndarray]:
    feat = encode_clip_feature_once(model, preprocess, device, pil)
    out: Dict[str, np.ndarray] = {}
    for folder_name, b in bundles.items():
        scaler, clf, meta = b["scaler"], b["clf"], b["meta"]

        # EXACTLY match notebook's predict_proba_3 expansion
        f_s = scaler.transform(feat)  # [1, D]
        probs_raw = clf.predict_proba(f_s)  # [1, C']

        classes = getattr(clf, "classes_", np.arange(probs_raw.shape[1]))
        probs_full = np.zeros((probs_raw.shape[0], 3), dtype=float)
        probs_full[:, classes] = probs_raw

        # Robust normalization
        probs_full = (probs_full + 1e-9) / (probs_full.sum(axis=1, keepdims=True) + 3e-9)
        p = probs_full[0]

        # Key for display: prefer meta aid name; else folder
        pretty = meta.get("aid", sanitize_to_title(folder_name))
        out[pretty] = p
    return out

# =============================
# Sidebar
# =============================
st.sidebar.header("Settings")
backend = st.sidebar.radio("Backend", ["CNN (Keras multi-head)", "VLM (CLIP + probes)"])

st.sidebar.subheader("Decision policy")
tau = st.sidebar.slider("Confidence threshold τ", 0.30, 0.90, 0.50, 0.01)
margin = st.sidebar.slider("Top-2 margin", 0.00, 0.30, 0.10, 0.01)
ent_tau = st.sidebar.slider("Entropy threshold", 0.60, 2.00, 1.20, 0.05)

st.sidebar.subheader("Input")
mode = st.sidebar.radio("Source", ["Upload", "Pick by ImageID"])

image_root: Optional[Path] = None
image_id: Optional[str] = None
uploaded_img = None

if mode == "Upload":
    uploaded_img = st.sidebar.file_uploader("Upload Image", type=["jpg", "jpeg", "png", "webp"])
else:
    folder_str = st.sidebar.text_input("Image folder", "")
    image_root = Path(folder_str) if folder_str else None
    image_id = st.sidebar.text_input("ImageID (e.g., 46864)", "")

if backend == "CNN (Keras multi-head)":
    st.sidebar.markdown("---")
    st.sidebar.subheader("CNN Settings")
    model_path = st.sidebar.text_input("Keras model path", DEFAULT_MODEL_PATH)
    cnn_img_size = st.sidebar.number_input("CNN Input Size", 96, 1024, DEFAULT_IMG_SIZE, 32)
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("VLM Settings")
    probes_dir = st.sidebar.text_input("Probes folder", DEFAULT_PROBES_DIR)
    # IMPORTANT: default to **openai** to match training notebook
    clip_model = st.sidebar.text_input("CLIP model", "ViT-B-32")
    clip_pretrained = st.sidebar.text_input("Pretrained tag", "openai")
    use_probe_thresholds = st.sidebar.checkbox("Use probe-specific thresholds", value=True)

# =============================
# Main
# =============================
st.title("Sidewalk Accessibility — CNN vs VLM")

# 1. Load Original PIL Image
pil_original: Optional[Image.Image] = None
chosen_name: Optional[str] = None

if uploaded_img is not None:
    pil_original = Image.open(uploaded_img).convert("RGB")
    chosen_name = uploaded_img.name
elif mode == "Pick by ImageID" and image_root and image_id:
    path = find_image_by_id(image_root, image_id)
    if path:
        pil_original = Image.open(path).convert("RGB")
        chosen_name = path.name
    else:
        st.error("ImageID not found in folder.")
        st.stop()
else:
    st.info("Select an image to begin.")
    st.stop()

# 2. Display Input
col_img, col_tbl = st.columns([1, 1.5])
with col_img:
    st.subheader("Input Image")
    st.image(pil_original, caption=chosen_name, width='stretch')

# 3. Run Inference
if backend == "CNN (Keras multi-head)":
    if not Path(model_path).exists():
        st.error(f"Model not found: {model_path}")
        st.stop()

    with st.spinner("Running CNN..."):
        model = load_model_cached(model_path)
        # Resize specifically for CNN
        img_np, _ = load_image_for_cnn(pil_original, cnn_img_size)
        probs_dict = predict_all_heads_cnn(model, img_np)

    # Table Building (CNN uses global sliders)
    rows = []
    for head, p in sorted(probs_dict.items()):
        pred_i = probs_to_label_index(p, tau, ent_tau, margin)
        pmax, mrg, ent = compute_diagnostics(p)
        rows.append({
            "MobilityAid": sanitize_to_title(head.replace("head_", "")),
            "p_no": p[0], "p_unsure": p[1], "p_yes": p[2],
            "argmax": CLASS_NAMES[np.argmax(p)],
            "decision": CLASS_NAMES[pred_i],
            "accepted": (pred_i != 1) or (np.argmax(p) == 1),  # crude indicator for CNN
            "pmax": pmax, "margin": mrg, "entropy": ent,
            "τ": tau, "Δ": margin, "Hτ": ent_tau,
        })
    df_out = pd.DataFrame(rows)

else:  # VLM Backend
    if not (torch and open_clip and joblib):
        st.error("Missing VLM dependencies (torch, open_clip, joblib).")
        st.stop()
    if not Path(probes_dir).exists():
        st.error(f"Probes folder not found: {probes_dir}")
        st.stop()

    with st.spinner(f"Running CLIP ({clip_model})..."):
        # Load CLIP
        clip_backbone, clip_preproc, device = load_clip_and_preprocess(clip_model, clip_pretrained)
        # Load Probes
        bundles = load_probes(probes_dir)

    # Warn if backbones differ from what probes expect
    selected_sig = f"open_clip::{clip_model}::{clip_pretrained}"
    for folder_name, b in bundles.items():
        meta = b["meta"]
        trained_sig = str(meta.get("backbone", ""))
        if trained_sig and trained_sig != selected_sig:
            st.warning(
                f"Probe '{meta.get('aid', sanitize_to_title(folder_name))}' was trained on **{trained_sig}**, "
                f"but you selected **{selected_sig}**. Predictions may be unreliable.",
                icon="⚠️",
            )

    # Predict using ORIGINAL PIL image
    probs_dict = predict_all_heads_vlm(clip_backbone, clip_preproc, device, pil_original, bundles)

    # Build reverse index: aid name -> bundle for quick threshold lookup
    aid_to_bundle = {}
    for folder_name, b in bundles.items():
        aid_name = b["meta"].get("aid", sanitize_to_title(folder_name))
        aid_to_bundle[aid_name] = b

    # Table Building (VLM supports per-probe thresholds)
    rows = []
    for aid, p in sorted(probs_dict.items()):
        # Defaults from sliders
        t_eff, m_eff, e_eff = tau, margin, ent_tau
        note = None

        if use_probe_thresholds and aid in aid_to_bundle:
            meta = aid_to_bundle[aid]["meta"]
            t_eff = meta.get("tau", t_eff)
            m_eff = meta.get("margin", m_eff)
            e_eff = meta.get("ent_tau", e_eff)
            seen = meta.get("classes_seen", [])
            if isinstance(seen, list) and len(seen) < 3:
                missing = [CLASS_NAMES[i] for i in range(3) if i not in seen]
                if missing:
                    note = f"trained without: {', '.join(missing)}"

        pred_i = probs_to_label_index(p, t_eff, e_eff, m_eff)

        # Diagnostics (pmax, margin, entropy, accepted)
        pmax, mrg, ent = compute_diagnostics(p)
        accepted = (pmax >= t_eff) and (mrg >= m_eff) and (ent <= e_eff)

        rows.append({
            "MobilityAid": aid,
            "p_no": p[0], "p_unsure": p[1], "p_yes": p[2],
            "argmax": CLASS_NAMES[np.argmax(p)],
            "decision": CLASS_NAMES[pred_i],
            "accepted": accepted,
            "pmax": pmax, "margin": mrg, "entropy": ent,
            "τ": t_eff, "Δ": m_eff, "Hτ": e_eff,
            "note": note,
        })
    df_out = pd.DataFrame(rows)

# 4. Display Results
with col_tbl:
    st.subheader(f"Results ({backend.split()[0]})")
    display_cols = [
        "MobilityAid", "p_no", "p_unsure", "p_yes", "argmax", "decision",
    ]
    fmt_cols = {"p_no": "{:.3f}", "p_unsure": "{:.3f}", "p_yes": "{:.3f}"}

    existing_cols = [c for c in display_cols if c in df_out.columns]
    st.dataframe(
        df_out[existing_cols].style.format(fmt_cols),
        width='stretch',
        column_order=existing_cols,
    )

st.markdown("---")
if st.checkbox("Show probability bars", value=False):
    st.subheader("Detailed Breakdown")
    for _, row in df_out.iterrows():
        chart_data = pd.DataFrame({
            "class": CLASS_NAMES,
            "probability": [row["p_no"], row["p_unsure"], row["p_yes"]],
        }).set_index("class")
        st.markdown(f"**{row['MobilityAid']}** — Decision: `{row['decision']}`")
        st.bar_chart(chart_data)
