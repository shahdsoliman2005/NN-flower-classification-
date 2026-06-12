import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌸 Flower Recognition",
    page_icon="🌸",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASSES   = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
IMG_SIZE  = 128
CLASS_EMO = {"daisy": "🌼", "dandelion": "🌱", "rose": "🌹",
             "sunflower": "🌻", "tulip": "🌷"}
MODEL_PATHS = {
    "CNN":            "checkpoints/cnn_best.keras",
    "DNN":            "checkpoints/dnn_best.keras",
    "MobileNetV2 TL": "checkpoints/tl_best.keras",
}

# ── Lazy TF import — only imported once, not on every rerun ──────────────────
@st.cache_resource(show_spinner=False)
def _import_tf():
    """Import TensorFlow once and cache the module object."""
    import tensorflow as tf
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    return tf

# ── Model loader (one cache entry per model name) ────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    """
    Load a .keras model and cache it permanently in the Streamlit process.
    - Loaded exactly ONCE per model_name across all reruns.
    - Graph warmed up with a dummy forward pass so first real prediction is instant.
    """
    tf = _import_tf()
    path = MODEL_PATHS[model_name]

    if not os.path.exists(path):
        return None, (
            f"Model file **{path}** not found.  \n"
            "Train the notebook first and place `.keras` files in `checkpoints/`."
        )

    model = tf.keras.models.load_model(path)

    # Warm-up: trace the graph once so first real prediction is instant
    dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
    model(dummy, training=False)

    return model, None


# ── Preprocessing (cached per image bytes to avoid redundant work) ────────────
@st.cache_data(show_spinner=False, max_entries=20)
def preprocess(image_bytes: bytes) -> np.ndarray:
    """
    Convert raw uploaded bytes to a normalised (1,128,128,3) float32 array.
    Cached by content hash — same image reused across reruns/tab switches.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(img.astype("float32") / 255.0, axis=0)


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🌸 Flower Recognition")
st.caption("Classify flowers with DNN, CNN, or MobileNetV2 Transfer Learning")

with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox(
        "Choose a model",
        ["CNN", "MobileNetV2 TL", "DNN"],
        index=0,
        help="CNN and MobileNetV2 TL recommended for best accuracy.",
    )

    model_path = MODEL_PATHS[model_choice]
    if os.path.exists(model_path):
        st.success(f"✅ `{model_path}` found")
    else:
        st.warning(f"⚠️ `{model_path}` not found")

    st.markdown("---")
    st.markdown(
        "**Classes**  \n" +
        "  \n".join(f"{CLASS_EMO[c]} {c.capitalize()}" for c in CLASSES)
    )
    st.markdown("---")
    st.markdown(
        "**Model guide**\n"
        "- **CNN** — residual conv net with SeparableConv\n"
        "- **MobileNetV2 TL** — fine-tuned ImageNet weights\n"
        "- **DNN** — fully-connected baseline"
    )

tab_upload, tab_camera, tab_about = st.tabs(["📤 Upload", "📷 Camera", "ℹ️ About"])


# ── Inference helper ──────────────────────────────────────────────────────────
def run_inference(image: Image.Image, source_bytes: bytes):
    with st.spinner(f"Loading {model_choice}…"):
        model, err = load_model(model_choice)
    if err:
        st.error(err)
        return

    tensor = preprocess(source_bytes)

    with st.spinner("Classifying…"):
        probs = model(tensor, training=False).numpy()[0]

    top_idx   = int(np.argmax(probs))
    top_label = CLASSES[top_idx]
    top_prob  = float(probs[top_idx])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(image, use_container_width=True, caption="Input image")
    with col2:
        st.markdown(f"### {CLASS_EMO[top_label]} {top_label.capitalize()}")
        st.metric("Confidence", f"{top_prob * 100:.1f}%")
        if top_prob >= 0.80:
            st.success("High confidence prediction ✅")
        elif top_prob >= 0.50:
            st.warning("Moderate confidence — result may vary ⚠️")
        else:
            st.error("Low confidence — try a clearer image ❌")

        top3_idx = np.argsort(probs)[::-1][:3]
        st.markdown("**Top 3 predictions:**")
        for idx in top3_idx:
            lbl = CLASSES[idx]
            pct = probs[idx] * 100
            st.markdown(f"{CLASS_EMO[lbl]} **{lbl.capitalize()}** — `{pct:.1f}%`")

    st.markdown("#### All class probabilities")
    bar_data = {c.capitalize(): float(p) for c, p in zip(CLASSES, probs)}
    st.bar_chart(bar_data, height=200)


# ── Upload tab ────────────────────────────────────────────────────────────────
with tab_upload:
    uploaded = st.file_uploader(
        "Upload a flower image",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        raw_bytes = uploaded.read()
        image = Image.open(uploaded).convert("RGB")
        run_inference(image, raw_bytes)
    else:
        st.info("👆 Upload a JPG, PNG, or WebP image to get started.")

# ── Camera tab ────────────────────────────────────────────────────────────────
with tab_camera:
    camera_img = st.camera_input("Take a photo of a flower")
    if camera_img:
        raw_bytes = camera_img.read()
        image = Image.open(camera_img).convert("RGB")
        run_inference(image, raw_bytes)

# ── About tab ─────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
## About this app

Classify flower images into **5 categories**:
Daisy 🌼, Dandelion 🌱, Rose 🌹, Sunflower 🌻, and Tulip 🌷.

### Dataset
[Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition)
dataset from Kaggle (~4 300 images across 5 classes).

### Models

| Model | Architecture | Notes |
|---|---|---|
| **CNN** | Residual blocks + SeparableConv | Skip connections + GAP head |
| **MobileNetV2 TL** | MobileNetV2 + deeper 512→256 head | Fine-tunes top 50 layers |
| **DNN** | 4-layer Dense (512→384→256→128) | L2 + LeakyReLU + label-smoothing |

### Speed improvements in this app
- TensorFlow imported **once** and cached — no reimport on reruns
- Models cached per name — loaded **once**, reused across all predictions
- Graph **warmed up** at load time — first prediction is instant
- Preprocessed tensors **cached by image hash** — same photo never reprocessed twice
""")
