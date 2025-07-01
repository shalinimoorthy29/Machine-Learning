import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------- Load model ------------------------- #
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "output_model/unet_trained_model.h5",
        compile=False          # no need to load training heads
    )

model = load_model()

# ------------------------- Sidebar ---------------------------- #
st.sidebar.header("⚙️  Options")
max_files = st.sidebar.slider(
    "Max files to process",
    min_value=1,
    max_value=20,
    value=5,
    help="Safety cap if a huge batch is dragged in."
)

# ------------------------- Main page -------------------------- #
st.title("🧠 Cell Segmentation using U-Net")
st.write(
    "Upload **one or many** phase-contrast images "
    "(they’ll be auto-resized to 512 × 512) to obtain predicted masks."
)

uploaded_files = st.file_uploader(
    "Upload Image(s)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)

# ------------------------- Inference loop --------------------- #
if uploaded_files:
    if len(uploaded_files) > max_files:
        st.warning(f"Showing first {max_files} of {len(uploaded_files)} files.")
        uploaded_files = uploaded_files[:max_files]

    for up in uploaded_files:
        with st.expander(f"📄 {up.name}", expanded=False):
            # --- show original -------------------------------- #
            image = Image.open(up).convert("RGB").resize((512, 512))
            st.image(image, caption="Input image", use_container_width=True)

            # --- preprocess & predict ------------------------- #
            arr = np.array(image, dtype=np.float32) / 255.0
            pred = model.predict(arr[None, ...])[0, :, :, 0]
            mask = (pred > 0.5).astype(np.uint8)

            # --- display mask --------------------------------- #
            st.subheader("Predicted mask")
            fig, ax = plt.subplots()
            ax.imshow(mask, cmap="gray")
            ax.axis("off")
            st.pyplot(fig)
