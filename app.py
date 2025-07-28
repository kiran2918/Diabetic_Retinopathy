import streamlit as st
import numpy as np
import cv2
import joblib
from PIL import Image
from skimage import exposure
import os
import gdown

# Streamlit Page Configuration
st.set_page_config(
    page_title="InfernoSight | Retina Classifier",
    page_icon="🩸",
    layout="centered"
)

# Sidebar UI
with st.sidebar:
    st.title("🩸 InfernoSight")
    st.markdown("""
    Detect **Diabetic Retinopathy** (DR) severity from retina scans.

    🔥 **InfernoSight** combines handcrafted preprocessing with an SVM model trained on real patient data.

    #### DR Stages:
    - 0: No DR
    - 1: Mild
    - 2: Moderate
    - 3: Severe
    - 4: Proliferative DR

    -----
    🧠 Built by **King**  
    😈 Summoned from shadows. Diagnoses with precision.
    """)

# Model Loading
model_path = "models/svm_model.pkl"
model_url = "https://drive.google.com/uc?id=1ULij3MiZPSWJ-EZaClFBtHZAmjOL_eIk"

os.makedirs("models", exist_ok=True)

if not os.path.exists(model_path):
    with st.spinner("🔁 Downloading SVM model..."):
        gdown.download(model_url, model_path, quiet=False)

try:
    model = joblib.load(model_path)
    st.success("🔥 Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# App Title
st.title("😈 InfernoSight | Retina Classifier")
st.caption("Built in chaos. Diagnoses with calm.")

# Image Upload
uploaded_file = st.file_uploader("🖼️ Upload a retina image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="📍 Uploaded Retina", use_container_width=True)

    try:
        # Step 1: Grayscale
        image = Image.open(uploaded_file).convert("L")
        image_np = np.array(image)

        # Step 2: Resize to 128x128
        image_resized = cv2.resize(image_np, (128, 128))

        # Step 3: Histogram equalization
        image_eq = exposure.equalize_adapthist(image_resized, clip_limit=0.03)

        # Step 4: Flatten
        image_flat = image_eq.flatten().reshape(1, -1)

        # Step 5: Predict
        prediction = model.predict(image_flat)[0]

        # Prediction Mapping
        class_names = {
            0: "No DR",
            1: "Mild DR",
            2: "Moderate DR",
            3: "Severe DR",
            4: "Proliferative DR"
        }
        risk_level = {
            0: "✅ Low risk",
            1: "⚠️ Mild risk",
            2: "⚠️ Moderate risk",
            3: "❗ High risk – needs attention",
            4: "🚨 Critical – urgent care required"
        }

        # Display Result
        st.markdown("### 🔍 Prediction Result")
        st.success(f"**{class_names[prediction]}**")
        st.info(f"**{risk_level[prediction]}**")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# Footer
st.markdown("---")
st.caption("🧠 Built by King | Summoned from the shadows 😈 | Powered by SVM and hustle.")
