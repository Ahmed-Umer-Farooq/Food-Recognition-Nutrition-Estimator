import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image, ImageOps
from tensorflow.keras import backend as K

# --- CONFIGURATION ---
# Must match training NORM_FACTORS exactly
NORM_FACTORS = np.array([1200.0, 100.0, 150.0, 100.0], dtype=np.float32)
IMG_SIZE = 260 # Must match training IMG_SIZE

# --- CUSTOM OBJECTS FOR LOADING ---
def f1_score(y_true, y_pred): return 0
def binary_focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred): return 0
    return focal_loss_fixed

# --- LOAD SYSTEM ---
@st.cache_resource
def load_system():
    if not os.path.exists('best_balanced_model_old.keras'):
        st.error("Model file not found!")
        return None, None

    with open('ingredients.json', 'r') as f:
        classes = json.load(f)

    custom_objects = {
        'f1_score': f1_score,
        'focal_loss_fixed': binary_focal_loss()
    }

    model = tf.keras.models.load_model('best_balanced_model.keras', custom_objects=custom_objects)
    return model, classes

st.set_page_config(page_title="Balanced Food AI", page_icon="🍲")
st.title("🍲 AI Food Analyzer (Balanced)")

model, class_names = load_system()

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded and model:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input Image", width=350)

    if st.button("Analyze"):
        with st.spinner("Analyzing shapes and textures..."):
            # Preprocess
            x = ImageOps.fit(img, (IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            x = np.array(x)
            x = np.expand_dims(x, 0)

            # Predict
            preds = model.predict(x)
            ing_pred = preds[0][0]
            nut_pred = preds[1][0]

            # --- 1. DENORMALIZE NUTRITION ---
            real_nut = nut_pred * NORM_FACTORS

            st.divider()
            st.subheader("📊 Nutrition")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Calories", f"{real_nut[0]:.0f}")
            c2.metric("Fat", f"{real_nut[1]:.1f}g")
            c3.metric("Carbs", f"{real_nut[2]:.1f}g")
            c4.metric("Protein", f"{real_nut[3]:.1f}g")

            # --- 2. INGREDIENTS ---
            st.divider()
            st.subheader("🥕 Ingredients")

            # Sort and Filter
            idxs = np.argsort(ing_pred)[::-1]
            found_any = False

            for i in idxs[:15]: # Check top 15
                conf = ing_pred[i]
                # Lower threshold slightly as model is now balanced/conservative
                if conf > 0.30:
                    found_any = True
                    name = class_names[i]
                    st.write(f"**{name.title()}** ({conf*100:.1f}%)")
                    st.progress(float(conf))

            if not found_any:
                st.warning("No ingredients detected with high confidence.")