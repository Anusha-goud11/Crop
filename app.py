import streamlit as st
import sqlite3
import datetime
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from deep_translator import GoogleTranslator

# ---------------- Database Functions ----------------
def init_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  prediction TEXT,
                  confidence REAL)''')
    conn.commit()
    conn.close()

def save_prediction(filename, prediction, confidence):
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("INSERT INTO predictions (filename, prediction, confidence) VALUES (?, ?, ?)",
              (filename, prediction, confidence))
    conn.commit()
    conn.close()

def fetch_predictions():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT filename, prediction, confidence FROM predictions ORDER BY id DESC")
    data = c.fetchall()
    conn.close()
    return data

# ---------------- Load Model ----------------
model = load_model("model_final.h5")
class_labels = ["Bacterial Leaf Blight", "Brown Spot", "Healthy Rice Leaf", "Leaf Blast", "Leaf scald", "Sheath Blight"]

# Remedies dictionary
remedies = {
    "Bacterial Leaf Blight": "Use resistant varieties, avoid overhead irrigation.",
    "Brown Spot": "Apply fungicide, use balanced fertilizer.",
    "Healthy Rice Leaf": "No action needed, crop is healthy.",
    "Leaf Blast": "Reduce nitrogen, apply fungicide spray.",
    "Leaf scald": "Plant resistant varieties, ensure good field drainage.",
    "Sheath Blight": "Avoid dense planting, use fungicide treatment."
}

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Leaf Disease Detection", page_icon="🌿", layout="centered")

st.title("🌿 Leaf Disease Detection with Remedies & Translations")
st.write("Upload a rice leaf image to detect disease, get remedies, and translations (Telugu & Hindi).")

# Initialize database
init_db()

# File uploader
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Leaf", use_column_width=True)
    st.success("✅ Image uploaded successfully!")

    # Preprocess image
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    preds = model.predict(img_array)
    predicted_class = class_labels[np.argmax(preds)]
    confidence = float(np.max(preds) * 100)

    # Remedy
    remedy = remedies.get(predicted_class, "No remedy available.")

    # 🔹 Translations
    prediction_te = GoogleTranslator(source="en", target="te").translate(predicted_class)
    remedy_te = GoogleTranslator(source="en", target="te").translate(remedy)

    prediction_hi = GoogleTranslator(source="en", target="hi").translate(predicted_class)
    remedy_hi = GoogleTranslator(source="en", target="hi").translate(remedy)

    # Show Results
    st.subheader("🔎 Prediction Result")
    st.write(f"**Prediction (English): {predicted_class}**")
    st.write(f"**Confidence:** {confidence:.2f}%")
    st.write(f"**Remedy (English):** {remedy}")

    st.markdown("### 🌍 Translation")
    st.write(f"**Telugu:** {prediction_te} → {remedy_te}")
    st.write(f"**Hindi:** {prediction_hi} → {remedy_hi}")

    # Save prediction to database
    save_prediction(uploaded_file.name, predicted_class, confidence)

# Show past predictions
st.subheader("📊 Past Predictions")
rows = fetch_predictions()
if rows:
    for row in rows:
       st.write(f"📂 {row[0]} | 🏷 {row[1]} | 🎯 {row[2]:.2f}%")

else:
    st.info("No predictions yet. Upload an image to start!")
