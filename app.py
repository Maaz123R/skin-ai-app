import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("skin_disease_model.keras")

class_names = {
    0: "Melanocytic nevi",
    1: "Melanoma",
    2: "Benign keratosis",
    3: "Basal cell carcinoma",
    4: "Actinic keratoses",
    5: "Vascular lesions",
    6: "Dermatofibroma"
}

disease_info = {
    "Melanocytic nevi": {
        "description": "Common moles that are usually benign.",
        "risk": "Low",
        "advice": "Monitor for changes in size, shape, or color."
    },
    "Melanoma": {
        "description": "A serious form of skin cancer that requires medical attention.",
        "risk": "High",
        "advice": "Consult a dermatologist immediately."
    },
    "Benign keratosis": {
        "description": "Non-cancerous skin growths.",
        "risk": "Low",
        "advice": "Usually harmless but monitor for changes."
    },
    "Basal cell carcinoma": {
        "description": "Common skin cancer that grows slowly.",
        "risk": "Medium",
        "advice": "Consult a dermatologist for evaluation."
    },
    "Actinic keratoses": {
        "description": "Pre-cancerous skin lesions caused by sun damage.",
        "risk": "Medium",
        "advice": "Medical consultation recommended."
    },
    "Vascular lesions": {
        "description": "Blood vessel-related skin growths.",
        "risk": "Low",
        "advice": "Usually harmless."
    },
    "Dermatofibroma": {
        "description": "Benign skin nodules.",
        "risk": "Low",
        "advice": "Typically harmless."
    }
}



st.title("🩺 AI Skin Disease Classification System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image_resized = image.resize((28, 28))
    image_array = np.array(image_resized) / 255.0
    image_array = image_array.reshape(1, 28, 28, 3)

    # Predict
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    # Show results
    st.subheader("🔍 Prediction Result")
    st.success(f"Predicted Disease: {class_names[predicted_class]}")
    st.progress(int(confidence))
    st.write(f"Confidence: {confidence:.2f}%")

    predicted_name = class_names[predicted_class]
    info = disease_info[predicted_name]

    st.subheader("📖 Disease Information")
    st.write(f"**Description:** {info['description']}")
    st.write(f"**Risk Level:** {info['risk']}")
    st.write(f"**Recommendation:** {info['advice']}")
