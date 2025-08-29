'''import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load Models
age_gender_model = load_model("age_gender_model.h5")
hair_length_model = load_model("hair_length_model.h5")

# Class labels
gender_labels = ["Male", "Female"]
hair_labels = ["Short Hair", "Long Hair"]

# Preprocessing function
def preprocess_image(image, target_size=(200, 200)):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def main():
    st.title("👱 Age, Gender & Hair Detection System")
    st.write("Upload an image to predict Age, Gender, and Hair Length.")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        st.image(image, channels="BGR", caption="Uploaded Image", use_column_width=True)

        # Preprocess for both models
        processed_img = preprocess_image(image, target_size=(200, 200))

        # --- Hair Prediction ---
        hair_pred = hair_length_model.predict(processed_img)[0][0]
        hair_result = hair_labels[int(hair_pred > 0.5)]

        # --- Age + Gender Prediction ---
        preds = age_gender_model.predict(processed_img)
        age_pred = int(preds[0][0])  # assuming model outputs [age, gender_prob]
        gender_prob = preds[0][1]
        gender_result = gender_labels[int(gender_prob > 0.5)]

        # --- Apply Override Rule ---
        if 20 <= age_pred <= 30 and hair_result == "Long Hair":
            gender_result = "Female (Overridden due to long hair)"

        # Display results
        st.subheader("Prediction Results:")
        st.markdown(f"**Age:** {age_pred} years")
        st.markdown(f"**Gender:** {gender_result}")
        st.markdown(f"**Hair Length:** {hair_result}")

if __name__ == "__main__":
    main()


import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.src.legacy.saving import legacy_h5_format

# ===============================
# CONFIG
# ===============================
st.set_page_config(
    page_title="Age, Gender & Hair Detector",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ===============================
# CSS
# ===============================
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .result-text {
        font-size: 1.3rem;
        font-weight: 500;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .image-container {
        margin-bottom: 2rem;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(237, 242, 247, 0.5);
    }
    .app-footer {
        text-align: center;
        margin-top: 2rem;
        opacity: 0.7;
    }
    .stButton>button {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1E40AF;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ===============================
# MODEL LOADING
# ===============================
@st.cache_resource
def load_age_gender_model():
    model_path = r"D:\age-gender-identification\Age_Gender\age_gender_model.h5"
    model = legacy_h5_format.load_model_from_hdf5(
        model_path, custom_objects={"mae": "mae"}
    )
    return model

@st.cache_resource
def load_hair_model():
    model_path = r"D:\age-gender-identification\Age_Gender\hair_length_model.h5"
    model = tf.keras.models.load_model(model_path)
    return model

# ===============================
# IMAGE PREPROCESSING
# ===============================
def preprocess_for_age_gender(uploaded_image):
    image = uploaded_image.convert("L")   # grayscale
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # add channel
    return np.expand_dims(image_array, axis=0)

def preprocess_for_hair(uploaded_image):
    image = uploaded_image.convert("RGB")
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

# ===============================
# PREDICTION FUNCTIONS
# ===============================
def predict_age_gender(model, image_array):
    predictions = model.predict(image_array)
    predicted_age = int(np.round(predictions[1][0]))
    gender_prob = predictions[0][0]
    predicted_gender = "Female" if gender_prob > 0.5 else "Male"
    gender_confidence = gender_prob if predicted_gender == "Female" else 1 - gender_prob
    return predicted_age, predicted_gender, float(gender_confidence)

def predict_hair_length(model, image_array):
    pred = model.predict(image_array)[0][0]
    return "Long Hair" if pred > 0.5 else "Short Hair", float(pred if pred > 0.5 else 1 - pred)

def predict_hair_length(model, image_array):
    pred = model.predict(image_array)[0][0]
    st.write("🔍 Raw hair prediction value:", pred)  # debug
    if pred > 0.5:
        return "Long Hair", float(pred)
    else:
        return "Short Hair", float(1 - pred)


# ===============================
# HELPER
# ===============================
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ===============================
# MAIN APP
# ===============================
def main():
    st.markdown('<div class="main-header">Age, Gender & Hair Detector</div>', unsafe_allow_html=True)

    # Load models
    with st.spinner("Loading models... Please wait."):
        age_gender_model = load_age_gender_model()
        hair_model = load_hair_model()

    # File uploader
    st.markdown('<div class="sub-header">Upload Images</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose one or more images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    # Process button
    if uploaded_files and st.button("Detect"):
        with st.spinner("Analyzing images..."):
            for i, uploaded_file in enumerate(uploaded_files):
                with st.container():
                    st.markdown(f'<div class="image-container">', unsafe_allow_html=True)
                    st.markdown(f"<h3>Image {i+1}</h3>", unsafe_allow_html=True)

                    col1, col2 = st.columns([1, 1])
                    image = Image.open(uploaded_file)
                    col1.image(image, caption=f"Image {i+1}: {uploaded_file.name}", use_container_width=True)

                    # Predictions
                    processed_age_gender = preprocess_for_age_gender(image)
                    age, gender, confidence = predict_age_gender(age_gender_model, processed_age_gender)

                    processed_hair = preprocess_for_hair(image)
                    hair_label, hair_conf = predict_hair_length(hair_model, processed_hair)

                    # Display results
                    col2.markdown('<div class="sub-header">Results:</div>', unsafe_allow_html=True)
                    col2.markdown(
                        f'<div class="result-text" style="background-color: rgba(37, 99, 235, 0.1);">Age: {age}</div>',
                        unsafe_allow_html=True,
                    )
                    gender_color = "#9F7AEA" if gender == "Female" else "#4F46E5"
                    col2.markdown(
                        f'<div class="result-text" style="background-color: rgba({", ".join(map(str, hex_to_rgb(gender_color)))}, 0.1);">'
                        f"Gender: {gender}<br><small>Confidence: {confidence:.2%}</small></div>",
                        unsafe_allow_html=True,
                    )
                    hair_color = "#16A34A" if hair_label == "Long Hair" else "#DC2626"
                    col2.markdown(
                        f'<div class="result-text" style="background-color: rgba({", ".join(map(str, hex_to_rgb(hair_color)))}, 0.1);">'
                        f"Hair Length: {hair_label}<br><small>Confidence: {hair_conf:.2%}</small></div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("</div>", unsafe_allow_html=True)
                    if i < len(uploaded_files) - 1:
                        st.markdown("<hr>", unsafe_allow_html=True)

    elif st.button("Detect", key="no_image_button"):
        st.info("Please upload one or more images first.")

    st.markdown('<div class="app-footer">Powered by Ahir Jaimi 🧑‍💻</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()'''



import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.src.legacy.saving import legacy_h5_format

# ==============================
# Streamlit Page Config
# ==============================
st.set_page_config(
    page_title="Age, Gender & Hair Detector",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ==============================
# CSS Styling
# ==============================
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #2563EB;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }
        .result-text {
            font-size: 1.5rem;
            font-weight: 500;
            padding: 0.75rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
        }
        .image-container {
            margin-bottom: 2rem;
            padding: 1rem;
            border-radius: 0.5rem;
            background-color: rgba(237, 242, 247, 0.5);
        }
        .app-footer {
            text-align: center;
            margin-top: 2rem;
            opacity: 0.7;
        }
        .stButton>button {
            background-color: #2563EB;
            color: white;
            font-weight: bold;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1E40AF;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ==============================
# Load Models (cached)
# ==============================
@st.cache_resource
def load_age_gender_model():
    try:
        model_path = r"D:\age-gender-identification\Age_Gender\age_gender_model.h5"
        model = legacy_h5_format.load_model_from_hdf5(
            model_path, custom_objects={"mae": "mae"}
        )
        return model
    except Exception as e:
        st.error(f"Error loading Age/Gender model: {e}")
        return None


@st.cache_resource
def load_hair_model():
    try:
        model_path = r"D:\age-gender-identification\Age_Gender\hair_length_model.h5"
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading Hair model: {e}")
        return None

# ==============================
# Preprocessing Functions
# ==============================
def preprocess_image_age_gender(uploaded_image):
    image = uploaded_image.convert("L")  # grayscale
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # (128,128,1)
    return np.expand_dims(image_array, axis=0)  # (1,128,128,1)


def preprocess_image_hair(uploaded_image):
    image = uploaded_image.convert("RGB")  # color
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)  # (1,128,128,3)

# ==============================
# Prediction Functions
# ==============================
def predict_age_gender(model, image_array):
    predictions = model.predict(image_array)
    predicted_age = int(np.round(predictions[1][0]))  # age
    gender_prob = predictions[0][0]  # gender
    predicted_gender = "Female" if gender_prob > 0.5 else "Male"
    gender_confidence = gender_prob if predicted_gender == "Female" else 1 - gender_prob
    return predicted_age, predicted_gender, float(gender_confidence)


def predict_hair_length(model, image_array):
    pred = model.predict(image_array)[0][0]
    # ⚠️ long=0, short=1 (alphabetical labeling in flow_from_directory)
    if pred > 0.5:
        return "Short Hair", float(pred)
    else:
        return "Long Hair", float(1 - pred)

# ==============================
# Helper
# ==============================
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# ==============================
# Main App
# ==============================
def main():
    st.markdown('<div class="main-header">Age, Gender & Hair Detector</div>', unsafe_allow_html=True)

    # Load Models
    with st.spinner("Loading models..."):
        age_gender_model = load_age_gender_model()
        hair_model = load_hair_model()

    if age_gender_model is None or hair_model is None:
        st.warning("Please check that both models exist at the specified paths.")
        return

    # File Upload
    st.markdown('<div class="sub-header">Upload Images</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Choose one or more images...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Detect Age, Gender & Hair"):
        with st.spinner("Analyzing images..."):
            for i, uploaded_file in enumerate(uploaded_files):
                with st.container():
                    st.markdown(f'<div class="image-container">', unsafe_allow_html=True)
                    st.markdown(f"<h3>Image {i+1}</h3>", unsafe_allow_html=True)

                    col1, col2 = st.columns([1, 1])
                    image = Image.open(uploaded_file)

                    # Display image
                    col1.image(image, caption=f"Image {i+1}: {uploaded_file.name}", use_container_width=True)

                    # Predictions
                    processed_age_gender = preprocess_image_age_gender(image)
                    processed_hair = preprocess_image_hair(image)

                    age, gender, conf = predict_age_gender(age_gender_model, processed_age_gender)
                    hair_label, hair_conf = predict_hair_length(hair_model, processed_hair)

                    # Results
                    col2.markdown('<div class="sub-header">Results:</div>', unsafe_allow_html=True)

                    col2.markdown(
                        f'<div class="result-text" style="background-color: rgba(37, 99, 235, 0.1);">Age: {age}</div>',
                        unsafe_allow_html=True,
                    )

                    gender_color = "#9F7AEA" if gender == "Female" else "#4F46E5"
                    col2.markdown(
                        f'<div class="result-text" style="background-color: rgba({", ".join(map(str, hex_to_rgb(gender_color)))}, 0.1);">'
                        f"Gender: {gender}<br><small>Confidence: {conf:.2%}</small></div>",
                        unsafe_allow_html=True,
                    )

                    hair_color = "#10B981" if hair_label == "Short Hair" else "#F59E0B"
                    col2.markdown(
                        f'<div class="result-text" style="background-color: rgba({", ".join(map(str, hex_to_rgb(hair_color)))}, 0.1);">'
                        f"Hair: {hair_label}<br><small>Confidence: {hair_conf:.2%}</small></div>",
                        unsafe_allow_html=True,
                    )

                    st.markdown("</div>", unsafe_allow_html=True)
                    if i < len(uploaded_files) - 1:
                        st.markdown("<hr>", unsafe_allow_html=True)

    elif st.button("Detect Age, Gender & Hair", key="no_image_button"):
        st.info("Please upload one or more images first.")

    st.markdown('<div class="app-footer">Powered by Ahir Jaimi 🧑‍💻</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()





