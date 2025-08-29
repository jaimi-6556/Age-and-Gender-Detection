import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras.src.legacy.saving import legacy_h5_format

# ==============================
# Load Models
# ==============================
def load_age_gender_model():
    try:
        model_path = r"D:\age-gender-identification\Age_Gender\age_gender_model.h5"
        model = legacy_h5_format.load_model_from_hdf5(
            model_path, custom_objects={"mae": "mae"}
        )
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Error loading Age/Gender model:\n{e}")
        return None


def load_hair_model():
    try:
        model_path = r"D:\age-gender-identification\Age_Gender\hair_length_model.h5"
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Error loading Hair model:\n{e}")
        return None


age_gender_model = load_age_gender_model()
hair_model = load_hair_model()

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
    if pred > 0.5:
        return "Long Hair", float(pred)
    else:
        return "Short Hair", float(1 - pred)

# ==============================
# Tkinter GUI
# ==============================
root = tk.Tk()
root.title("Age, Gender & Hair Detector")
root.geometry("800x600")
root.configure(bg="#F3F4F6")

title_label = tk.Label(
    root,
    text="👤 Age, Gender & Hair Detector",
    font=("Helvetica", 20, "bold"),
    fg="#1E3A8A",
    bg="#F3F4F6",
)
title_label.pack(pady=20)

# Image display
image_label = tk.Label(root, bg="#F3F4F6")
image_label.pack(pady=10)

# Results
result_text = tk.Text(root, height=10, width=80, wrap="word", font=("Helvetica", 12))
result_text.pack(pady=10)

# ==============================
# Button Actions
# ==============================
def open_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
    )
    if not file_path:
        return

    try:
        # Load image
        img = Image.open(file_path)
        tk_img = ImageTk.PhotoImage(img.resize((250, 250)))
        image_label.config(image=tk_img)
        image_label.image = tk_img

        # Preprocess
        processed_age_gender = preprocess_image_age_gender(img)
        processed_hair = preprocess_image_hair(img)

        # Predict
        age, gender, conf = predict_age_gender(age_gender_model, processed_age_gender)
        hair_label, hair_conf = predict_hair_length(hair_model, processed_hair)

        # Special rule for ages 20–30 (hair overrides gender)
        final_gender = gender
        if 20 <= age <= 30:
            if hair_label == "Long Hair":
                final_gender = "Female"
            else:
                final_gender = "Male"

        # Display results
        result_text.delete("1.0", tk.END)
        result_text.insert(
            tk.END,
            f"Results for {file_path.split('/')[-1]}:\n\n"
            f"Age: {age}\n"
            f"Detected Gender: {gender} (Conf: {conf:.2%})\n"
            f"Hair: {hair_label} (Conf: {hair_conf:.2%})\n"
            f"\n🔹 Final Gender (Rule Applied): {final_gender}\n",
        )

    except Exception as e:
        messagebox.showerror("Error", f"Error processing image:\n{e}")


upload_button = tk.Button(
    root,
    text="Upload Image",
    command=open_file,
    bg="#2563EB",
    fg="white",
    font=("Helvetica", 12, "bold"),
    relief="flat",
    padx=10,
    pady=5,
)
upload_button.pack(pady=10)

footer = tk.Label(
    root,
    text="Powered by Ahir Jaimi 🧑‍💻",
    font=("Helvetica", 10),
    bg="#F3F4F6",
    fg="gray",
)
footer.pack(side="bottom", pady=10)

root.mainloop()


