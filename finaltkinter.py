import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras.src.legacy.saving import legacy_h5_format
import os

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
        messagebox.showerror("Error", f"Failed to load Age/Gender model:\n{e}")
        return None

def load_hair_model():
    try:
        model_path = r"D:\age-gender-identification\Age_Gender\hair_length_model.h5"
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load Hair model:\n{e}")
        return None


age_gender_model = load_age_gender_model()
hair_model = load_hair_model()

# ==============================
# Preprocessing
# ==============================
def preprocess_image_age_gender(uploaded_image):
    image = uploaded_image.convert("L")  # grayscale
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)  # (128,128,1)
    return np.expand_dims(image_array, axis=0)  # (1,128,128,1)

def preprocess_image_hair(uploaded_image):
    image = uploaded_image.convert("RGB")
    image = image.resize((128, 128))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)  # (1,128,128,3)

# ==============================
# Predictions
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
        return "Short Hair", float(pred)
    else:
        return "Long Hair", float(1 - pred)

# ==============================
# Main Application
# ==============================
class AgeGenderHairApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Age, Gender & Hair Detector 👤")
        self.root.geometry("600x700")
        self.root.configure(bg="#f0f4f8")

        # Title
        self.title_label = tk.Label(
            root, text="Age, Gender & Hair Detector",
            font=("Helvetica", 20, "bold"), fg="#1E3A8A", bg="#f0f4f8"
        )
        self.title_label.pack(pady=20)

        # Image Display
        self.image_label = tk.Label(root, bg="#f0f4f8")
        self.image_label.pack(pady=10)

        # Results
        self.result_text = tk.Text(root, height=12, width=60, font=("Arial", 12))
        self.result_text.pack(pady=10)

        # Buttons
        self.upload_btn = tk.Button(root, text="Upload Image", command=self.upload_image, bg="#2563EB", fg="white", font=("Arial", 12, "bold"))
        self.upload_btn.pack(pady=10)

        self.analyze_btn = tk.Button(root, text="Detect Age, Gender & Hair", command=self.analyze_image, bg="#10B981", fg="white", font=("Arial", 12, "bold"))
        self.analyze_btn.pack(pady=10)

        # Store selected image
        self.selected_image = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        self.selected_image = Image.open(file_path)

        # Show image in Tkinter
        img_resized = self.selected_image.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img_resized)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"Uploaded: {os.path.basename(file_path)}\n")

    def analyze_image(self):
        if self.selected_image is None:
            messagebox.showinfo("Info", "Please upload an image first.")
            return

        if age_gender_model is None or hair_model is None:
            messagebox.showerror("Error", "Models not loaded properly.")
            return

        # Preprocess
        img_age_gender = preprocess_image_age_gender(self.selected_image)
        img_hair = preprocess_image_hair(self.selected_image)

        # Predict
        age, gender, conf = predict_age_gender(age_gender_model, img_age_gender)
        hair_label, hair_conf = predict_hair_length(hair_model, img_hair)

        # Override Logic
        if 20 <= age <= 30:
            final_gender = "Female" if hair_label == "Long Hair" else "Male"
            logic_note = "Override applied (age 20–30, gender decided by hair)"
        else:
            final_gender = gender
            logic_note = "Normal prediction"

        # Show Results
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, f"Age: {age}\n")
        self.result_text.insert(tk.END, f"Original Gender: {gender} (Conf: {conf:.2%})\n")
        self.result_text.insert(tk.END, f"Final Gender: {final_gender} ({logic_note})\n")
        self.result_text.insert(tk.END, f"Hair: {hair_label} (Conf: {hair_conf:.2%})\n")


# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    root = tk.Tk()
    app = AgeGenderHairApp(root)
    root.mainloop()
