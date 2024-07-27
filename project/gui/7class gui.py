import customtkinter as ctk
from tkinter import filedialog
import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
import time

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")

# Initialize model variable
inception_model = None

# Function to load the model
def load_model():
    global inception_model
    inception_model = tf.keras.models.load_model('units/final_best_inceptionv3_model.h5')

# Initially load the model
load_model()

class_names = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis-like lesions', 
               'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']

current_image_path = None
processing = False
last_process_time = 0

def prepare(image_path, img_size):
    img_array = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    new_array = cv2.resize(img_array, (img_size, img_size))
    return new_array.reshape(-1, img_size, img_size, 3) / 255.0

def select_image():
    global current_image_path, last_process_time
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
    if file_path:
        current_time = time.time()
        if current_time - last_process_time > 1:  # 1 second delay
            current_image_path = file_path
            update_image_display(file_path)
            last_process_time = current_time
            root.after(1000, process_image)  # Schedule processing after 1 second
        else:
            print("Please wait before selecting another image.")

def update_image_display(file_path):
    image = Image.open(file_path)
    display_image = image.copy()
    display_image.thumbnail((120, 120))
    ctk_image = ctk.CTkImage(light_image=display_image, dark_image=display_image, size=(120, 120))
    image_label.configure(image=ctk_image, text="")
    image_label.image = ctk_image

def process_image():
    global processing, current_image_path
    if processing:
        root.after(100, process_image)
        return
    
    if current_image_path:
        processing = True
        try:
            # Process with InceptionV3 model
            prepared_image = prepare(current_image_path, 128)
            predictions = inception_model.predict(prepared_image)[0]
            
            update_detailed_classification(predictions)

        except Exception as e:
            print(f"Error processing image: {e}")
        finally:
            processing = False
            current_image_path = None

def update_detailed_classification(predictions):
    sorted_indices = np.argsort(predictions)[::-1]
    
    # Update the main detailed label with the top prediction
    top_prediction = class_names[sorted_indices[0]]
    top_probability = predictions[sorted_indices[0]] * 100
    detailed_label.configure(text=f"Top Prediction:\n{top_prediction}\n{top_probability:.2f}%", 
                             font=("Helvetica", 14, "bold"))
    
    # Update the secondary label with all predictions
    secondary_text = "All predictions:\n"
    for i in range(len(class_names)):
        index = sorted_indices[i]
        secondary_text += f"{class_names[index]}: {predictions[index]*100:.2f}%\n"
    
    secondary_detailed_label.configure(text=secondary_text)

def load_and_resize_bg(image_path, target_width, target_height):
    original = Image.open(image_path)
    original = original.convert('RGBA')
    
    aspect = original.width / original.height
    target_aspect = target_width / target_height
    
    if aspect > target_aspect:
        new_width = int(target_aspect * original.height)
        left = (original.width - new_width) / 2
        original = original.crop((left, 0, left + new_width, original.height))
    elif aspect < target_aspect:
        new_height = int(original.width / target_aspect)
        top = (original.height - new_height) / 2
        original = original.crop((0, top, original.width, top + new_height))
    
    return original.resize((target_width, target_height), Image.Resampling.LANCZOS)

root = ctk.CTk()
root.title("Skin Lesion Classifier")
root.geometry("500x500")  # Reduced height further

bg_image = load_and_resize_bg("R.jpg", 1368, 768)
bg_ctk_image = ctk.CTkImage(light_image=bg_image, dark_image=bg_image, size=(1368, 768))
bg_label = ctk.CTkLabel(root, image=bg_ctk_image, text="")
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a smaller frame
frame = ctk.CTkFrame(root, fg_color="#1E1E1E", corner_radius=10, width=400, height=450)
frame.place(relx=0.5, rely=0.5, anchor="center")

# Adjust the size of the image label
image_label = ctk.CTkLabel(frame, text="No Image Selected", fg_color="#2C2C2C", corner_radius=10, width=120, height=120)
image_label.pack(pady=10)

select_button = ctk.CTkButton(frame, text="Choose Image", command=select_image, font=("Helvetica", 11), corner_radius=10)
select_button.pack(pady=5)

# Add detailed classification label
detailed_label = ctk.CTkLabel(frame, text="Detailed Classification:", font=("Helvetica", 14, "bold"), 
                              fg_color="#2C2C2C", justify="center", wraplength=300)
detailed_label.pack(pady=10)

secondary_detailed_label = ctk.CTkLabel(frame, text="", font=("Helvetica", 11), 
                                        fg_color="#2C2C2C", justify="left", wraplength=300)
secondary_detailed_label.pack(pady=5)

disclaimer_label = ctk.CTkLabel(frame, text="This application is for educational purposes only.", font=("Helvetica", 9), text_color="#FF5252", wraplength=300, justify="center")
disclaimer_label.pack(pady=5)

root.mainloop()