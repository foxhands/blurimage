import os
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk
import cv2
import face_recognition
import pickle
from PIL import Image, ImageTk


class Logger:
    """Simple logger class to handle logging messages."""

    def __init__(self, text_widget):
        self.text_widget = text_widget

    def log(self, message):
        self.text_widget.insert(tk.END, message + '\n')
        self.text_widget.see(tk.END)


class Face:
    """Class to represent a detected face and its operations."""

    def __init__(self, image, location):
        self.image = image
        self.location = location

    def save(self, face_name, output_dir):
        """Save detected face in the specified directory."""
        top, right, bottom, left = self.location
        face_img = self.image[top:bottom, left:right]

        # Generate unique file name
        face_files = [f for f in os.listdir(output_dir) if f.startswith(face_name)]

        # Get the highest number used in existing files
        face_numbers = [int(f.split('_')[-1].split('.')[0]) for f in face_files if f.endswith('.jpg')]
        face_number = max(face_numbers) + 1 if face_numbers else 0  # Start from current max or 0

        # Create file path with unique number
        file_path = os.path.join(output_dir, f"{face_name}_{face_number}.jpg")
        cv2.imwrite(file_path, face_img)
        return file_path


class FaceSelectionApp:
    def __init__(self, master):
        self.master = master
        master.title("Face Selection")

        self.img_originals = []  # List to store original images
        self.faces = []  # List to store detected faces
        self.thumbnail_buttons = []  # Buttons for face previews
        self.face_images = []  # Initialize face_images list

        # Create a text field for logs
        self.log_text = tk.Text(master, height=10, width=70)
        self.log_text.pack()
        self.logger = Logger(self.log_text)

        self.model_name = None  # Хранение имени модели

        # Поле ввода для имени модели
        self.model_name_label = tk.Label(master, text="Enter model name:")
        self.model_name_label.pack()

        self.model_name_entry = tk.Entry(master)
        self.model_name_entry.pack(pady=5)

        # Create frame for buttons
        self.frame_buttons = ttk.Frame(master)
        self.frame_buttons.pack(pady=10)

        self.upload_button = tk.Button(self.frame_buttons, text="Upload Images", command=self.upload_images)
        self.upload_button.pack(side="left", padx=5)

        self.clear_button = tk.Button(self.frame_buttons, text="Clear Selection", command=self.clear_selection)
        self.clear_button.pack(side="left", padx=5)

    def clear_selection(self):
        """Clear the selected faces and reset the application state."""
        self.logger.log("Clearing selections.")
        self.faces.clear()
        self.img_originals.clear()
        self.thumbnail_buttons.clear()
        self.logger.log("Selections cleared.")

    def upload_images(self):
        """Open file dialog to upload images."""
        self.model_name = self.model_name_entry.get()  # Получаем имя модели из поля ввода
        if not self.model_name:
            self.logger.log("Model name is required before uploading images.")
            return

        self.logger.log("Opening file dialog to upload images.")
        file_paths = filedialog.askopenfilenames(title="Select Images",
                                                 filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_paths:
            self.logger.log(f"Selected {len(file_paths)} images.")
            for path in file_paths:
                self.process_image(path)
        else:
            self.logger.log("No images were selected.")

    def process_image(self, image_path):
        """Process each uploaded image."""
        self.logger.log(f"Processing image: {image_path}")
        try:
            img_original = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

            # Find all faces in the image
            face_locations = face_recognition.face_locations(img_rgb)
            self.logger.log(f"Found {len(face_locations)} faces in {image_path}.")

            if face_locations:
                self.img_originals.append(img_original)  # Save the original image
                for loc in face_locations:
                    self.faces.append(Face(img_original, loc))  # Save each detected face

                self.show_face_thumbnails()
            else:
                self.logger.log(f"No faces found in {image_path}.")
        except Exception as e:
            self.logger.log(f"Error processing image {image_path}: {str(e)}")

    def show_face_thumbnails(self):
        """Display face thumbnails for selection."""
        self.logger.log("Entering show_face_thumbnails method.")

        # Remove old buttons and reset lists
        self.logger.log(f"Removing {len(self.thumbnail_buttons)} old buttons (if any).")
        for btn in self.thumbnail_buttons:
            btn.destroy()
        self.thumbnail_buttons = []
        self.face_images.clear()  # Clear the list of images

        if not self.faces:
            self.logger.log("No faces detected to display thumbnails.")
            return

        self.logger.log(f"Preparing to display {len(self.faces)} faces.")

        for idx, face in enumerate(self.faces):
            try:
                self.logger.log(f"Processing face {idx + 1}/{len(self.faces)}.")

                top, right, bottom, left = face.location
                self.logger.log(f"Face {idx + 1} location: top={top}, right={right}, bottom={bottom}, left={left}.")

                # Crop the face image
                face_img = face.image[top:bottom, left:right]
                self.logger.log(f"Face {idx + 1} image cropped successfully.")

                # Resize for preview
                face_img_resized = cv2.resize(face_img, (100, 100))
                self.logger.log(f"Face {idx + 1} resized to 100x100.")

                # Convert image for Tkinter
                face_img_pil = Image.fromarray(cv2.cvtColor(face_img_resized, cv2.COLOR_BGR2RGB))
                face_img_tk = ImageTk.PhotoImage(face_img_pil)

                # Save the image in the list
                self.face_images.append(face_img_tk)  # Store the reference
                self.logger.log(f"Face {idx + 1} thumbnail created and added to memory.")

                # Create a button with the face image
                btn = tk.Button(self.master, image=face_img_tk,
                                command=lambda f=face: self.save_selected_face(f))
                btn.image = face_img_tk  # Keep a reference to the image
                btn.pack(side="left")
                self.thumbnail_buttons.append(btn)
                self.logger.log(f"Button for face {idx + 1} created successfully and packed.")
            except Exception as e:
                self.logger.log(f"Error while processing face {idx + 1}: {str(e)}")

    def save_selected_face(self, face):
        """Save the selected face image."""
        if self.model_name is None:
            self.logger.log("Model name is not set.")
            return

        output_dir = os.path.join("faces", self.model_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.log(f"Created directory for {self.model_name}.")

        try:
            saved_path = face.save(self.model_name, output_dir)
            self.logger.log(f"Face saved as {saved_path}.")
        except Exception as e:
            self.logger.log(f"Error saving face: {str(e)}")

        # Clear faces after saving
        self.faces.clear()
        self.img_originals.clear()

def run(master):
    app = FaceSelectionApp(master)
    master.mainloop()
