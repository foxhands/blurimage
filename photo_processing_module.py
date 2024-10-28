import os
import tkinter as tk
from tkinter import filedialog
import face_recognition
import json
import cv2
import numpy as np
from mtcnn import MTCNN
import logging
from PIL import Image

# Настройка логирования
logging.basicConfig(level=logging.INFO)

class PhotoProcessingApp:
    def __init__(self):
        self.min_face_size = 5
        self.thresholds = 1.9
        self.selected_faces = None
        self.face_encodings = {}
        self.detector = MTCNN()  # Инициализация детектора MTCNN
        self.load_face_encodings()  # Загрузка кодировок при инициализации

    def load_face_encodings(self):
        """Загрузка кодировок лиц из файлов JSON."""
        faces_folder = 'faces'
        for name in os.listdir(faces_folder):
            json_file_path = os.path.join(faces_folder, name, f"{name}.json")
            if os.path.isfile(json_file_path):
                with open(json_file_path, 'r') as json_file:
                    data = json.load(json_file)
                    self.face_encodings[name] = [face['encodings'] for face in data['files']]
                    logging.info(f"Loaded {len(data['files'])} encodings for {name}.")

    def choose_reference_faces(self):
        """Выбор референсных лиц для определения."""
        choices = list(self.face_encodings.keys())
        selection_window = tk.Toplevel()
        selection_window.title("Select Reference Faces")

        listbox = tk.Listbox(selection_window, selectmode=tk.MULTIPLE)
        listbox.pack(pady=20)
        for choice in choices:
            listbox.insert(tk.END, choice)

        confirm_button = tk.Button(selection_window, text="OK", command=lambda: self.confirm_selection(listbox, selection_window, choices))
        confirm_button.pack(pady=10)

        selection_window.transient()  # Устанавливаем окно как модальное
        selection_window.grab_set()  # Блокируем доступ к родительскому окну
        selection_window.wait_window()  # Ожидание закрытия окна

    def confirm_selection(self, listbox, window, choices):
        """Подтверждение выбора референсных лиц."""
        selected_indices = listbox.curselection()
        self.selected_faces = [choices[i] for i in selected_indices]
        window.destroy()  # Закрываем окно выбора
        logging.info(f"Selected faces: {self.selected_faces}")

    def upload_image(self):
        """Выбор референсных лиц и загрузка изображения для распознавания лиц."""
        self.choose_reference_faces()
        if not self.selected_faces:
            logging.warning("No reference faces selected.")
            return

        reference_encodings = self.get_reference_encodings()
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.process_image(file_path, reference_encodings)  # Передаем кодировки референсных лиц
        else:
            logging.warning("No image selected.")

    def get_reference_encodings(self):
        """Получаем кодировки референсных лиц."""
        return [encoding for name in self.selected_faces for encoding in self.face_encodings.get(name, [])]

    def resize_image(self, img, max_size=5000):
        """Изменение размера изображения, если оно больше max_size."""
        height, width = img.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_size = (int(width * scale), int(height * scale))
            img = cv2.resize(img, new_size)
            logging.info(f"Image resized to {new_size}.")
        return img

    def process_image(self, image_path, reference_encodings):
        """Обработка загруженного изображения и распознавание лиц."""
        img = face_recognition.load_image_file(image_path)
        img = self.resize_image(img)  # Изменяем размер изображения
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Преобразуем в RGB

        face_boxes = self.detector.detect_faces(img_rgb, self.min_face_size, self.thresholds)
        face_locations = self.extract_face_locations(face_boxes)

        img_encodings = face_recognition.face_encodings(img_rgb, face_locations)
        if not img_encodings:
            logging.warning("No faces found in the image.")
            return

        reference_location = self.get_reference_location(img_encodings, face_locations, reference_encodings)
        if reference_location is None:
            logging.warning("Reference face not found in the image.")
            return

        self.log_face_locations(face_locations)

        # Этапы размытия лиц
        self.blur_faces_with_recognition(img_rgb, face_locations, reference_encodings, reference_location)
        self.blur_faces_with_mtcnn(img_rgb, face_boxes, reference_location)

        self.save_image(img_rgb)

    def extract_face_locations(self, face_boxes):
        """Извлечение координат лиц из результатов MTCNN."""
        return [(box['box'][1], box['box'][0] + box['box'][2], box['box'][3] + box['box'][1], box['box'][0]) for box in face_boxes]

    def log_face_locations(self, face_locations):
        """Логирование координат найденных лиц."""
        for i, location in enumerate(face_locations):
            logging.info(f"Face {i + 1}: Location {location}")

    def get_reference_location(self, img_encodings, face_locations, reference_encodings):
        """Получение координат референсного лица."""
        for i, (face_encoding, location) in enumerate(zip(img_encodings, face_locations)):
            distances = np.linalg.norm(reference_encodings - face_encoding, axis=1)
            if np.min(distances) < 0.4:  # Пороговое значение
                logging.info(f"Reference face found at location {location}.")
                return location
        return None

    def blur_faces_with_recognition(self, image, face_locations, reference_encodings, reference_location):
        """Размываем лица на изображении, используя face_recognition."""
        logging.info("Start blur_faces_with_recognition")
        img_encodings = face_recognition.face_encodings(image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, img_encodings):
            distances = np.linalg.norm(reference_encodings - face_encoding, axis=1)
            min_distance = np.min(distances)

            if min_distance > 0.4 and not self.is_reference_face((top, right, bottom, left), reference_location):
                roi = image[top:bottom, left:right]
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                image[top:bottom, left:right] = blurred_roi
                logging.info(f"Blurred face at location {top, left, bottom, right}.")

    def blur_faces_with_mtcnn(self, image, face_boxes, reference_location):
        """Размываем лица на изображении, используя MTCNN."""
        logging.info("Start blur_faces_with_mtcnn")
        face_locations = self.extract_face_locations(face_boxes)
        img_encodings = face_recognition.face_encodings(image, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, img_encodings):
            if not self.is_reference_face((top, right, bottom, left), reference_location):
                roi = image[top:bottom, left:right]
                blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                image[top:bottom, left:right] = blurred_roi
                logging.info(f"Blurred face at location {top, left, bottom, right}.")

    def is_reference_face(self, face_location, reference_location):
        """Проверяем, является ли текущее лицо референсным."""
        return face_location == reference_location

    def save_image(self, image):
        """Сохраняем изображение в формате RGB."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        blurred_image_path = "output_blurred.jpg"
        pil_image.save(blurred_image_path, "JPEG")
        logging.info(f"Image saved to {blurred_image_path}.")

if __name__ == "__main__":
    app = PhotoProcessingApp()
    root = tk.Tk()
    root.title("Photo Processing Application")
    upload_button = tk.Button(root, text="Upload Image", command=app.upload_image)
    upload_button.pack(pady=20)
    root.mainloop()

