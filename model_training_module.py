import face_recognition
import os
import json
import numpy as np
import tkinter as tk


class ModelTrainingApp:
    def __init__(self, master, image_folder='faces'):
        self.image_folder = image_folder
        self.master = master
        self.master.title("Model Training App")

        # Создаем текстовое поле для вывода сообщений
        self.text_output = tk.Text(master, wrap=tk.WORD, height=20, width=50)
        self.text_output.pack(pady=10)

        # Запускаем процесс обработки лиц
        self.process_all_faces()

    # Функция для обработки всех папок и создания/обновления файлов с кодировками
    def process_all_faces(self):
        for folder_name in os.listdir(self.image_folder):
            folder_path = os.path.join(self.image_folder, folder_name)

            # Проверяем, является ли это папкой
            if os.path.isdir(folder_path):
                self.text_output.insert(tk.END, f"Processing folder: {folder_name}\n")
                self.master.update()  # Обновляем окно для отображения изменений

                # Правильный путь для JSON файла: внутри папки с именем лица
                json_file_path = os.path.join(folder_path, f"{folder_name}.json")
                encodings_dict = {'files': []}
                save_required = False  # Флаг, чтобы отслеживать, нужно ли сохранять файл

                # Если JSON файла нет, создаём его с новыми кодировками
                if not os.path.isfile(json_file_path):
                    self.text_output.insert(tk.END, f"No JSON file found for {folder_name}, creating new one.\n")
                else:
                    self.text_output.insert(tk.END, f"Loading known face encodings from {json_file_path}...\n")
                    with open(json_file_path, 'r') as json_file:
                        encodings_dict = json.load(json_file)

                # Считываем все файлы изображений в папке
                image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

                # Обрабатываем изображения и добавляем кодировки
                for filename in image_files:
                    if not any(file_info['file_name'] == filename for file_info in encodings_dict['files']):
                        image_path = os.path.join(folder_path, filename)
                        self.text_output.insert(tk.END, f"Loading image: {image_path}\n")
                        self.master.update()  # Обновляем окно

                        try:
                            image = face_recognition.load_image_file(image_path)
                            encodings = face_recognition.face_encodings(image)
                            for encoding in encodings:
                                encodings_dict['files'].append({
                                    "file_name": filename,
                                    "encodings": encoding.tolist()  # Сохраняем кодировки в формате списка
                                })
                            self.text_output.insert(tk.END, f"Added {len(encodings)} encoding(s) for {filename}\n")
                            save_required = True  # Устанавливаем флаг для сохранения
                        except Exception as e:
                            self.text_output.insert(tk.END, f"Error loading {filename}: {e}\n")

                # Сохранение кодировок в JSON файл только при необходимости
                if save_required:
                    with open(json_file_path, 'w') as json_file:
                        json.dump(encodings_dict, json_file, indent=4)
                        self.text_output.insert(tk.END, f"Saved encodings to {json_file_path}\n")
                else:
                    self.text_output.insert(tk.END, f"No new encodings added for {folder_name}, skipping save.\n")

                # Прокручиваем текстовое поле вниз
                self.text_output.see(tk.END)

        # Закрываем окно через 5 секунд
        self.master.after(5000, self.master.destroy)


def run(master):
    app = ModelTrainingApp(master)
    master.mainloop()  # Запускаем главный цикл для нового окна
