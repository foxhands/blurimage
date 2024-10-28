import os
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk

import photo_processing_module


def launch_face_selection():
    """Function to launch the Face Selection App."""
    try:
        import face_selection_module  # Импортируем модуль для добавления лиц
        new_root = tk.Toplevel()  # Создаем новое окно для Face Selection
        face_selection_module.run(new_root)  # Передаем новое окно
    except ImportError:
        messagebox.showerror("Error", "Face selection module not found.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def launch_photo_processing():
    """Function to launch the Photo Processing App."""
    try:
        photo_processing_module.run_photo_processing()  # Запускаем функцию обработки фото
    except ImportError:
        messagebox.showerror("Error", "Photo processing module not found.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


def launch_train_model():
    """Function to initiate the training of the face recognition model."""
    try:
        import model_training_module  # Импортируем модуль для обучения модели
        new_root = tk.Toplevel()  # Создаем новое окно для Face Selection
        model_training_module.run(new_root)  # Запускаем функцию обучения
    except ImportError:
        messagebox.showerror("Error", "Model training module not found.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


class MainApp:
    def __init__(self, master):
        self.master = master
        master.title("Main Blur Application")

        # Установка минимального размера окна
        master.minsize(500, 500)

        # Создание меню
        self.menu = tk.Menu(master)
        master.config(menu=self.menu)

        # Создание подменю
        self.file_menu = tk.Menu(self.menu, tearoff=0)
        # Добавление команд в подменю
        self.menu.add_command(label="Add Faces", command=launch_face_selection)
        self.menu.add_command(label="Train Model", command=launch_train_model)
        self.menu.add_command(label="Quit", command=master.quit)

        # Кнопка для загрузки фото в центре окна
        self.center_frame = ttk.Frame(master)
        self.center_frame.pack(expand=True)

        self.launch_photo_button = tk.Button(self.center_frame, text="Load Photo", command=launch_photo_processing,
                                             width=20)
        self.launch_photo_button.pack(pady=20)


if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
