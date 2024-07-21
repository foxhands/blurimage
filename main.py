import cv2
import numpy as np
from mtcnn import MTCNN
import face_recognition
from concurrent.futures import ThreadPoolExecutor


def load_reference_face(reference_image_path):
    """Load the reference face and extract its encoding."""
    reference_image = face_recognition.load_image_file(reference_image_path)
    reference_face_encodings = face_recognition.face_encodings(reference_image)
    if len(reference_face_encodings) == 0:
        print("No faces found in the reference image.")
        return None
    print("Reference face encoding loaded.")
    return reference_face_encodings[0]


def resize_image(image, scale_percent=50):
    """Resize the image by a given percentage."""
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def preprocess_image(image, scale_percent=50):
    """Resize image for processing and then return it to the original size."""
    image_resized = resize_image(image, scale_percent)
    return cv2.resize(image_resized, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


def detect_faces_and_blur(image, reference_face_encoding, output_path, min_face_size=30):
    """Detect faces in the image, blur faces that are not the reference face, and save the result."""
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize MTCNN detector
    detector = MTCNN()
    faces = detector.detect_faces(image_rgb)

    print(f"Detected {len(faces)} face(s) in the image.")

    faces_to_blur = []
    for face in faces:
        (x, y, width, height) = face['box']
        if width < min_face_size or height < min_face_size:
            continue
        face_location = (y, x + width, y + height, x)
        face_encoding = face_recognition.face_encodings(image_rgb, [face_location])

        if not face_encoding:
            continue

        is_reference_face = face_recognition.compare_faces([reference_face_encoding], face_encoding[0])[0]
        if not is_reference_face:
            faces_to_blur.append(face_location)
            print(f"Face detected at (top={y}, right={x + width}, bottom={y + height}, left={x}) - Blurring")
        else:
            print(f"Reference face detected at (top={y}, right={x + width}, bottom={y + height}, left={x})")

    for (top, right, bottom, left) in faces_to_blur:
        roi = image[top:bottom, left:right]
        blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
        image[top:bottom, left:right] = blurred_roi
        print(f"Blurred face at (top={top}, right={right}, bottom={bottom}, left={left})")

    cv2.imwrite(output_path, image)
    print(f"Output saved to {output_path}")


def process_image(image_path, output_path, reference_face_encoding):
    """Load and process a single image."""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    # Resize image for faster processing
    image = preprocess_image(image)

    print(f"Processing image: {image_path}")
    detect_faces_and_blur(image, reference_face_encoding, output_path)


def process_images_parallel(reference_image_path, image_paths, output_paths):
    """Process multiple images in parallel, excluding the reference face from blurring."""
    reference_face_encoding = load_reference_face(reference_image_path)
    if reference_face_encoding is None:
        return

    with ThreadPoolExecutor() as executor:
        for image_path, output_path in zip(image_paths, output_paths):
            executor.submit(process_image, image_path, output_path, reference_face_encoding)


# Example usage
reference_image_path = 'input.jpg'  # Path to the image of the model to be preserved
image_paths = ['1.jpg', '2.jpg', '3.jpg']  # List of image paths to process
output_paths = ['output1.jpg', 'output2.jpg', 'output3.jpg']  # Corresponding output paths

process_images_parallel(reference_image_path, image_paths, output_paths)
