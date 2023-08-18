import cv2
from cv2 import VideoCapture
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import face_recognition

# Load and preprocess your dataset
def load_and_preprocess_data(data_path, input_shape):
    images = []
    labels = []

    for label, person_folder in enumerate(os.listdir(data_path)):
        person_path = os.path.join(data_path, person_folder)
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, input_shape[:2])
            images.append(image)
            labels.append(label)  # Use label index as the label

    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Build the CNN model
def build_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Set your dataset path
dataset_path = r"C:\Users\grove\Downloads\images for training"  # Replace with the actual path to your dataset
tamanna_folder = r"C:\Users\grove\Downloads\images for training\Tamanna" # Replace with actual path
krrish_folder = r"C:\Users\grove\Downloads\images for training\Krrish" # Replace with actual path

# Load and preprocess data
input_shape = (64, 64, 3)   # Specify input image shape
images, labels = load_and_preprocess_data(dataset_path, input_shape)
num_classes = len(np.unique(labels))

# Calculate known embeddings and labels for each person
known_embeddings = []
known_labels = []

# Calculate known embeddings for "Tamanna"
tamanna_image_paths = [os.path.join(tamanna_folder, filename) for filename in os.listdir(tamanna_folder)]
for image_path in tamanna_image_paths:
    image = face_recognition.load_image_file(image_path)
    embedding = face_recognition.face_encodings(image)[0]  # Assuming one face per image
    known_embeddings.append(embedding)
    known_labels.append("Tamanna")

# Calculate known embeddings for "Krrish"
krrish_image_paths = [os.path.join(krrish_folder, filename) for filename in os.listdir(krrish_folder)]
for image_path in krrish_image_paths:
    image = face_recognition.load_image_file(image_path)
    embedding = face_recognition.face_encodings(image)[0]  # Assuming one face per image
    known_embeddings.append(embedding)
    known_labels.append("Krrish")

known_embeddings = np.array(known_embeddings)
known_labels = np.array(known_labels)

# ... (Previous code) ...

while True:
    ret, frame = VideoCapture.read()

    if not ret:
        break
    
    # Perform face detection on the frame using face_recognition
    face_locations = face_recognition.face_locations(frame)
    
    # Extract face embeddings using the trained model
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    face_names = []
    
    for face_encoding in face_encodings:
        # Compare the detected face embedding with known embeddings
        distances = np.linalg.norm(known_embeddings - face_encoding, axis=1)
        min_distance_index = np.argmin(distances)
        recognized_name = known_labels[min_distance_index]  # Use known_labels instead of known_names
        face_names.append(recognized_name)
    
    # Draw rectangles and labels on the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    
    # Display the frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()