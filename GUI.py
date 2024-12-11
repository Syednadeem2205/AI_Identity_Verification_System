import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import json
import sqlite3
import os
from ultralytics import YOLO
import face_recognition
import numpy as np
from datetime import datetime

# Database Setup
def setup_database():
    conn = sqlite3.connect("identity_verification.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        age TEXT NOT NULL,
                        branch TEXT NOT NULL,
                        image_folder TEXT NOT NULL,
                        first_detected TEXT,
                        last_detected TEXT)''')
    conn.commit()
    conn.close()

def get_db_connection():
    return sqlite3.connect("identity_verification.db")

def save_metadata(name, age, branch, folder_name):
    metadata = {"name": name, "age": age, "branch": branch, "folder": folder_name}
    with open("metadata.json", "a") as file:
        json.dump(metadata, file)
        file.write("\n")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, age, branch, image_folder, first_detected, last_detected) VALUES (?, ?, ?, ?, ?, ?)",
                   (name, age, branch, folder_name, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def update_detection_time(name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET last_detected = ? WHERE name = ?", (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name))
    conn.commit()
    conn.close()

# Capture Image Function
def capture_image(name_entry, age_entry, branch_entry):
    name = name_entry.get().strip()
    age = age_entry.get().strip()
    branch = branch_entry.get().strip()

    if not name or not age or not branch:
        messagebox.showerror("Invalid Input", "All fields are required!")
        return

    dataset_dir = "datasets"
    user_folder = os.path.join(dataset_dir, name.replace(" ", "_"))
    if not os.path.exists(user_folder):
        os.makedirs(user_folder)

    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        image_index = len([f for f in os.listdir(user_folder) if f.endswith('.jpg')]) + 1
        image_filename = f"{name.replace(' ', '_')}_{image_index}.jpg"
        image_path = os.path.join(user_folder, image_filename)
        cv2.imwrite(image_path, frame)

        save_metadata(name, age, branch, user_folder)

        cv2.imshow("Captured Image", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        messagebox.showinfo("Success", f"Image and details saved for {name}.")
    else:
        messagebox.showerror("Capture Failed", "Failed to capture image.")
    cap.release()

# Load YOLOv8 Model
def load_yolov8_model():
    model = YOLO('yolov8n.pt')  # Ensure you have the correct path to the model
    return model

model = load_yolov8_model()  # Load YOLO model globally

# Load Stored Faces from Dataset
def load_stored_faces():
    known_face_encodings = []
    known_face_names = []

    # Loop through the folders where images are stored
    for folder in os.listdir("datasets"):
        user_folder = os.path.join("datasets", folder)
        if os.path.isdir(user_folder):
            for image_file in os.listdir(user_folder):
                if image_file.endswith(".jpg"):
                    image_path = os.path.join(user_folder, image_file)
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)

                    if encoding:  # If face encoding is found
                        known_face_encodings.append(encoding[0])
                        known_face_names.append(folder.replace("_", " "))  # The folder name is the person's name

    return known_face_encodings, known_face_names

# Start Live Detection
def start_live_detection():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Failed to access camera.")
        return

    # Load stored faces from the database
    known_face_encodings, known_face_names = load_stored_faces()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Use YOLO to detect faces
        results = model(frame)
        for result in results[0].boxes:
            x_center, y_center, width, height = result.xywh[0]
            confidence = result.confidence[0] if hasattr(result, 'confidence') else 1.0

            if confidence < 0.5:
                continue

            left = int((x_center - width / 2))
            top = int((y_center - height / 2))
            right = int((x_center + width / 2))
            bottom = int((y_center + height / 2))

            face_frame = frame[top:bottom, left:right]

            try:
                # Convert the face image to RGB
                rgb_face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

                # Get face encodings from the detected face
                face_encodings = face_recognition.face_encodings(rgb_face_frame)

                if face_encodings:
                    # Compare the detected face encoding with the known faces
                    face_encoding = face_encodings[0]
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                    name = "Unknown"
                    face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distance)

                    # If there is a match, use the best match
                    if True in matches:
                        name = known_face_names[best_match_index]
                        update_detection_time(name)

                    # Get the first and last detected times
                    conn = get_db_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT first_detected, last_detected FROM users WHERE name = ?", (name,))
                    result = cursor.fetchone()

                    first_detected = result[0]
                    last_detected = result[1]

                    # Display the name, first and last detected time, and confidence score
                    cv2.putText(frame, f"Name: {name}", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"First Detected: {first_detected}", (left, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Last Detected: {last_detected}", (left, top + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Confidence: {confidence * 100:.2f}%", (left, top + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                else:
                    cv2.putText(frame, "No face found", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            except Exception as e:
                print(f"Error in face recognition: {str(e)}")
                cv2.putText(frame, "Error in recognition", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.imshow("Live Face Detection and Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Application Window
def main_window():
    root = tk.Tk()
    root.attributes('-fullscreen', True)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    space_image = Image.open("background-image.jpg")
    space_image = space_image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
    space_bg = ImageTk.PhotoImage(space_image)

    bg_label = tk.Label(root, image=space_bg)
    bg_label.place(relwidth=1, relheight=1)

    frame = tk.Frame(root, bg="white", bd=10, relief=tk.RIDGE)
    frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    tk.Label(frame, text="Name:", font=('Orbitron', 16), bg="white").grid(row=0, column=0, padx=10, pady=10)
    name_entry = tk.Entry(frame, font=('Orbitron', 16))
    name_entry.grid(row=0, column=1, padx=10, pady=10)

    tk.Label(frame, text="Age:", font=('Orbitron', 16), bg="white").grid(row=1, column=0, padx=10, pady=10)
    age_entry = tk.Entry(frame, font=('Orbitron', 16))
    age_entry.grid(row=1, column=1, padx=10, pady=10)

    tk.Label(frame, text="Branch:", font=('Orbitron', 16), bg="white").grid(row=2, column=0, padx=10, pady=10)
    branch_entry = tk.Entry(frame, font=('Orbitron', 16))
    branch_entry.grid(row=2, column=1, padx=10, pady=10)

    button_style = {
        "font": ('Orbitron', 14, 'bold'),
        "bg": "#4CAF50",
        "fg": "white",
        "activebackground": "#45a049",
        "activeforeground": "white",
        "relief": tk.RAISED,
        "bd": 3
    }

    tk.Button(frame, text="Capture Image", command=lambda: capture_image(name_entry, age_entry, branch_entry), **button_style).grid(row=3, column=0, pady=10)
    tk.Button(frame, text="Start Live Detection", command=start_live_detection, **button_style).grid(row=3, column=1, pady=10)

    root.mainloop()

# Login Window
def login_window():
    login = tk.Tk()
    login.attributes('-fullscreen', True)

    screen_width = login.winfo_screenwidth()
    screen_height = login.winfo_screenheight()

    space_image = Image.open("background-image.jpg")
    space_image = space_image.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
    space_bg = ImageTk.PhotoImage(space_image)

    bg_label = tk.Label(login, image=space_bg)
    bg_label.place(relwidth=1, relheight=1)

    frame = tk.Frame(login, bg="white", bd=10, relief=tk.RIDGE)
    frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    tk.Label(frame, text="Username:", font=('Orbitron', 16), bg="white").grid(row=0, column=0, padx=10, pady=10)
    username_entry = tk.Entry(frame, font=('Orbitron', 16))
    username_entry.grid(row=0, column=1, padx=10, pady=10)

    tk.Label(frame, text="Password:", font=('Orbitron', 16), bg="white").grid(row=1, column=0, padx=10, pady=10)
    password_entry = tk.Entry(frame, show="*", font=('Orbitron', 16))
    password_entry.grid(row=1, column=1, padx=10, pady=10)

    def authenticate():
        username = username_entry.get()
        password = password_entry.get()
        if username == "admin" and password == "password":
            login.destroy()
            main_window()
        else:
            messagebox.showerror("Login Failed", "Invalid credentials")

    button_style = {
        "font": ('Orbitron', 14, 'bold'),
        "bg": "#2196F3",
        "fg": "white",
        "activebackground": "#0b7dda",
        "activeforeground": "white",
        "relief": tk.RAISED,
        "bd": 3
    }

    tk.Button(frame, text="Login", command=authenticate, **button_style).grid(row=2, column=0, columnspan=2, pady=10)

    login.mainloop()

# Set up the database and start the login window
setup_database()
login_window()
