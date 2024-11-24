import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, r2_score, mean_squared_error, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Load recognizer and cascade for face detection
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# Define the confidence threshold
THRESHOLD = 20  # Adjust as needed

# Initialize ID counter and names (replace with actual names)
names = ['Unknown', 'Person1', 'Person2', 'Person3']  # Add more names as required

# Start video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Define the start time
start_time = time.time()
DURATION = 20  # Duration in seconds

# Attendance logging function
def log_attendance(person_name):
    if not os.path.exists("attendance.xlsx"):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_excel("attendance.xlsx", index=False)

    df = pd.read_excel("attendance.xlsx")
    today = datetime.now().date()
    if not ((df["Name"] == person_name) & (df["Date"] == str(today))).any():
        new_entry = {"Name": person_name, "Date": str(today), "Time": datetime.now().strftime("%H:%M:%S")}
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
        df.to_excel("attendance.xlsx", index=False)

# Lists for storing actual labels, predictions, and confidence scores
y_true = []
y_pred = []
confidence_scores = []

# Recognition and attendance marking loop
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(0.1 * cam.get(3)), int(0.1 * cam.get(4))))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Predict ID and confidence
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
        confidence_percent = round(100 - confidence)

        if confidence_percent >= THRESHOLD:
            name = names[id] if id < len(names) else "Unknown"
            log_attendance(name)
            confidence_text = f" {confidence_percent}% (Present)"
            y_true.append(1)  # Assuming 1 indicates presence
            y_pred.append(1)
        else:
            name = "Unknown"
            confidence_text = f" {confidence_percent}% (Below threshold)"
            y_true.append(1)  # True face is present
            y_pred.append(0)  # Model fails to recognize correctly

        # Collect confidence scores for each detection
        confidence_scores.append(confidence_percent / 100.0)  # Normalize score to [0,1]

        cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x + 5, y + h - 5), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 0), 1)

    cv2.imshow('Camera', img)

    # Check if the runtime exceeds the desired duration
    if time.time() - start_time > DURATION:
        print("Recognition completed.")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()

# Calculate performance metrics
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')
conf_matrix = confusion_matrix(y_true, y_pred)
r2 = r2_score(y_true, confidence_scores)
mse = mean_squared_error(y_true, confidence_scores)

# Print metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"R² Score: {r2}")
print(f"MSE: {mse}")

# Plot Precision-Recall Curve
precisions, recalls, thresholds = precision_recall_curve(y_true, confidence_scores)
plt.plot(recalls, precisions, marker='.')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# Plot Confusion Matrix Heatmap
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix Heatmap")
plt.show()
