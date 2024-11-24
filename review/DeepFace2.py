import cv2
import pandas as pd
from datetime import datetime
import time
import os
from deepface import DeepFace
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, r2_score, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

# Define constants
THRESHOLD = 0.5  # Similarity threshold for FaceNet
DURATION = 30  # Duration for the recognition loop in seconds

# Start the video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Width
cam.set(4, 480)  # Height

# Load the Haar cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Initialize lists for metrics calculation
y_true = []            # Actual labels (1 for present, 0 for absent)
y_pred = []            # Predicted labels (1 for recognized, 0 for unrecognized)
confidence_scores = [] # Recognition confidence scores for each prediction

# Recognition and attendance marking loop
start_time = time.time()
while True:
    ret, img = cam.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, 
        minSize=(int(0.1 * cam.get(3)), int(0.1 * cam.get(4)))
    )

    for (x, y, w, h) in faces:
        face_img = img[y:y+h, x:x+w]
        
        try:
            # Recognize face using DeepFace with FaceNet
            results = DeepFace.find(
                img_path=face_img, db_path="./dataset", 
                model_name="Facenet", enforce_detection=False
            )
            
            # Check if results list contains DataFrames and is not empty
            if results and isinstance(results[0], pd.DataFrame) and not results[0].empty:
                result = results[0]
                identity = result.iloc[0]['identity']
                distance = result.iloc[0]['distance']
                
                if distance <= THRESHOLD:
                    label = os.path.basename(identity).split(".")[0]
                    y_true.append(1)            # Ground truth: present
                    y_pred.append(1)            # Predicted: recognized correctly
                    confidence_scores.append(1 - distance) # Higher confidence for lower distance
                else:
                    label = "Unknown"
                    y_true.append(1)            # Ground truth: present
                    y_pred.append(0)            # Predicted: not recognized (false negative)
                    confidence_scores.append(1 - distance)
            else:
                label = "Unknown"
                y_true.append(1)                # Ground truth: present
                y_pred.append(0)                # Predicted: not recognized (false negative)
                confidence_scores.append(0)     # No confidence for unrecognized face

            # Draw bounding box and label
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        except Exception as e:
            y_true.append(1)                    # Ground truth: present
            y_pred.append(0)                    # Predicted: not recognized (false negative)
            confidence_scores.append(0)         # No confidence for errors

    # Display the video feed with bounding boxes and labels
    cv2.imshow("Face Recognition", img)

    # Check if the runtime exceeds the desired duration
    if time.time() - start_time > DURATION:
        break

    # Exit on pressing 'q'
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
from sklearn.metrics import precision_recall_curve

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


# import cv2
# import pandas as pd
# from datetime import datetime
# import time
# import os
# from deepface import DeepFace
# from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, r2_score, mean_squared_error
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Define constants
# THRESHOLD = 0.5  # Similarity threshold for FaceNet
# DURATION = 10  # Duration for the recognition loop in seconds

# # Start the video capture
# cam = cv2.VideoCapture(0)
# cam.set(3, 640)  # Width
# cam.set(4, 480)  # Height

# # Load the Haar cascade
# faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # Initialize lists for metrics calculation
# y_true = []            # Actual labels (1 for present, 0 for absent)
# y_pred = []            # Predicted labels (1 for recognized, 0 for unrecognized)
# confidence_scores = [] # Recognition confidence scores for each prediction

# # Recognition and attendance marking loop
# start_time = time.time()
# while True:
#     ret, img = cam.read()
#     if not ret:
#         print("Failed to grab frame.")
#         break

#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(0.1 * cam.get(3)), int(0.1 * cam.get(4))))

#     for (x, y, w, h) in faces:
#         face_img = img[y:y+h, x:x+w]
        
#         try:
#             # Recognize face using DeepFace with FaceNet
#             results = DeepFace.find(img_path=face_img, db_path="./dataset", model_name="Facenet", enforce_detection=False)
            
#             # Check if results list contains DataFrames and is not empty
#             if results and isinstance(results[0], pd.DataFrame) and not results[0].empty:
#                 result = results[0]
#                 distance = result.iloc[0]['distance']
                
#                 if distance <= THRESHOLD:
#                     y_true.append(1)            # Ground truth: present
#                     y_pred.append(1)            # Predicted: recognized correctly
#                     confidence_scores.append(1 - distance) # Higher confidence for lower distance
#                 else:
#                     y_true.append(1)            # Ground truth: present
#                     y_pred.append(0)            # Predicted: not recognized (false negative)
#                     confidence_scores.append(1 - distance)
#             else:
#                 y_true.append(1)                # Ground truth: present
#                 y_pred.append(0)                # Predicted: not recognized (false negative)
#                 confidence_scores.append(0)     # No confidence for unrecognized face
        
#         except Exception as e:
#             y_true.append(1)                    # Ground truth: present
#             y_pred.append(0)                    # Predicted: not recognized (false negative)
#             confidence_scores.append(0)         # No confidence for errors

#     # Check if the runtime exceeds the desired duration
#     if time.time() - start_time > DURATION:
#         break

# # Cleanup
# cam.release()
# cv2.destroyAllWindows()

# # Calculate performance metrics
# precision = precision_score(y_true, y_pred, average='binary')
# recall = recall_score(y_true, y_pred, average='binary')
# f1 = f1_score(y_true, y_pred, average='binary')
# conf_matrix = confusion_matrix(y_true, y_pred)
# r2 = r2_score(y_true, confidence_scores)
# mse = mean_squared_error(y_true, confidence_scores)

# # Print metrics
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1 Score: {f1}")
# print(f"Confusion Matrix:\n{conf_matrix}")
# print(f"R² Score: {r2}")
# print(f"MSE: {mse}")

# # Plot Precision-Recall Curve
# from sklearn.metrics import precision_recall_curve

# precisions, recalls, thresholds = precision_recall_curve(y_true, confidence_scores)
# plt.plot(recalls, precisions, marker='.')
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.show()

# # Plot Confusion Matrix Heatmap
# sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
# plt.xlabel("Predicted Labels")
# plt.ylabel("True Labels")
# plt.title("Confusion Matrix Heatmap")
# plt.show()
