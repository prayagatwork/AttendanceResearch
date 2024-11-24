import cv2
from deepface import DeepFace
import os
import pandas as pd

# Load the Haar cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start the video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Width
cam.set(4, 480)  # Height

THRESHOLD = 0.5  # Similarity threshold

print("\n [INFO] Starting face recognition... Press 'q' to exit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        
        try:
            # Recognize face using DeepFace
            # results = DeepFace.find(img_path=face_img, db_path="./dataset", enforce_detection=False)
            results = DeepFace.find(
                    img_path=face_img, 
                    db_path="./dataset", 
                    model_name="ArcFace",  # Use a robust model
                    enforce_detection=False, 
                    align=True  # Align detected faces
                    )
            if results and isinstance(results[0], pd.DataFrame) and not results[0].empty:
                match = results[0]
                identity = match.iloc[0]['identity']
                distance = match.iloc[0]['distance']

                if distance <= THRESHOLD:
                    # Extract face ID from the file name
                    file_name = os.path.basename(identity)
                    face_id = file_name.split('.')[1]  # Assuming the format User.<face_id>.<count>.jpg
                else:
                    face_id = "Unknown"
            else:
                face_id = "Unknown"
        
        except Exception as e:
            face_id = "Unknown"

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, face_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the video feed with bounding boxes and IDs
    cv2.imshow("Face Recognition", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()
