import cv2
import numpy as np
import os 
import time  # Import time module for time tracking
import fetch_time as ft
import models

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['faiz', 'iza', 'akmal', 'gilang', 'reza']

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

isabsence = 0
minute = ft.fetch_time_minute()
day = ft.fetch_date_day()

# Initialize variables for data collection
recognized_ids = []
recognizing_time = 5  # Time duration for recognizing
start_time = time.time()

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip vertically

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        # Expand the bounding box to include more of the face
        offsetW = int(0.05 * w)
        offsetH = int(0.05 * h)
        x -= offsetW
        y -= offsetH
        w += 2 * offsetW
        h += 2 * offsetH

        # Ensure the expanded bounding box is within the image bounds
        x = max(0, x)
        y = max(0, y)
        w = min(img.shape[1] - x, w)
        h = min(img.shape[0] - y, h)

        # Extract the ROI for face recognition
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = img[y:y+h, x:x+w]

        # Perform face recognition on the ROI
        id, confidence = recognizer.predict(face_roi_gray)

        # Check confidence level for recognition
        confidence_text = round(100 - confidence)
        
        if confidence_text >= 50:
            recognized_ids.append(id)

    # Check if 5 seconds have passed
    elapsed_time = time.time() - start_time
    if elapsed_time >= recognizing_time:
        # Calculate the average recognized ID
        if recognized_ids:
            average_id = int(round(np.mean(recognized_ids)))
            recognized_name = names[average_id] if average_id > 0 else "unknown"
            print("Recognized person:", recognized_name)
            # Reset variables for next recognition cycle
            recognized_ids = []
            start_time = time.time()

    cv2.imshow('camera', img)

    # Press 'ESC' for exiting the video
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
