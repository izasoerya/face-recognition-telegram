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
        offsetW = (5 / 100) * w
        x = int(x - offsetW)
        w = int(w + offsetW * 2)
        offsetH = (5 / 100) * h
        y = int(y - offsetH * 3)
        h = int(h + offsetH * 3.5)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Extract the ROI for face recognition
        face_roi_gray = gray[y:y+h, x:x+w]
        face_roi_color = img[y:y+h, x:x+w]

        # Perform face recognition on the ROI
        id, confidence = recognizer.predict(face_roi_gray)

        # Check confidence level for recognition
        confidence_text = round(100 - confidence)
        
        if confidence_text>=50 :
            employee = models.employee(names[id])

            # Check if the current minute is different from the last recorded minute
            if (minute+1) < ft.fetch_time_minute():
                employee.reset_attendance()
                minute = ft.fetch_time_minute()

            id = names[id]
            confidence_text = "{0}%".format(confidence_text)


            # Check attendance and send Telegram message if needed
            # if not employee.check_attendance():
            #     employee.send_telegram_msg(id)

        elif confidence_text<50:
            id = "unknown"
            employee = models.employee(id)
            confidence_text = "{0}%".format(confidence_text)
            cv2.putText(img, "Fixed your angle camera", (x+5, y+h+20), font, 1, (0, 0, 255), 2)
            

        cv2.putText(img, str(id), (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_text), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    # Press 'ESC' for exiting the video
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
