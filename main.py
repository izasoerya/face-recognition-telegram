import cv2
import fetch_time as ft
import models
import telegram

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

# Read threshold values from the text file
with open('threshold.txt', 'r') as file:
    threshold_lines = file.readlines()
    threshold = int(threshold_lines[1].strip())
    reset_time = int(threshold_lines[4].strip())
    avg_face_count = int(threshold_lines[7].strip())

# Initialize variables
recognition_count = {name: 0 for name in models.user_attendance_list}
unknown_faces = []
minute = ft.fetch_time_minute()
isrecog = False
isrecogUnknown = False
unknown = 0

recogname = ""

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=2.8,
        minNeighbors=5, 
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        # Draw a rectangle around the faces
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Use the entire face region for prediction
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Check confidence level
        confidence_text = round(100 - confidence)
        if confidence_text >= threshold:
            name = models.user_attendance_list[id]
            employee = models.employee(name)

            # Reset attendance if the reset time has passed
            if (minute + reset_time) < ft.fetch_time_minute():
                employee.reset_attendance()
                minute = ft.fetch_time_minute()

            # Show the percentage of confidence
            confidence_text = "{0}%".format(confidence_text)
            
            # Average the user recognition
            recognition_count[name] += 1
            if recognition_count[name] >= avg_face_count:
                for reset in recognition_count:
                    # Reset the count for all names to zero
                    recognition_count[reset] = 0
                if not employee.check_attendance():
                    employee.send_telegram_msg(name)
                    isrecog = True
                    recogname = name
                recognition_count[name] = 0

            if recognition_count[name] == avg_face_count / 2:
                isrecog = False
                recogname = ""

            # Debug output
            print(f"Recognized {name} with confidence {confidence_text}")
        else:
            name = "unknown"
            confidence_text = "{0}%".format(confidence_text)
            cv2.putText(img, "Fix your camera angle", (x + 5, y + h + 20), font, 1, (0, 0, 255), 2)
            unknown += 1
            if unknown == 50:
                isrecogUnknown = False
            if unknown >= 80:
                isrecogUnknown = True
                # telegram.send_telegram_message('Unknown')
                unknown = 0

        if isrecog:
            cv2.putText(img, recogname + " is present", (x + 5, y + h + 50), font, 1, (0, 0, 255), 2)
        # if isrecogUnknown:
        #     cv2.putText(img, "Unknown is present", (x + 5, y + h + 50), font, 1, (0, 0, 255), 2)
        cv2.putText(img, name, (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_text), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    # Process unknown faces
    # if len(unknown_faces) >= avg_face_count:
    #     for reset in recognition_count:
    #         recognition_count[reset] = 0  # Reset the count for all names to zero
    #     for face_coords in unknown_faces:
    #         x, y, w, h = face_coords
    #         isrecogUnknown = True
    #     if isrecogUnknown:
    #         cv2.putText(img, "Unknown is present", (x + 5, y + h + 50), font, 1, (0, 0, 255), 2)
    #     # Send a Telegram notification for unknown faces
    #     timestamp = ft.fetch_time_minute()
    #     unknown_faces = []

    print(recognition_count)
    cv2.imshow('camera', img)

    # Press 'ESC' for exiting the video
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Do a bit of cleanup
print("\n[INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
