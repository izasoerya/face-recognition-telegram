import cv2
import fetch_time as ft
import models

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

#iniciate id counter
id = 0

# Initialize and start realtime video capture
cam = cv2.VideoCapture(2)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video he  ight

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

isabscence = 0
minute = ft.fetch_time_minute()
day = ft.fetch_date_day()

recognition_count = {name: 0 for name in models.user_attendance_list}
isrecog = False
recogname = ""

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
        # Draw a rectangle around the faces
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Use the entire face region for prediction
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        # Check if confidence is less than 100 ==> "0" is a perfect match
        confidence_text = round(100 - confidence)
        if confidence_text>=72 :
            name = models.user_attendance_list[id]
            employee = models.employee(name)

            # Check if the current minute is different from the last recorded minute
            if (minute+1) < ft.fetch_time_minute():
                employee.reset_attendance()
                minute = ft.fetch_time_minute()

            # id = names[id]
            confidence_text = "{0}%".format(confidence_text)
            
            recognition_count[name] += 1  # Increment recognition count
            if recognition_count[name] >= 100:
                if not employee.check_attendance():
                    employee.send_telegram_msg(name)
                    # sec = ft.fetch_time_second()
                    # if (ft.fetch_time_second() > sec + 5) : 
                    isrecog = True
                    recogname = name
                    # else :
                    # isrecog = False
                    # recogname = ""
                    if recognition_count[name] == 30:
                        isrecog = False
                        recogname = ""
                recognition_count[name] = 0
                
        elif confidence_text<72:
            name = "unknown"
            confidence_text = "{0}%".format(confidence_text)
            cv2.putText(img, "Fixed your angle camera", (x+5, y+h+20), font, 1, (0, 0, 255), 2)

        if(isrecog):    
            cv2.putText(img, recogname + " telah hadir", (x+5, y+h+50), font, 1, (0, 0, 255), 2)
        cv2.putText(img, name, (x+5, y-5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence_text), (x+5, y+h-5), font, 1, (255, 255, 0), 1)

    print (recognition_count)
    cv2.imshow('camera', img)

    # Press 'ESC' for exiting the video
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()