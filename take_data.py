import cv2

# Function to clean image by removing noise
def clean_image(image):
    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

# For each person, enter one numeric face id
face_id = input('\n enter user id end press <return> ==> ')
print("\n [INFO] Initializing face capture. Look at the camera and wait...")

# Initialize individual sampling face count
count = 0
# Define the offset percentages for width and height adjustments
offsetPercentageW = 5  # Adjust this value according to your requirements
offsetPercentageH = 5  # Adjust this value according to your requirements

while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Clean the image by removing noise
    cleaned_image = clean_image(gray)

    # Apply normalization for better image quality
    cleaned_image = cv2.equalizeHist(cleaned_image)
    
    # Resize image to a uniform size
    img = cv2.resize(img, (640, 480))

    faces = face_detector.detectMultiScale(cleaned_image, 1.3, 5)

    for (x, y, w, h) in faces:
        # Add offset to the detected face coordinates
        offsetW = (offsetPercentageW / 100) * w
        x = int(x - offsetW)
        w = int(w + offsetW * 2)
        offsetH = (offsetPercentageH / 100) * h
        y = int(y - offsetH * 3)
        h = int(h + offsetH * 3.5)

        # Draw the rectangle with the adjusted coordinates
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' +
                    str(count) + ".jpg", gray[y:y + h, x:x + w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27 or count >= 40:  # Take 40 face samples and stop video
        break

# Cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
