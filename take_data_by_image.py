import cv2
import os

# Function to clean image by removing noise
def clean_image(image):
    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image

# Initialize face detector
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

# Database directory
database_dir = 'database/'

# Define the offset percentages for width and height adjustments
offsetPercentageW = 5  # Adjust this value according to your requirements
offsetPercentageH = 5  # Adjust this value according to your requirements

count = 0  # Initialize face count
max_samples = 600  # Maximum number of face samples to capture

# Process each image in the directory
for filename in os.listdir(database_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load image
        img_path = os.path.join(database_dir, filename)
        img = cv2.imread(img_path)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Clean the image by removing noise
        cleaned_image = clean_image(gray)

        # Apply normalization for better image quality
        cleaned_image = cv2.equalizeHist(cleaned_image)

        # Resize image to a uniform size
        img = cv2.resize(img, (640, 480))

        # Detect faces in the cleaned image
        faces = face_detector.detectMultiScale(cleaned_image, 1.3, 5)

        print(f"Number of faces detected in {filename}: {len(faces)}")  # Debugging statement

        for (x, y, w, h) in faces:
            # Add offset to the detected face coordinates
            offsetW = (offsetPercentageW / 100) * w
            x = int(x - offsetW)
            w = int(w + offsetW * 2)
            offsetH = (offsetPercentageH / 100) * h
            y = int(y - offsetH * 3)
            h = int(h + offsetH * 3.5)

            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Save the face region into the datasets folder
            cv2.imwrite("dataset/" + filename[:-4] + '_face_' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            # Display the image
            cv2.imshow('image', img)

            count += 1
            if count >= max_samples:
                break

        if count >= max_samples:
            break

# Cleanup
print("\n[INFO] Exiting Program and cleanup stuff")
cv2.destroyAllWindows()
