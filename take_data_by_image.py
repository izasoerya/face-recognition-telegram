import cv2
import os

# Function to clean image by removing noise
def clean_image(image):
    # Apply Gaussian Blur to reduce noise
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    return blurred_image

# Initialize face detector
face_detector = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

# Main directory containing subfolders with images
main_dir = 'lek1/'

# Define the offset percentages for width and height adjustments
offsetPercentageW = 5  # Adjust this value according to your requirements
offsetPercentageH = 5  # Adjust this value according to your requirements

count = 0  # Initialize face count
max_samples = 600  # Maximum number of face samples to capture

# Mapping of folder names to user IDs
folder_to_user_id = {
    'aaa': 'User0',
    'akbaresa': 'User1',
    'alfianto': 'User2',
    'bima': 'User3',
    'budiyanto': 'User4',
    'kenrio': 'User5',
    'kukuh': 'User6',
    'rana': 'User7',
    'Raul': 'User8',
    'riski mas': 'User9'
}

# Create dataset directory if it doesn't exist
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Process each subfolder in the main directory
for subfolder in os.listdir(main_dir):
    subfolder_path = os.path.join(main_dir, subfolder)
    if os.path.isdir(subfolder_path):
        user_id = folder_to_user_id.get(subfolder, 'Unknown')

        # Process each image in the subfolder
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # Load image
                img_path = os.path.join(subfolder_path, filename)
                img = cv2.imread(img_path)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Clean the image by removing noise
                cleaned_image = clean_image(gray)

                # Apply normalization for better image quality
                cleaned_image = cv2.equalizeHist(cleaned_image)

                # Resize image to a uniform size
                img = cv2.resize(img, (640, 480))

                # Detect faces in the cleaned image
                faces = face_detector.detectMultiScale(cleaned_image, 1.3, 10)

                print(f"Number of faces detected in {filename}: {len(faces)}")  # Debugging statement

                for (x, y, w, h) in faces:
                    # Add offset to the detected face coordinates
                    offsetW = (offsetPercentageW / 100) * w
                    x = int(x - offsetW)
                    w = int(w + offsetW * 2)
                    offsetH = (offsetPercentageH / 100) * h
                    y = int(y - offsetH * 3)
                    h = int(h + offsetH * 3.5)

                    # Ensure the coordinates are within image boundaries
                    x = max(0, x)
                    y = max(0, y)
                    w = min(w, img.shape[1] - x)
                    h = min(h, img.shape[0] - y)

                    # Draw rectangle around the face
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

                    # Create new filename for the face image
                    new_filename = f"{user_id}.{filename.split('_')[1]}"

                    # Ensure the cropped face region is not empty
                    face_region = gray[y:y + h, x:x + w]
                    if face_region.size > 0:
                        # Save the face region into the dataset folder
                        cv2.imwrite(os.path.join("dataset", new_filename), face_region)

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
