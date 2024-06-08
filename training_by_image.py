import cv2
import numpy as np
import os
from PIL import Image

# Path to the dataset directory
path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    faceSamples = []
    ids = []
    
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # Convert to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            # Extract the user ID from the filename
            filename = os.path.basename(imagePath)
            user_id = int(filename.split('.')[0].replace('User', ''))
            ids.append(user_id)
    
    return faceSamples, ids

print("\n[INFO] Training faces. It will take a few seconds. Please wait...")

faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the trained model
if not os.path.exists('trainer'):
    os.makedirs('trainer')
recognizer.save('trainer/trainer.yml')

print("\n[INFO] Training complete. {0} faces trained. Exiting Program.".format(len(np.unique(ids))))
