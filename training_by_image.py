import cv2
import numpy as np
import os
from PIL import Image

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml")

# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]
    faceSamples = []
    ids = []
    
    # Assign a user ID for all images
    user_id = 0
    
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # Convert to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(user_id)
    
    return faceSamples, ids

print("\n[INFO] Training faces. It will take a few seconds. Please wait...")

faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the trained model
recognizer.save('trainer/trainer.yml')

user_id = 0  # Define user_id here

print("\n[INFO] Training complete. {0} faces trained for user ID {1}. Exiting Program.".format(len(np.unique(ids)), user_id))
