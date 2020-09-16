import os
import face_recognition
import argparse
import cv2
import pickle
from imutils import paths
import dlib
dlib.DLIB_USE_CUDA = True
'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", required=True, help="path to input directory of faces and images")
ap.add_argument("-e", "--encodings", required=True, help="path to serialized db of facial encodings")
ap.add_argument("-d", "--detection-method", type=str, default="hog", help="face detection model to use: "
                                                                          "either 'hog' or 'cnn'")
args = vars(ap.parse_args())
'''
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images("dataset"))

knownEncodings = []
knownNames = []

for (i, imagePath) in enumerate(imagePaths):
    print("[INFO] processing image {}/{}" .format(i + 1, len(imagePaths)))
    print(imagePaths[i])
    name = imagePath.split(os.path.sep)[-2]

    image = cv2.imread(imagePath)
    rbg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rbg, model="hog")
    encodings = face_recognition.face_encodings(rbg, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)

print("[INFO] serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open("encodings.pickle", "wb")
f.write(pickle.dumps(data))
f.close()