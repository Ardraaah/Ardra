import cv2
import numpy as np

face_model = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def deepfake_artifact_score(face):
    blur = cv2.GaussianBlur(face,(5,5),0)
    diff = cv2.absdiff(face,blur)
    noise = np.mean(diff)
    return noise


def detect_deepfake(image_path):

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_model.detectMultiScale(gray,1.3,5)

    if len(faces)==0:
        return "No face detected",0

    scores=[]

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]

        texture = np.var(face)
        edges = cv2.Canny(face,100,200)
        edge_score = np.mean(edges)
        artifact = deepfake_artifact_score(face)

        score = (texture*0.4)+(edge_score*0.2)+(artifact*0.4)
        scores.append(score)

    final = np.mean(scores)/10

    if final>65:
        label="REAL FACE"
    elif final>40:
        label="SUSPICIOUS"
    else:
        label="DEEPFAKE FACE"

    return label, round(final,2)