# %% Librerias
import cv2
import numpy as np
# %% Codigo 

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detec_face(img) -> np.uint8:
    face_image = img.copy()
    face_rec = face_cascade.detectMultiScale(face_image,scaleFactor=1.2,minNeighbors=5)
    for x,y,w,h in face_rec:
        cv2.rectangle(face_image,(x,y),(x+w,y+h),(255,0,0),3,cv2.LINE_4)
    return face_image

def detec_eye(image)-> np.uint8:
    face_eye = image.copy()
    eye_rec = eye_cascade.detectMultiScale(face_eye,scaleFactor=1.1,minNeighbors=5)
    for x,y,w,h in eye_rec:
        cv2.rectangle(face_eye,(x,y),(x+w,y+h),(0,224,0),3,cv2.LINE_4)
    
    return face_eye

cap = cv2.VideoCapture(0)


while True:
    ret,frame = cap.read(0)
    frame = detec_face(frame)
    frame = detec_eye(frame)
    cv2.imshow("Detection Face",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow("Detection Face")