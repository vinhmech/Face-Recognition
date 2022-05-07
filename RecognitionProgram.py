import cv2
import numpy as np
import os
import sqlite3
from PIL import Image

# phải import 2 API từ 2 phần trước
# API phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# API nhận diện khuôn mặt và dữ liệu đã train
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('C:\\Users\\Vince\\Documents\\Machine Learning\\Practice\\Face Regonition With Cam\\recognizer\\trainingDate.yml')

# lấy id người dùng từ csdl
def getProfile(id):
    conn = sqlite3.connect('C:\\Users\\Vince\\Documents\\Machine Learning\\Practice\\Face Regonition With Cam\\FaceData.db')
    query = "SELECT * FROM Person WHERE ID=" + str(id)
    cursor = conn.execute(query)
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

# truy cập cam
cap = cv2.VideoCapture(0)

# đặt font chữ
fontface = cv2.FONT_HERSHEY_SIMPLEX
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,225,0),2)
        # cắt ảnh trên cam
        roi_gray = gray[y:y+h, x:x+w]
        # so sách ảnh trên cam với dữ liệu
        id, confidence = recognizer.predict(roi_gray)
        # If the confidence is higher, then it means that the pictures are less similar, or in other words the lower, the better.
        if confidence < 45:
            # lấy dữ liệu từ csdl
            profile = getProfile(id)
            if (profile != None):
                cv2.putText(frame, ""+str(profile[1]), (x+10, y+h+30), fontface, 1, (0,225,0), 2)
        else:
            cv2.putText(frame, "Unknow", (x+10, y+h+30), fontface, 1, (0,0,255), 2)
    
    cv2.imshow('image',frame)
    if(cv2.waitKey(1) == ord('q')):
        break;

cap.release()
cv2.destroyAllWindows()