from sqlite3.dbapi2 import Cursor
import cv2
import numpy as np
import sqlite3
import os

# kết nối đến csdl sqltite3
def insertOrUpdate(id, name):
    conn = sqlite3.connect('C:\\Users\\Vince\\Documents\\Machine Learning\\Practice\\Face Regonition With Cam\\FaceData.db')
    # câu lệnh sql
    # kiểm tra id nhập vào có tồn tại chưa
    # nếu rồi thì sẽ update còn chưa thì sẽ insert
    query = "SELECT * FROM Person WHERE ID =" + str(id)
    cursor = conn.execute(query)
    isRecordExist = 0           # cờ lệnh nếu id tồn tại

    for row in cursor:
        isRecordExist = 1       # duyệt từng hàng nếu có bản ghi chuyển thành 1

    if(isRecordExist == 0):     # nếu chưa có bản ghi
        query = "INSERT INTO Person(ID, NAME) VALUES("+str(id) + ",'" + str(name) + "')"
    else:
        query = "UPDATE Person SET NAME ='" + str(name)+ "' WHERE ID =" + str(id)

    conn.execute(query)
    conn.commit()
    conn.close()

# nhận diện khuôn mặt từ cam và lưu ảnh vào csdl
# gọi API haarcascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# truy cập vào cam
cap = cv2.VideoCapture(0)

# cho user nhập vào id và name cho người trong tập ảnh
id = input("Enter ID: ")
name = input("Enter your Name: ")
insertOrUpdate(id, name)            # cập nhật csdl

sampleNum = 0

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 1.3 là scale factor, 5 là min label những điểm gần nó nhất
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # vẽ hình xung quanh mặt
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        # tạo folder để lưu ảnh
        if not os.path.exists('dataSet'):
            os.makedirs('dataSet')
        
        sampleNum += 1

        # lưu ảnh
        cv2.imwrite('dataSet\\User.' + str(id) + '.' + str(sampleNum) + '.jpg', gray[y:y+h, x:x+w])

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

# chương trình dừng khi lấy đủ 100 ảnh
    if sampleNum > 100:
        break

cap.release()
cv2.destroyAllWindows()








    
