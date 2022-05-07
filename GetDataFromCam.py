# Khai báo thư viện
import cv2
import numpy as np


# thư viện khuôn mặt
# API nhận diện mặt người 
# tên biến face_cascade hàm cv2.CascadeClassifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# API nhận diện mắt người
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# truy cập cam
# hàm cv2.VideoCapture sử dụng cam số 0
cap = cv2.VideoCapture(0)
# lấy video liên tục bằng vòng while
# truy cập cam thành công ret là true
# frame dữ liệu từ cam


while(True):                        
    ret, frame = cap.read() 
    # đổi video qua ảnh xám để phù hợp để train  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = face_cascade.detectMultiScale(gray) 
    # vẽ hình chữ nhật màu 225 độ dày 2 bao quanh khuôn mặt 
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,225,0), 2)
    cv2.imshow('Detecting faces', frame)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()          #giải phóng vùng nhớ
cv2.destroyAllWindows() #đóng cửa sổ

