import cv2
import numpy as np
import os
from PIL import Image

# API nhận diện khuôn mặt
recognizer = cv2.face.LBPHFaceRecognizer_create()

# đường dẫn đến foler ảnh
path = 'dataSet'

# hàm lấy id của từng tấm ảnh trong dataSet
def getImageWithId(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    #print(imagePaths)
    # tạo mảng để lưu mặt và ID
    faces = []
    IDs = []

    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        print(faceNp)

        # các ảnh này thuộc id nào
        Id = int(imagePath.split('\\')[1].split('.')[1])
        faces.append(faceNp)
        IDs.append(Id)
        cv2.imshow('training',faceNp)
        cv2.waitKey(10)
    return faces, IDs


faces, Ids = getImageWithId(path)   # in ra tất cả path đến từng ảnh
# gọi API để train
recognizer.train(faces, np.array(Ids))

# sau khi train xong máy sẽ trả lại 1 file yml
# cần lưu file đếy lại, tạo folder để lưu
if not os.path.exists('recognizer'):
    os.makedirs('recognizer')

recognizer.save('recognizer\\trainingDate.yml')

cv2.destroyAllWindows()





