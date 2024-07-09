import cv2
import os

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4,480)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Nhập tên của người dùng
user_name = input('\nNhập tên của bạn: ')

print("\n[INFO] Khởi tạo Camera...")
count = 0

# Tạo thư mục lưu trữ ảnh của người dùng nếu chưa tồn tại
user_folder = 'dataset/' + user_name
if not os.path.exists(user_folder):
    os.makedirs(user_folder)

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        # Lưu ảnh vào thư mục của người dùng
        cv2.imwrite(user_folder + "/img." + str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif count >= 20:
        break

print("\n[INFO] Thoát")
cam.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
from PIL import Image
import os

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        # Get ID from folder name
        id = int(os.path.basename(imagePath))
        
        # Read all images in the folder
        for filename in os.listdir(imagePath):
            imgPath = os.path.join(imagePath, filename)
            PIL_img = Image.open(imgPath).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')
            
            # Detect faces in the image
            faces = detector.detectMultiScale(img_numpy)
            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)

    return faceSamples, ids

path = 'dataset'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("\n[INFO] Đang train dữ liệu...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Lưu mô hình đã train
recognizer.write('trainer/train.yml')

print("\n[INFO] {0} khuôn mặt đã được train. Thoát.".format(len(np.unique(ids))))
