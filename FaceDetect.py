import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4,480)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_id = input('\n Nhap ID Khuon Mat <return> ==> ')

print("\n [INFO] Khoi Tao Camera ...")
count = 0

while(True):
    
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0),2)
        count +=1

        cv2.imwrite("dataset/anh." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif count >= 20:
        break

print("/n [INFO] Thoat")
cam.release()
cv2.destroyAllWindows()