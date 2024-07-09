import cv2

recog = cv2.face.LBPHFaceRecognizer_create()
recog.read('trainer/train.yml')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX

id = 0
names = ['Joker','Nguyen','Obama','Hoang']

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

minW = 0.1 * cam.get(3)
minN = 0.1 * cam.get(4)

while True:

    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,scaleFactor= 1.1,minNeighbors=5,minSize=(int(minW), int(minN)))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(0,255,0), 2)
        id, conf = recog.predict(gray[y:y+h, x:x+w])

        if (conf< 100):
            id = names[id]
            conf = " {0}%".format(round(100-conf))
        else:
            id = "Unknown"
            conf = " {0}%".format(round(100-conf))

        cv2.putText(frame, str(id), (x+5,y-5), font, 1, (255,255,255), 2)

    cv2.imshow("nhan dien khuon mat", frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
print("Exit")
cam.release()
cv2.destroyAllWindows()