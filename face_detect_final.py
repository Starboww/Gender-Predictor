import os.path
import os
import cv2
import numpy as np

from sklearn.neighbors import  KNeighborsClassifier
cap = cv2.VideoCapture(0)  # you can provide a path of video also     #BGR
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
data = np.load("faces.npy")
X=data[:,:-1].astype(int)# all rows column 1 and above
y=data[:,-1]
model = KNeighborsClassifier(4)
model.fit(X,y)

while True:
    ret, image = cap.read()

    if ret:
        faces = classifier.detectMultiScale(image, scaleFactor=1.3)
        for face in faces:
            [x, y, w, h] = face
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            chopped = image[y:y + h, x:x + w]
            chopped = cv2.resize(chopped, (50, 50))
            # print("chopped shape",np.array(chopped).shape)
            gray = cv2.cvtColor(chopped, cv2.COLOR_BGR2GRAY)
            names= model.predict([gray.flatten()])
            cv2.putText(image, names[0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imshow("my cam", image)


    key = cv2.waitKey(1)
    if key == ord("q"):
        break



