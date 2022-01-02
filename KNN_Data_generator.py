import os.path

import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # you can provide a path of video also     #BGR
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
name = input("enter your name")
count = 10
path = "./faces.npy"
data = []
capture = False
while count > 0:
    ret, image = cap.read()

    if ret:
        faces = classifier.detectMultiScale(image, scaleFactor=1.3)
        for face in faces:
            [x, y, w, h] = face
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 5)
            chopped = image[y:y + h, x:x + w]
            chopped = cv2.resize(chopped, (50, 50))
           # print("chopped shape",np.array(chopped).shape)
            gray = cv2.cvtColor(chopped, cv2.COLOR_BGR2GRAY)
            cv2.imshow("my cam", image)
            if capture:
                xt= gray.flatten()
               # print(" xt shape",np.array(xt).shape)
                data.append(gray.flatten())
                # 50 X 50 2D array flattened to 1X2500  1 D array
              #  print(" data shape is :",np.array(data).shape)
                count -= 1
                print(count, "captures remaining")
                capture = False




    key = cv2.waitKey(1)
    if key == ord("q"):
        break
    if key == ord("c"):
        capture = True

    X = np.array(data)
    y = np.array([[name]] * len(data))  #as x ,y  were coordinates we are stacking them horizontally
# print("X.shape:",X.shape)
# print(X)
#
# print("y.shap:",y.shape)
# print(y)
# print("len of data",len(data))
# print(data)
output = np.hstack([X, y])
if os.path.exists(path):
    old = np.load(path)
    output= np.vstack([old,output]) #we pasted our new data over old data
np.save(path,output) #rewriting the file  with (old + new )data  which is stacked vertically

