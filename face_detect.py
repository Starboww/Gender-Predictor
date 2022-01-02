import cv2

cap = cv2.VideoCapture(0)  # you can provide a path of video also #BGR
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
     ret, image = cap.read()

     if ret:
        faces = classifier.detectMultiScale(image, scaleFactor=1.3)
        for face in faces:
            [x, y, w, h] = face
            image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("my cam",image)

     key = cv2.waitKey(30)
     if key == ord("q"):
        break
     if key == ord("c"):
            cv2.imwrite("classroom.png", image)
