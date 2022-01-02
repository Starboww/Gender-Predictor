import cv2
cap = cv2.VideoCapture(0)# you can provide a path of video also #BGR
while True:

    retval, image = cap.read()

    if retval:

        cv2.imshow("MY camera",image)

    key = cv2.waitKey(30)  # we are refreshing data in  30 times every sec (1 sec = 1000 mili sec) here 30 is in mili sec
    if key == ord("q"):
        break
    if key == ord("c"):
        cv2.imwrite("classroom.png", image)


