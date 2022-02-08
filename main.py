import cv2
import numpy as np

cap = cv2.VideoCapture(0)

stop=cv2.imread("images/stop.png.jpeg")
stop=cv2.resize(stop,(64,64))
stop=cv2.inRange(stop,(89, 91, 149), (255, 255, 255))
cv2.imshow("stop", stop)

while True:
    ret, frame = cap.read()
    frameCopy = frame.copy()

    cv2.imshow("Frame", frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv = cv2.blur(hsv, (5,5))

    mask = cv2.inRange(hsv, (89, 124, 73), (255, 255, 255))
    #cv2.imshow("Mask", mask)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=4)
    #cv2.imshow("Mask2", mask)

    contours = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = contours[0]

    if contours:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        cv2.drawContours(frame, contours, 0, (255, 0, 255), 3)
        #cv2.imshow('Conours', frame)

        (x, y, w, h) = cv2.boundingRect(contours[0])
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('Rect', frame)

        roImg = frameCopy[y:y+h, x:x+w]

        cv2.imshow("Detect", roImg)
        roImg=cv2.resize(roImg, (64,64))
        roImg=cv2.inRange(roImg, (89,91,149), (255,255,255))
        cv2.imshow("ResizedRoi", roImg)

        stop_val=0


        for i in range(64):
            for j in range(64):
                if roImg[i][j]==stop[i][j]:
                    stop_val+=1

        print(stop_val)

        if stop_val in range(1600, 1950):
            print("уступи дорогу")
        elif stop_val in range(1950, 2300):
            print('пешеходный переход')
        elif stop_val in range(2500, 2900):
            print('вперед')
        elif stop_val in range(3000, 3100):
            print('направо')
        elif stop_val in range(3100, 3200):
            print('налево')
        elif stop_val in range(2900, 3000):
            print('стоп')
        elif stop_val > 3300:
            print('кирпич')

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()