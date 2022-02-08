import cv2
import numpy as np

cap=cv2.VideoCapture(0)

if cap.isOpened()==False:
    print("Cannot open input video")
    exit()

img_size = [200,360] # Размеры изображения с которым мы работаем

while(cv2.waitKey(1) != 27):
    ret, frame = cap.read()
    if ret==False:
        print ("End of video")
        break

    resized = cv2.resize (frame, (img_size[1], img_size[0]))
    cv2.imshow("frame",resized)
    r_channel=resized[:,:,2]
    binary=np.zeros_like(r_channel)
    binary[(r_channel>200)]=1
    #cv2.imshow("r_channel",binary)

    hls=cv2.cvtColor(resized,cv2.COLOR_BGR2HLS)