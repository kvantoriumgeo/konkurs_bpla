import cv2
import numpy as np

cap=cv2.VideoCapture(0)

img_size = [200,360] # Размеры изображения с которым мы работаем

src = np.float32([[20,200],
                [350,200],
                [275,120],
                [85,120]])

src_draw=np.array(src, dtype=np.int32)

dst=np.float32([[0, img_size[0]],
                 [img_size[1],img_size[0]],
                 [img_size[1],0],
                 [0,0]])

while (cv2.waitKey (1) !=27):
    ret, frame = cap.read()

    resized = cv2.resize(frame, (img_size[1], img_size[0]))
    cv2.imshow("frame",resized)
    r_channel=resized[:,:,2]
    binary=np.zeros_like(r_channel)
    binary[(r_channel>200)]=1
    #cv2.imshow("r_channel",binary)

    hls=cv2.cvtColor(resized,cv2.COLOR_BGR2HLS)
    s_channel=resized[:,:,2]
    binary2 = np.zeros_like(s_channel)
    binary2[(r_channel>160)]=1

    allBinary=np.zeros_like(binary)
    allBinary[((binary==1)|(binary2==1))]=255
    cv2.imshow("binary",allBinary)

    allBinary_visual=allBinary.copy()
    cv2.polylines(allBinary_visual,[src_draw],True,255)
    cv2.imshow("polygon",allBinary_visual)

    M = cv2.getPerspectiveTransform(src,dst)
    warped = cv2.warpPerspective(allBinary, M, (img_size[1],img_size[0]),flags=cv2.INTER_LINEAR)
    cv2.imshow("warped",warped)

    histogram = np.sum(warped[warped.shape[0]//2:,:],axis=0)

    midpoint = histogram.shape[0]//2
    IndWhitestColumnL = np.argmax(histogram[:midpoint])
    IndWhitestColumnR = np.argmax(histogram[midpoint:])+midpoint
    warped_visual = warped.copy()
    cv2.line(warped_visual, (IndWhitestColumnL,0), (IndWhitestColumnL, warped_visual.shape[0]),110,2)
    cv2.line(warped_visual, (IndWhitestColumnR,0), (IndWhitestColumnR, warped_visual.shape[0]),110,2)
    cv2.imshow("WitestColumn", warped_visual)

    nwindows = 9
    window_height = np.int32(warped.shape[0]/nwindows)
    window_half_width = 25

    XCenterLeftWindow = IndWhitestColumnL
    XCenterRightWindow = IndWhitestColumnR

    left_lane_inds = np.array([], dtype=np.int16)
    right_lane_inds = np.array([], dtype=np.int16)

    out_img = np.dstack((warped, warped, warped))

    nonzero = warped.nonzero ()
    WhitePixelIndY = np.array(nonzero[0])
    WhitePixelIndX = np.array(nonzero[1])

    for window in range(nwindows):


        win_y1 = warped.shape[0] - (window+1) * window_height
        win_y2 = warped.shape[0] - (window) * window_height

        left_win_x1 = XCenterLeftWindow-window_half_width
        left_win_x2 = XCenterLeftWindow+window_half_width
        right_win_x1 = XCenterRightWindow-window_half_width
        right_win_x2 = XCenterRightWindow+window_half_width


        cv2.rectangle(out_img, (left_win_x1, win_y1),(left_win_x2, win_y2),(50 + window *21, 0,0),2)
        cv2.rectangle(out_img, (right_win_x1, win_y1),(right_win_x2, win_y2),(0, 0, 50 + window *21),2)
        cv2.imshow("windows", out_img)

        good_left_inds = ((WhitePixelIndY>=win_y1) & (WhitePixelIndY<=win_y2) & (WhitePixelIndX>= left_win_x1) & (WhitePixelIndX <= left_win_x2)).nonzero()[0]
        good_right_inds = ((WhitePixelIndY>=win_y1) & (WhitePixelIndY<=win_y2) & (WhitePixelIndX>= right_win_x1) & (WhitePixelIndX <= right_win_x2)).nonzero()[0]

        left_lane_inds = np.concatenate ((left_lane_inds, good_left_inds))
        right_lane_inds = np.concatenate ((right_lane_inds, good_right_inds))

        if len (good_left_inds) > 50:
            XCenterLeftWindow = np.int32(np.mean(WhitePixelIndX[good_left_inds]))
        if len (good_right_inds)> 50:
            XCenterRightWindow = np.int32(np.mean (WhitePixelIndX [good_right_inds]))

    out_img[WhitePixelIndY[left_lane_inds],WhitePixelIndX [left_lane_inds]]=[255,0,0]
    out_img[WhitePixelIndY[right_lane_inds],WhitePixelIndX [right_lane_inds]]=[0,0,255]
    cv2.imshow("Lane", out_img)

    leftx=WhitePixelIndX[left_lane_inds]
    lefty=WhitePixelIndY[left_lane_inds]
    rightx=WhitePixelIndX[right_lane_inds]
    righty=WhitePixelIndY[right_lane_inds]

    left_fit=np.polyfit(lefty, leftx,2)
    right_fit=np.polyfit (righty, rightx, 2)

    center_fit=((left_fit+right_fit)/2)

    for ver_ind in range(out_img.shape[0]):
        gor_ind = ((center_fit[0]) * (ver_ind ** 2) +
                   center_fit[1] * ver_ind + center_fit[2])
        cv2.circle(out_img, (int (gor_ind),int (ver_ind)), 2, (255,0,255),1)

    cv2.imshow ("CenterLine", out_img)
