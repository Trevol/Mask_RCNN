import numpy as np
import cv2


# video = '/HDD_DATA/nfs_share/video_2.mp4'
# cap = cv2.VideoCapture(video)
# cap.set(cv2.CAP_PROP_POS_FRAMES, 1700)

video = '/HDD_DATA/nfs_share/video_6.mp4'
cap = cv2.VideoCapture(video)
cap.set(cv2.CAP_PROP_POS_FRAMES, 4700)

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while True:
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.imshow('frame2', bgr)

    delay = 10 if cap.get(cv2.CAP_PROP_POS_FRAMES)<1700 else -1
    k = cv2.waitKey(delay)
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', bgr)
    prvs = next
