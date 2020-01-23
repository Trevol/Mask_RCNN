import cv2
import numpy as np


def framesSequence():
    videoSource = '/hdd/nfs_share/video_6.mp4'

    startFrame = 4180
    x1, y1, x2, y2 = (305, 559, 459, 691)
    initialRect = (x1, y1, x2 - x1, y2 - y1)

    cap = cv2.VideoCapture(videoSource)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        r, frame = cap.read()
        if not r: return
        yield frame, initialRect
        while True:
            r, frame = cap.read()
            if not r: break
            yield frame
    finally:
        cap.release()


def main():
    # https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/
    # KCF tracker
    tracker = cv2.TrackerKCF_create()
    frames = framesSequence()
    frame, initialRect = next(frames)

    tracker.init(frame, initialRect)
    x, y, w, h = initialRect
    dispFrame = cv2.rectangle(frame.copy(), (x, y), (x + w, y + h),
                              (0, 200, 0), 1)
    cv2.imshow('frame', dispFrame)
    cv2.waitKey()

    for frame in frames:
        success, rect = tracker.update(frame)
        if not success:
            print('not success', rect)
            return

        x, y, w, h = rect
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        dispFrame = cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (200, 0, 0), 1)
        cv2.imshow('frame', dispFrame)

        key = cv2.waitKey()
        if key == 27:
            break


main()
