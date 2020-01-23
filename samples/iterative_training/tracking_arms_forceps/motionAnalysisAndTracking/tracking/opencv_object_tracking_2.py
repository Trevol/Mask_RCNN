# USAGE
# python opencv_object_tracking.py
# python opencv_object_tracking.py --video dashcam_boston.mp4 --tracker csrt

from imutils.video import FPS
import imutils
import time
import cv2
import numpy as np
from kcf_dsst_tracker.tracker import KCFTracker as KCFDSSTTracker


def main():
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "mosse": cv2.TrackerMOSSE_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "kcf_dsst": lambda: KCFDSSTTracker(True, True, True),
    }

    trackerName = 'kcf_dsst'
    tracker = OPENCV_OBJECT_TRACKERS[trackerName]()
    # tracker.save("default_csrt.xml")
    # fs = cv2.FileStorage("default_csrt.xml", cv2.FILE_STORAGE_READ)
    # fn = fs.getFirstTopLevelNode()
    # tracker.read(fn)

    initBB = None

    # videoSource = 'dashcam_boston.mp4'
    videoSource = '/hdd/nfs_share/video_6.mp4'

    # startFrame = 5101  # 4170 + 900
    # initBB = (634, 444, 105, 85)

    initBB = (633, 490, 114, 85)
    startFrame = 5102

    vs = cv2.VideoCapture(videoSource)
    vs.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

    if initBB and any(initBB):
        frame = vs.read()[1]
        if frame is None:
            return
        frame = imutils.resize(frame, height=900)
        # tracker.init(frame, initBB)
        tracker.init(initBB, frame)
        (x, y, w, h) = (int(v) for v in initBB)
        f = cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow("Frame", f)
        cv2.waitKey()

    while True:
        frame = vs.read()[1]
        if frame is None:
            break

        dispFrame = frame = imutils.resize(frame, height=900)
        (H, W) = frame.shape[:2]

        # check to see if we are currently tracking an object
        if initBB is not None:
            # grab the new bounding box coordinates of the object
            # (success, box) = tracker.update(frame)
            (success, box) = True, tracker.update(frame)

            # check to see if the tracking was a success
            if success:
                (x, y, w, h) = [int(v) for v in box]
                dispFrame = cv2.rectangle(frame.copy(), (x, y), (x + w, y + h), (0, 255, 0), 2)

            # initialize the set of information we'll be displaying on
            # the frame
            info = [
                ("Tracker", trackerName),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(0)),
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(dispFrame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # show the output frame
        cv2.imshow("Frame", dispFrame)
        key = cv2.waitKey()

        # if the 's' key is selected, we are going to "select" a bounding
        # box to track
        if key == ord("s"):
            # select the bounding box of the object we want to track (make
            # sure you press ENTER or SPACE after selecting the ROI)
            initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                                   showCrosshair=True)
            if any(initBB):
                print(initBB, vs.get(cv2.CAP_PROP_POS_FRAMES))
                # tracker.clear()
                tracker = OPENCV_OBJECT_TRACKERS[trackerName]()
                # tracker.read(fn)
                # _, _, w, h = initBB
                # mask = np.ones([h*2, w*2], np.float32)
                # tracker.setInitialMask(mask)

                # tracker.init(frame, initBB)
                tracker.init(initBB, frame)
            else:
                initBB = None

        # if the `q` key was pressed, break from the loop
        elif key in [ord("q"), 27]:
            break

    vs.release()

    # close all windows
    cv2.destroyAllWindows()


main()
