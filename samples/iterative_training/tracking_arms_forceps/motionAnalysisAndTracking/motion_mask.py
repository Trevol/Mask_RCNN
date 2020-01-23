import numpy as np
import cv2

from samples.iterative_training.Timer import timeit


def resize(src, factor=.5):
    return cv2.resize(src, None, None, factor, factor)


def optFlowSpecificVideoIterator(videoSource, startFrame):
    cap = cv2.VideoCapture(videoSource)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, startFrame)
        r, prevFrame = cap.read()
        if not r: return
        prevGrayFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)
        while True:
            r, nextFrame = cap.read()
            if not r: break
            nextGrayFrame = cv2.cvtColor(nextFrame, cv2.COLOR_BGR2GRAY)
            yield prevGrayFrame, nextGrayFrame, prevFrame, nextFrame
            prevGrayFrame = nextGrayFrame
            prevFrame = nextFrame
    finally:
        cap.release()


def main():
    kernel = np.ones((5, 5), np.uint8)

    video = '/hdd/nfs_share/video_6.mp4'
    startFrame = 4170

    # optFlow = cv2.FarnebackOpticalFlow_create(numLevels=3, pyrScale=0.5, fastPyramids=True,
    #                                           winSize=15, numIters=3, polyN=5, polySigma=1.2, flags=0)
    optFlow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = None
    for prevGrayFrame, nextGrayFrame, prevFrame, nextFrame in optFlowSpecificVideoIterator(video, startFrame):

        flow = optFlow.calc(prevGrayFrame, nextGrayFrame, flow)
        # flow = cv2.optflow.calcOpticalFlowSF(prevFrame, nextFrame, 3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18,
        #                                      55.0, 25.5, 10)

        flow = np.abs(flow)

        # motionMask = np.logical_or(flow[..., 0] > 1, flow[..., 1] > 1)
        # uintMask = np.uint8(motionMask * 255)
        # morphMask = cv2.morphologyEx(uintMask, cv2.MORPH_OPEN, kernel, iterations=2)
        # morphMask = cv2.morphologyEx(morphMask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # morphMask = cv2.erode(uintMask, kernel, iterations=2)
        # morphMask = cv2.dilate(morphMask, kernel, iterations=2)

        normalizedFlow = cv2.normalize(flow[..., 0] + flow[..., 1], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # cv2.imshow('mask', resize(np.uint8(motionMask * 255)))
        # cv2.imshow('morth', resize(morphMask))
        cv2.imshow('normalizedFlow', resize(normalizedFlow))
        cv2.imshow('frame', resize(nextFrame))

        while True:
            key = cv2.waitKey()
            if key == 27:
                return
            if key in [ord(' ')]:
                break


main()
