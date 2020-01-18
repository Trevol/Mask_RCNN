import itertools
import numpy as np

import cv2

from samples.iterative_training.Timer import timeit


def drawOpticalFlow(srcImage, flow, color):
    assert srcImage.shape[:2] == flow.shape[:2]
    # flowX = flow[..., 0]
    # flowY = flow[..., 1]
    h, w = srcImage.shape[:2]
    step = 10
    for x in range(step, w, step):
        for y in range(step, h, step):
            flowInPoint = flow[y, x]
            flowX = int(round(flowInPoint[0]))
            flowY = int(round(flowInPoint[1]))
            if abs(flowX) <= 1 and abs(flowY) <= 1:
                continue
            endOfFlow = (x + flowX, y + flowY)
            cv2.circle(srcImage, (x, y), 1, color, -1)
            cv2.line(srcImage, (x, y), endOfFlow, color, 1)
    return srcImage


def imageSequence():
    pathTemplate = '/HDD_DATA/nfs_share/video_6/{num:06d}.jpg'
    pos1 = 4650
    i = 0
    while True:
        frame = cv2.imread(pathTemplate.format(num=pos1 + i), cv2.IMREAD_GRAYSCALE)
        i += 1
        yield cv2.resize(frame, None, None, .5, .5)


def imageSequence_2():
    shape = [256, 256]

    def bgFn():
        return np.zeros(shape, np.uint8)

    start = np.array([20, 20])
    step = np.array([3, 7])
    r = 18
    color = 127
    i = 0
    while True:
        bg = bgFn()
        center = start + i * step
        i += 1
        yield cv2.circle(bg, tuple(center), r, color, thickness=-1)


def opticalFlow(prev, next):
    # print('opticalFlow')
    pyr_scale = 0.5
    levels = 1
    winsize = 15
    iterations = 1
    poly_n = 5
    poly_sigma = 1.2
    flags = 0
    return cv2.calcOpticalFlowFarneback(prev, next, None, pyr_scale, levels, winsize, iterations,
                                        poly_n, poly_sigma, flags)


def main():
    def imagesFlow(stop):
        seq = imageSequence()
        prev = next(seq)
        # frame Num, frame, flow, flowVisualization
        yield 0, prev, None, prev
        for i in range(1, stop):
            n = next(seq)
            with timeit():
                flow = opticalFlow(prev, n)
            yield i, n, flow, drawOpticalFlow(n.copy(), flow, 255)
            prev = n

    sequence = itertools.cycle(imagesFlow(70))

    while True:
        i, frame, flow, flowOnFrame = next(sequence)
        cv2.imshow('frame', flowOnFrame)
        cv2.setWindowTitle('frame', f'frame {i}')

        if cv2.waitKey() == 27:
            break


main()
