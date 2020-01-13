from collections import namedtuple

import cv2
import os

ConvertionConfig = namedtuple('ConvertionConfig', ['inputFile', 'outputDir'])


def conversionConfigs():
    yield ConvertionConfig(inputFile='/HDD_DATA/nfs_share/video_2.mp4', outputDir='/HDD_DATA/nfs_share/video_2')
    yield ConvertionConfig(inputFile='/HDD_DATA/nfs_share/video_6.mp4', outputDir='/HDD_DATA/nfs_share/video_6')


def main():
    for config in conversionConfigs():
        os.makedirs(config.outputDir, exist_ok=True)
        inputName = os.path.basename(config.inputFile)
        cap = cv2.VideoCapture(config.inputFile)
        framePos = -1
        while True:
            r, frame = cap.read()
            if not r:
                break
            framePos = framePos + 1
            outputName = f'{framePos:06d}.jpg'
            outputPath = os.path.join(config.outputDir, outputName)
            cv2.imwrite(outputPath, frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
            if framePos > 0 and framePos % 100 == 0:
                print(inputName, framePos)
        cap.release()


main()
