import os
import sys
import numpy as np
import cv2
import skimage.draw

sys.path.append('./')

from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import visualize

class TELContactsConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "tel_contacts"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
class InferenceConfig(TELContactsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
config = InferenceConfig()

model = modellib.MaskRCNN(mode="inference", config=config, model_dir='logs')
model.load_weights("mask_rcnn_coco.h5", by_name=True)




class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


imagePaths = [os.path.join("samples/coco", f) for f in ["persons.jpg", "plain_persons.jpg", "persons.jpg"]]


import my_vis as vis

def detect_and_visualize(imgPath):
    image = cv2.imread(imgPath)    
    result = model.detect([image], verbose=0)[0]
    result_image = show_instances(image, result)
    return {'result': result, 'image': image, 'result_image': result_image}

def show_instances(image, result):
    N = result['rois'].shape[0]
    colors =  vis.random_colors(N)
    masks =  result['masks']
    for i in range(N):
        color = colors[i]
        mask = masks[:,:,i]
        vis.apply_mask(image, mask, color)    
    return image

result_images = []
masks = []

import timeit

for imName in imagePaths:
    start = timeit.default_timer()
    result = detect_and_visualize(imName)
    print(f'-------------{timeit.default_timer() - start:.3}-------------')
    result_images.append(result['result_image'])
    
    


#for i in range(len(result_images)):
 #   cv2.imshow(f'{i}', result_images[i])
    

#cv2.waitKey()



'''
detect
apply aplash
apply mask and bbox
'''    




