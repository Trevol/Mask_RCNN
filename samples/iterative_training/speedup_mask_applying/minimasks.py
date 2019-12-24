import skimage.io
import numpy as np
from cv2 import cv2

from samples.iterative_training.ImshowWindow import ImshowWindow
from samples.iterative_training.MaskRCNNEx import MaskRCNNEx
from samples.iterative_training.Timer import timeit
from samples.iterative_training.Utils import Utils
from samples.iterative_training.pins.PinsConfig import PinsInferenceConfig
from mrcnn import utils


class NewDetector:
    @staticmethod
    def unmold_mask(mask, bbox):
        """Converts a mask generated by the neural network to a format similar
        to its original shape.
        mask: [height, width] of type float. A small, typically 28x28 mask.
        bbox: [y1, x1, y2, x2]. The box to fit the mask in.

        Returns a binary mask with the same size as the bbox.
        """

        threshold = 0.5
        y1, x1, y2, x2 = bbox
        mask = utils.resize(mask, (y2 - y1, x2 - x1))
        mask = mask >= threshold
        # full_mask[y1:y2, x1:x2] = mask # Put the mask in the right location.
        return mask

    @staticmethod
    def unmold_detections(detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """

        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image

        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to mask with size of box
            full_mask = NewDetector.unmold_mask(masks[i], boxes[i])
            full_masks.append(full_mask)

        return boxes, class_ids, scores, full_masks

    @staticmethod
    def detect(model, images):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert model.mode == "inference", "Create model in inference mode."
        assert len(images) == model.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = model.mold_inputs(images)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = model.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (model.config.BATCH_SIZE,) + anchors.shape)

        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ = \
            model.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks = \
                NewDetector.unmold_detections(detections[i], mrcnn_mask[i],
                                              image.shape, molded_images[i].shape,
                                              windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    @staticmethod
    def applyMasks(image, bboxes, masks, colorMap, alpha=.5):
        if colorMap is None:
            colorMap = Utils.createColorMap()
        zeroInstanceMask = masks[0]
        instanceIndexes = np.argmax(masks, -1).astype(np.uint8)
        # First instance marked by 0 and BG has 0 value
        # so set different value for first instance
        instanceIndexes[zeroInstanceMask] = 255
        instancesMask = instanceIndexes.astype(np.bool)
        coloredInstances = cv2.applyColorMap(instanceIndexes, colorMap)
        blendedAndMasked = cv2.addWeighted(image, 1 - alpha, coloredInstances, alpha, 0)[instancesMask]
        image[instancesMask] = blendedAndMasked
        return image

    @staticmethod
    def applyBboxMasks(image, bboxes, masks, colorMap, alpha=.5):
        if colorMap is None:
            colorMap = Utils.createColorMap()
        for i, ((y1, x1, y2, x2), mask) in enumerate(zip(bboxes, masks)):
            color = colorMap[i, 0]
            h = y2 - y1
            w = x2 - x1
            colored = np.full([h, w, 3], color, np.uint8)
            image[y1:y2, x1:x2][mask] = image[y1:y2, x1:x2][mask] * alpha + colored[mask] * (1 - alpha)
        return image


def main():
    config = PinsInferenceConfig()
    modelDir = 'logs'
    model = MaskRCNNEx(mode='inference', config=config, model_dir=modelDir)
    model.load_weights('mask_rcnn_pins_0008.h5', by_name=True)

    image = skimage.io.imread('f_7926_528400.00_528.40.jpg')

    # r = model.detect([image])[0]    # 0.9291874,  0.9310132,  0.9401661
    # old_boxes, old_masks, old_classIds, old_scores = r['rois'], r['masks'], r['class_ids'], r['scores']
    # instancesImage = Utils.display_instances(image, old_boxes, old_masks, old_classIds, old_scores)

    with timeit():
        r = NewDetector.detect(model, [image])[0]
    # with timeit():
    #     r = NewDetector.detect(model, [image])[0]
    # with timeit():
    #     r = NewDetector.detect(model, [image])[0]
    # with timeit():
    #     r = NewDetector.detect(model, [image])[0]

    boxes, masks, classIds, scores = r['rois'], r['masks'], r['class_ids'], r['scores']
    colors = Utils.random_colors(256)
    colorMap = np.reshape(colors, [256, 1, 3]).astype(np.uint8)
    instancesImage = NewDetector.applyBboxMasks(image, boxes, masks, colorMap)
    ImshowWindow('').imshow(instancesImage)
    cv2.waitKey()


main()
