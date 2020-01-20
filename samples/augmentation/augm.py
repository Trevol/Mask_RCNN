from imgaug import augmenters as iaa
import cv2
import numpy as np


def imageAndMask():
    image = cv2.imread('/hdd/nfs_share/frames_6/f_7761_517400.00_517.40.jpg')
    return [image, image]


def main():
    seq = iaa.Sequential([
        iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
        iaa.Fliplr(0.5),  # horizontally flip 50% of the images
        iaa.GaussianBlur(sigma=(0, 1.5)),  # blur images with a sigma of 0 to 3.0
        iaa.Sharpen((0.0, 1.0)),
        iaa.Affine(rotate=(-10, 10)),
        iaa.Affine(shear=(-10, 10)),
        iaa.Affine(scale=(1, 1.1))
    ])

    images = imageAndMask()
    while True:
        # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
        # or a list of 3D numpy arrays, each having shape (height, width, channels).
        # Grayscale images must have shape (height, width, 1) each.
        # All images must have numpy's dtype uint8. Values are expected to be in
        # range 0-255.
        det = seq.to_deterministic()
        images_aug = [
            det.augment_image(images[0]),
            det.augment_image(images[1])
        ]
        np.testing.assert_array_equal(images_aug[0], images_aug[1])
        for i, img_aug in enumerate(images_aug):
            cv2.imshow(f'aug_{i}', img_aug)
        if cv2.waitKey() == 27:  break


main()
