from imgaug import augmenters as iaa
import cv2

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)),  # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5),  # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 3.0))  # blur images with a sigma of 0 to 3.0
])
images = [cv2.imread('/HDD_DATA/nfs_share/frames_6/f_7761_517400.00_517.40.jpg')]
while True:
    # 'images' should be either a 4D numpy array of shape (N, height, width, channels)
    # or a list of 3D numpy arrays, each having shape (height, width, channels).
    # Grayscale images must have shape (height, width, 1) each.
    # All images must have numpy's dtype uint8. Values are expected to be in
    # range 0-255.
    images_aug = seq(images=images)
    cv2.imshow('aug', images_aug[0])
    if cv2.waitKey() == 27:  break
