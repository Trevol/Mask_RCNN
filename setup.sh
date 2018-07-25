#!/bin/bash
apt-get update
apt-get -y install python3-pip git unzip

#install python packages
pip3 install numpy scipy Pillow cython matplotlib scikit-image tensorflow keras opencv-python h5py imgaug IPython

mkdir repos

#clone repos
cd ./repos
git clone https://github.com/Trevol/Mask_RCNN.git

cd ./Mask_RCNN 
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5

cd ./samples/balloon
mkdir dataset
cd ./dataset
wget https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip

unzip balloon_dataset.zip "balloon/*.*"

cd balloon
mv train .. && mv val .. && cd .. rm -r balloon
