python ./train_for_arm_tracking.py --lr=.001 --max-sessions=3
python ./process_video_frames.py --images_per_gpu=8
python ./train_for_arm_tracking.py --lr=.001 --max-sessions=3
python ./process_video_frames.py --images_per_gpu=8
python ./train_for_arm_tracking.py --lr=.001 --max-sessions=3
python ./process_video_frames.py --images_per_gpu=8