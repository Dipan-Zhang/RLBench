import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# read avi file videos from the directory
videos_dir = './outputs/rlbench_videos/'
video_files = [f for f in os.listdir(videos_dir) if f.endswith('.avi')]
# sort the video files
video_files.sort()
# create a list to store the 10th frames
frames = []
# loop through the video files
for video_file in tqdm(video_files):
    video_path = os.path.join(videos_dir, video_file)
    # read the video
    cap = cv2.VideoCapture(video_path)
    # check if the video is opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_file}")
        continue
    # read the 10th frame
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read frame {i} from video {video_file}")
            break
    # append the 10th frame to the list
    frames.append(frame)
    # release the video capture object
    cap.release()
# create a grid of frames   
num_frames = len(frames)
num_cols = 3
num_rows = (num_frames + num_cols - 1) // num_cols
# create a blank image for the grid
grid_height = frames[0].shape[0] * num_rows
grid_width = frames[0].shape[1] * num_cols
grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
# loop through the frames and place them in the grid
for i, frame in enumerate(frames):
    row = i // num_cols
    col = i % num_cols
    y_start = row * frame.shape[0]
    y_end = y_start + frame.shape[0]
    x_start = col * frame.shape[1]
    x_end = x_start + frame.shape[1]
    grid[y_start:y_end, x_start:x_end] = frame
# save the grid as a thumbnail
thumbnail_path = os.path.join(videos_dir, 'task_thumbnail.png')
cv2.imwrite(thumbnail_path, grid)

# save the grid in a slight smaller size
thumbnail_path = os.path.join(videos_dir, 'task_thumbnail_small.jpg')
thumbnail = cv2.resize(grid, (grid_width//3, grid_height//3))
cv2.imwrite(thumbnail_path, thumbnail)