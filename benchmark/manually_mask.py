# create a script to manually mask the object in the scene


import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# input image
image_path = 'benchmark_dataset/OpenSlideCabinet/right/color_000000.png'
img = cv2.imread(image_path, -1)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask_org_path = image_path.replace('color_000000.png', 'mask_000001.png')
mask_org = cv2.imread(mask_org_path, -1)


# select corner points using matpltlib

fig, ax = plt.subplots(figsize=(12, 8))
ax.set_title("please choose the object that you want to mask")
ax.imshow(img)

points = []
labels = []

def onclick(event):
    if event.button == 1:  # Left click for positive point
        points.append([event.xdata, event.ydata])
        labels.append(1)
        ax.scatter(event.xdata, event.ydata, color='green', marker='*', s=200, edgecolor='white', linewidth=1.25)
    elif event.button == 3:  # Right click for negative point
        points.append([event.xdata, event.ydata])
        labels.append(0)
        ax.scatter(event.xdata, event.ydata, color='red', marker='*', s=200, edgecolor='white', linewidth=1.25)
    fig.canvas.draw()

fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()
print("points: ", points)
print("labels: ", labels)
# convert points to mask
assert len(points) == 2, "please select two points"
left_pt = points[0]
right_pt = points[1]
mask = np.zeros(img.shape[:2], dtype=np.uint8)
mask[int(left_pt[1]):int(right_pt[1]), int(left_pt[0]):int(right_pt[0])] = 1
# import ipdb; ipdb.set_trace()

mask_final = mask & mask_org

# save mask
mask_save_path = image_path.replace('color_000000.png', 'mask_000000.png')
cv2.imwrite(mask_save_path, mask_final*255)