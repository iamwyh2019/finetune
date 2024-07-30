import os
import cv2
import numpy as np

# fname = '_0A_W6lEi-7W0RvVEiKkyQ'
# image_name = f'{fname}.jpg'
# instance_name = f'{fname}.png'

# image = cv2.imread(image_name)
# instances = cv2.imread(instance_name)

# target_id = 63
# target_pixel = (target_id, target_id, target_id)
# target = (instances == target_pixel).all(axis=2).astype(np.uint8)

# masked = cv2.bitwise_and(image, image, mask=target)
# cv2.imwrite(f'{fname}-masked.jpg', masked)

fname = '6sSAGuGHmAhrjPubRv8MXw'
image_name = f'{fname}.jpg'
label_name = f'{fname}.txt'

image = cv2.imread(image_name)
h,w = image.shape[:2]
sz = np.array([w,h])

blank = np.zeros((h,w,1), np.uint8)

with open(label_name, 'r') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    label_id, *points = line.split()
    if label_id != '0':
        continue
    points = list(map(float, points))
    contour = ((np.array(points).reshape(-1,2))*sz).astype(np.int32)
    image = cv2.fillPoly(image, [contour], (0,0,255))
    contour_center = np.mean(contour, axis=0).astype(np.int32)
    cv2.putText(image, label_id, tuple(contour_center), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

cv2.imwrite(f'{fname}-filled.jpg', image)