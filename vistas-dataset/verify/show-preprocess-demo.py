import os
import cv2
import random
import numpy as np
import yaml
from tqdm import tqdm
import json

original_path = '../vistas-original'
yolo_path = '../vistas-yolo'
removed_path = '../vistas-yolo-removed'
output_path = 'output'
images_path = os.path.join(original_path, 'training', 'images')
polygon_path = os.path.join(original_path, 'training', 'v2.0', 'polygons')

all_labels = os.listdir(os.path.join(yolo_path, 'labels', 'train'))


with open(os.path.join(original_path, 'config_v2.0.json'), 'r') as f:
    METADATA = json.load(f)
    LABELS = METADATA['labels']
    name_to_id = {label['name']: i for i, label in enumerate(LABELS)}
    id_to_name = {i: label['name'] for i, label in enumerate(LABELS)}
    num_classes = len(LABELS)


with open(os.path.join(yolo_path, 'data-vistas.yaml'), 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    sid_to_name = data['names']


with open(os.path.join(removed_path, 'data-vistas-removed.yaml'), 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    sid_to_name_removed = data['names']


def add_mask(image, contour, color=(0,255,0), alpha=0.4):
    overlay = image.copy()
    cv2.fillPoly(overlay, [contour], color)
    return cv2.addWeighted(overlay, alpha, image, 1-alpha, 0)

def add_text(image, contour, text, color=(0,0,255), size=1, thickness=1):
    blank = np.zeros((image.shape[0], image.shape[1], 1), np.uint8)
    cv2.fillPoly(blank, [contour], 1)
    M = cv2.moments(blank)
    contour_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, thickness)[0]
    text_loc = (contour_center[0] - text_size[0]//2, contour_center[1] + text_size[1]//2)
    return cv2.putText(image, text, tuple(text_loc), cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness)


for label_name in all_labels:
    fname = label_name.split('.')[0]
    image_path = os.path.join(images_path, f'{fname}.jpg')

    image = cv2.imread(image_path)
    cv2.imwrite(os.path.join(output_path, f'{fname}.jpg'), image)

    image_plot = image.copy()
    h,w = image.shape[:2]
    sz = np.array([w,h])

    blank = np.zeros((h,w,1), np.uint8)

    # first, plot the original one
    polygon_pth = os.path.join(polygon_path, f'{fname}.json')
    with open(polygon_pth, 'r') as f:
        data = json.load(f)
        objects = data['objects']
    for obj in objects:
        label = obj['label']
        label_id = name_to_id[label]
        polygon = np.array(obj['polygon']).astype(np.int32)
        image_plot = add_mask(image_plot, polygon)

        text = f'{label_id}: {label}'
        image_plot = add_text(image_plot, polygon, text)

    original_classes = [obj['label'] for obj in objects]
    original_contour = [np.array(obj['polygon']).astype(np.int32) for obj in objects]

    cv2.imwrite(os.path.join(output_path, f'{fname}-original.jpg'), image_plot)


    # second, plot the yolo and removed one
    def work(image, label_path, sid_to_name, removed=False):
        global sz, fname, blank
        image_plot = image.copy()
        # plot the original one
        with open(label_path, 'r') as f:
            lines = f.readlines()

        ids = []
        classes = []
        contours = []

        for line in lines:
            line = line.strip()
            label_id, *points = line.split()
            points = list(map(float, points))
            label_name = sid_to_name[int(label_id)]
            contour = ((np.array(points).reshape(-1,2))*sz).astype(np.int32)

            image_plot = add_mask(image_plot, contour)

            text = f'{label_id}: {label_name}'
            image_plot = add_text(image_plot, contour, text)

            ids.append(int(label_id))
            classes.append(label_name)
            contours.append(contour)

        output_pth = os.path.join(output_path, f'{fname}-yolo.jpg' if not removed else f'{fname}-removed.jpg')
        cv2.imwrite(output_pth, image_plot)
        return ids, classes, contours

    label_path = os.path.join(yolo_path, 'labels', 'train', label_name)
    r_label_path = os.path.join(removed_path, 'labels', 'train', label_name)
    yolo_ids, yolo_classes, yolo_contours = work(image, label_path, sid_to_name, removed=False)
    work(image, r_label_path, sid_to_name_removed, removed=True)

    assert original_classes == yolo_classes, f'{original_classes} != {yolo_classes}'
    assert len(original_contour) == len(yolo_contours), f'{len(original_contour)} != {len(yolo_contours)}'
    # for contours, check if the area is very close
    for oc, yc in zip(original_contour, yolo_contours):
        area_diff = abs(cv2.contourArea(oc) - cv2.contourArea(yc)) / cv2.contourArea(oc)
        assert abs(area_diff) < 1e-3, f'{area_diff}'