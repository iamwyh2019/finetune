import os
import cv2
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Colors
from ultralytics import YOLO
import asyncio
from concurrent.futures import ThreadPoolExecutor

from typing import List, Dict, Callable, Any, Union, Tuple
import torch
import numpy as np
import time

import traceback


model_path = os.path.join(os.path.dirname(__file__), 'pt', 'yolov8x-segp.pt')
model = YOLO(model_path)

CLASSES: Dict[int, str] = model.names
REV_CLASSES: Dict[str, int] = {name: i for (i, name) in CLASSES.items()}
COLORS = Colors()

executor = ThreadPoolExecutor(max_workers = 30)

def get_geometric_center(masks: torch.Tensor) -> np.ndarray:
    N, H, W = masks.shape

    x = torch.arange(W, device=masks.device).view(1, 1, W).expand(N, H, W)
    y = torch.arange(H, device=masks.device).view(1, H, 1).expand(N, H, W)

    x = (x * masks).sum(dim=(1, 2)) / masks.sum(dim=(1, 2))
    y = (y * masks).sum(dim=(1, 2)) / masks.sum(dim=(1, 2))

    return torch.stack([x, y], dim=1).to(torch.int32).cpu().numpy()

def parse_result(result: "Results", width, height, dwidth, dheight) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse the detection results."""
    # score threshold is already handled by the model
    # top_k is handled by the model

    # if no object is detected, return empty np arrays
    if len(result) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    # get the masks and cast to uint8
    masks_torch = result.masks.data.to(torch.uint8)
    # crop the masks to the image size
    # masks_torch = masks_torch[:, dheight//2-height//2:dheight//2+height//2, dwidth//2-width//2:dwidth//2+width//2]

    # get the geometric centers
    centers = get_geometric_center(masks_torch)

    # convert to numpy
    boxes: np.ndarray = result.boxes.xyxy.to(torch.int32).cpu().numpy()
    labels: np.ndarray = result.boxes.cls.to(torch.int32).cpu().numpy()
    scores: np.ndarray = result.boxes.conf.cpu().numpy()
    masks: np.ndarray = masks_torch.cpu().numpy()

    return masks, boxes, labels, scores, centers


def process_image(image:np.ndarray,
                  score_threshold: float = 0.3,
                  top_k: int = 15,
                  filter_classes: Union[List[int], None] = None) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray, np.ndarray, np.ndarray, List[List[int]]]:
    global model

    # compute the image size
    height, width, _ = image.shape
    # get the smallest multiple of 32 that is larger than the image size
    dwidth = width + 32 - width % 32
    dheight = height + 32 - height % 32

    results = model.predict(image, imgsz=(dheight, dwidth), conf=score_threshold,
                    max_det=top_k, device='cuda:0', verbose=False, half=True,
                    classes=filter_classes, retina_masks=True)
    result = results[0]

    masks, boxes, labels, scores, geometry_center = parse_result(result, width, height, dwidth, dheight)

    # get contours and geometry centers
    mask_contours = [None for _ in range(len(masks))]
    for i, mask in enumerate(masks):
        # crop the mask by box plus padding of 5 pixels
        x1, y1, x2, y2 = boxes[i]
        # x1 = max(0, x1 - 5)
        # y1 = max(0, y1 - 5)
        # x2 = min(image.shape[1], x2 + 5)
        # y2 = min(image.shape[0], y2 + 5)
        mask_crop = mask[y1:y2, x1:x2]

        if mask_crop.max() == 0:
            nonzero = np.nonzero(mask)
            if len(nonzero[0]) == 0 or len(nonzero[1]) == 0:
                print('no nonzero', i, CLASSES[labels[i]])
            x1 = min(nonzero[1])
            x2 = max(nonzero[1])
            y1 = min(nonzero[0])
            y2 = max(nonzero[0])
            mask_crop = mask[y1:y2, x1:x2]
            boxes[i] = [x1, y1, x2, y2]

        contours = bitmap_to_polygon(mask_crop)
        # find the largest contour
        contours.sort(key=lambda x: len(x), reverse=True)
        largest_contour = max(contours, key = cv2.contourArea)

        mask_contours[i] = largest_contour

    return masks, mask_contours, boxes, labels, scores, geometry_center


def get_recognition(image: np.ndarray,
                    filter_objects: List[str] = [],
                    score_threshold: float = 0.3,
                    top_k: int = 15) -> Dict[str, Any]:
    global CLASSES, COLORS

    if filter_objects:
        object_ids = [REV_CLASSES[object_name] for object_name in filter_objects if object_name in REV_CLASSES]

    masks, mask_contours, boxes, labels, scores, geometry_center = process_image(
        image,
        score_threshold,
        top_k,
        object_ids if filter_objects else None
    )

    mask_contours = [contour.tolist() for contour in mask_contours]
    geometry_center = geometry_center.tolist()
    boxes = boxes.tolist()
    scores = scores.tolist()
    labels = labels.tolist()

    # convert labels to class names
    class_names = [CLASSES[label] for label in labels]

    # get colors
    # each color is a 3-tuple (R, G, B)
    color_list = [COLORS(label, False) for label in labels]

    result = {
        "masks": masks,
        "mask_contours": mask_contours,
        "boxes": boxes,
        "scores": scores,
        "labels": labels, # "labels" is the original label, "class_names" is the class name
        "class_names": class_names,
        "geometry_center": geometry_center,
        "colors": color_list,
    }

    return result


async def async_get_recognition(image: np.ndarray,
                                filter_objects: List[str] = [],
                                score_threshold: float = 0.3,
                                top_k: int = 15) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, get_recognition, image, filter_objects, score_threshold, top_k)


def async_draw_recognition(image: np.ndarray, result: Dict[str, Any],
                     black: bool = False, draw_contour: bool = False, draw_mask: bool = True, 
                     draw_box: bool = False, draw_text: bool = True, draw_score = True,
                     draw_center = False, draw_tag = False, draw_arrow = False,
                     alpha: float = 0.45) -> np.ndarray:
    masks = result['masks']
    mask_contours = result['mask_contours']
    boxes = result['boxes']
    class_names = result['class_names']
    scores = result['scores']
    geometry_center = result['geometry_center']
    labels = result['labels']
    
    if black:
        image = np.zeros_like(image)

    if len(masks) == 0:
        return image

    # colors
    # each color is a 3-tuple (B, G, R)
    colors = []
    color_list = []
    for label in labels:
        color = COLORS(label, True)
        color = (color[2], color[1], color[0])
        color_list.append(color)
        colors.append(np.array(color, dtype=float).reshape(1,1,1,3))
    colors = np.concatenate(colors, axis=0)

    # yield to other tasks
    # await asyncio.sleep(0)
    
    if draw_mask:
        # masks N*H*W
        masks = np.array(masks, dtype=float)
        # change to N*H*W*1
        masks = np.expand_dims(masks, axis=3)

        masks_color = masks.repeat(3, axis=3) * colors * alpha

        inv_alpha_masks = masks * (-alpha) + 1

        masks_color_summand = masks_color[0]
        if len(masks_color) > 1:
            inv_alpha_cumul = inv_alpha_masks[:-1].cumprod(axis=0)
            masks_color_cumul = masks_color[1:] * inv_alpha_cumul
            masks_color_summand += masks_color_cumul.sum(axis=0)

        image = image * inv_alpha_masks.prod(axis=0) + masks_color_summand
        image = image.astype(np.uint8)

    # yield to other tasks
    # await asyncio.sleep(0)

    # draw the contours
    if draw_contour:
        for i, contour in enumerate(mask_contours):
            # contour is relative to the box, need to add the box's top-left corner
            x1, y1, _, _ = boxes[i]
            contour = np.array(contour) + np.array([x1, y1])
            contour = np.array(contour, dtype=np.int32)
            color = color_list[i]
            cv2.drawContours(image, [contour], -1, color, min(image.shape[:2])//100)

    # yield to other tasks
    # await asyncio.sleep(0)

    # draw box
    if draw_box:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            color = color_list[i]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    # yield to other tasks
    # await asyncio.sleep(0)

    # place text at the center
    if draw_text:
        for i, center in enumerate(geometry_center):
            text = class_names[i]
            if draw_score:
                text += f' {scores[i]:.2f}'
            cv2.putText(image, text, center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # draw center with a green circle, and center of bounding box with a yellow circle
    if draw_center:
        for i, center in enumerate(geometry_center):
            cv2.circle(image, tuple(center), 3, (0, 255, 0), -1)
            x1, y1, x2, y2 = boxes[i]
            center_box = ((x1+x2)//2, (y1+y2)//2)
            cv2.circle(image, center_box, 3, (0, 255, 255), -1)

    if draw_tag:
        for i, center in enumerate(geometry_center):
            # draw a white rectangle centered at the gemoetry center
            rec_tl_x = center[0] - 80
            rec_tl_y = center[1] - 30
            rec_br_x = center[0] + 80
            rec_br_y = center[1] + 30
            cv2.rectangle(image, (rec_tl_x, rec_tl_y), (rec_br_x, rec_br_y), (255, 255, 255), -1)
        
        for i, center in enumerate(geometry_center):
            text = class_names[i]
            rec_tl_x = center[0] - 80
            rec_tl_y = center[1] - 30
            rec_br_x = center[0] + 80
            rec_br_y = center[1] + 30
            if text == 'refrigerator':
                text = 'fridge'
            # draw the text
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontscale = 1.5
            fontthickness = 3
            textsize = cv2.getTextSize(text, font, fontscale, fontthickness)[0]
            
            textX = (rec_tl_x + rec_br_x - textsize[0]) // 2
            textY = (rec_tl_y + rec_br_y + textsize[1]) // 2
            cv2.putText(image, text, (textX, textY), font, fontscale, (0, 0, 0), fontthickness)

    if draw_arrow:
        for i, center in enumerate(geometry_center):
            x1, y1, x2, y2 = boxes[i]
            center_box = ((x1+x2)//2, (y1+y2)//2)
            cv2.arrowedLine(image, center, center_box, (0, 255, 255), 2)
    
    return image


def get_filtered_objects(result: Dict[str, Any], filter_objects: List[str]) -> Dict[str, Any]:
    """Filter the objects by class names."""
    assert 'class_names' in result

    fields = result.keys()
    new_result = {field: [] for field in fields}
    class_names = result['class_names']

    for i, class_name in enumerate(class_names):
        if class_name in filter_objects:
            for field in fields:
                new_result[field].append(result[field][i])

    return new_result


def bitmap_to_polygon(bitmap):
    """Convert masks from the form of bitmaps to polygons.

    Args:
        bitmap (ndarray): masks in bitmap representation.

    Return:
        list[ndarray]: the converted mask in polygon representation.
        bool: whether the mask has holes.
    """
    outs = cv2.findContours(bitmap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[-2]
    contours = [c.reshape(-1, 2) for c in contours]
    return contours


def main():
    filename = 'crosswalk.jpg'
    image = cv2.imread(filename)
    if image is None:
        print("Error: Image could not be read.")
        return

    results = get_recognition(image, score_threshold=0.5, top_k=70)
    
    image = async_draw_recognition(image, results, black=False, draw_contour=True, draw_mask=True, alpha=0.5, 
                                   draw_box=True, draw_text=True, draw_score=True, draw_center=True, draw_tag=True)

    output_filename = filename.split('.')[0] + '-demo.jpg'
    cv2.imwrite(output_filename, image)
    print(f"Processed image saved as {output_filename}")

if __name__ == '__main__':
    main()
