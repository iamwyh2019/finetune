import numpy as np
import ujson as json
import os
from typing import List, Dict
from tqdm import trange, tqdm
import cv2
from ultralytics.data.converter import merge_multi_segment

img_dir = './images'
label_dir = './labels'
vistas_dir = '../vistas-original'

metadata_path = os.path.join(vistas_dir, 'config_v2.0.json')
with open(metadata_path, 'r', encoding='utf-8') as f:
    METADATA = json.load(f)
    LABELS = METADATA['labels']
    name_to_id = {label['name']: i for i, label in enumerate(LABELS)}
    id_to_name = {i: label['name'] for i, label in enumerate(LABELS)}
    num_classes = len(LABELS)


def convert_from_json(mode: str) -> None:
    """
    Convert the json files specified by the mode ("train" or "val")
    """
    images = os.listdir(os.path.join(img_dir, mode))
    json_dir = os.path.join(vistas_dir, 'training' if mode == 'train' else 'validation', 'v2.0', 'polygons')
    panoptic_dir = os.path.join(vistas_dir, 'training' if mode == 'train' else 'validation', 'v2.0', 'panoptic')

    for image in tqdm(images):
        image_name = image.split('.')[0]
        with open(os.path.join(json_dir, image_name + '.json'), 'r', encoding='utf-8') as f:
            data = json.load(f)

            # create the output file, with the same name as the image plus ".txt"
            output_path = os.path.join(label_dir, mode, image_name + '.txt')

            # read in panoptic annotation png
            panoptic_path = os.path.join(panoptic_dir, image_name + '.png')
            panoptic = cv2.imread(panoptic_path)
            # ids == R + 256 * G + 256^2 * B
            # be careful with the order of the channels
            ids = panoptic[:, :, 2] + 256 * panoptic[:, :, 1] + 256**2 * panoptic[:, :, 0]

            with open(output_path, 'w', encoding='utf-8') as out:
                # get the height and width of the image to normalize the coordinates
                height = data['height']
                width = data['width']
                sz_np = np.array([width, height])
                # objects are in the 'objects' key, in a list format
                # each object has "id" (not used), "label" (name of the object), "polygon" (list of points, as a list of lists)
                objects = data['objects']

                for obj in objects:
                    # get the label name and convert it to the corresponding id
                    label = obj['label']
                    label_id = name_to_id[label]
                    # output one line for each object
                    # format: id x_0 y_0 x_1 y_1 ... x_n y_n
                    out.write(str(label_id))
                    polygon = obj['polygon']
                    for point in polygon:
                        out.write(' ' + str(point[0] / width) + ' ' + str(point[1] / height))
                    out.write('\n')

                # # parse the panoptic annotation
                # for i, obj in enumerate(objects):
                #     mask = (ids == i+1)
                #     label = obj['label']

                #     label_id = name_to_id[label]

                #     # get the contour of the mask
                #     contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                #     if len(contours) == 0:
                #         # fallback to polygon
                #         polygon = np.array(obj['polygon']).astype(np.float32)
                #         merged = (polygon / sz_np).reshape(-1).tolist()
                #     elif len(contours) > 1:
                #         # merge the contours into one
                #         merged = merge_multi_segment(contours)
                #         merged = (np.concatenate(merged, axis=0) / sz_np).reshape(-1).tolist()
                #     else:
                #         merged = (contours[0] / sz_np).reshape(-1).tolist()

                #     out.write(str(label_id))
                #     for point in merged:
                #         out.write(' ' + str(point))
                #     out.write('\n')


def to_yaml() -> None:
    """
    Convert the metadata to a yaml file
    """
    import yaml

    output = {'names': id_to_name,
              'path': '../vistas-yolo',
            'train': 'images/train',
            'val': 'images/val',}

    with open('data-vistas.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(output, f, allow_unicode=True)


def main() -> None:
    pass
    convert_from_json('train')
    convert_from_json('val')
    to_yaml()

if __name__ == '__main__':
    main()