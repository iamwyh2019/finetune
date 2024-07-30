"""
    This file shows how to load and use the dataset
"""

from __future__ import print_function
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image


parser = argparse.ArgumentParser(description="Mapillary Vistas demo")
parser.add_argument("--v1_2", action="store_true",
                    help="Demo version 1.2 of the dataset instead of 2.0")


def apply_color_map(image_array, labels):
    color_array = np.zeros((image_array.shape[0], image_array.shape[1], 3), dtype=np.uint8)

    for label_id, label in enumerate(labels):
        # set all pixels with the current label to the color of the current label
        color_array[image_array == label_id] = label["color"]

    return color_array

def main(args):
    version = "v1.2" if args.v1_2 else "v2.0"

    # a nice example
    image_id = '--jRsD9anHdGxv4ZOCQktA'

    # read in config file
    with open('config_{}.json'.format(version)) as config_file:
        config = json.load(config_file)
    # in this example we are only interested in the labels
    labels = config['labels']

    # print labels
    # print("There are {} labels in the config file".format(len(labels)))
    # for label_id, label in enumerate(labels):
    #     print("{:>30} ({:2d}): {:<40} has instances: {}".format(label["readable"], label_id, label["name"], label["instances"]))

    # set up paths for every image
    image_path = "training/images/{}.jpg".format(image_id)
    label_path = "training/{}/labels/{}.png".format(version, image_id)
    instance_path = "training/{}/instances/{}.png".format(version, image_id)
    panoptic_path = "training/{}/panoptic/{}.png".format(version, image_id)

    # load images
    base_image = Image.open(image_path)
    label_image = Image.open(label_path)
    instance_image = Image.open(instance_path)
    panoptic_image = Image.open(panoptic_path)

    # convert labeled data to numpy arrays for better handling
    label_array = np.array(label_image)
    instance_array = np.array(instance_image, dtype=np.uint16)

    # now we split the instance_array into labels and instance ids
    instance_label_array = np.array(instance_array / 256, dtype=np.uint8)
    instance_ids_array = np.array(instance_array % 256, dtype=np.uint8)

    # for visualization, we apply the colors stored in the config
    colored_label_array = apply_color_map(label_array, labels)
    colored_instance_label_array = apply_color_map(instance_label_array, labels)


    # plot the result
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20,15))

    ax[0][0].imshow(base_image)
    ax[0][0].get_xaxis().set_visible(False)
    ax[0][0].get_yaxis().set_visible(False)
    ax[0][0].set_title("Base image")
    ax[0][1].imshow(colored_label_array)
    ax[0][1].get_xaxis().set_visible(False)
    ax[0][1].get_yaxis().set_visible(False)
    ax[0][1].set_title("Labels")
    ax[1][0].imshow(instance_ids_array)
    ax[1][0].get_xaxis().set_visible(False)
    ax[1][0].get_yaxis().set_visible(False)
    ax[1][0].set_title("Instance IDs")
    ax[1][1].imshow(colored_instance_label_array)
    ax[1][1].get_xaxis().set_visible(False)
    ax[1][1].get_yaxis().set_visible(False)
    ax[1][1].set_title("Labels from instance file (identical to labels above)")
    ax[2][0].imshow(panoptic_image)
    ax[2][0].get_xaxis().set_visible(False)
    ax[2][0].get_yaxis().set_visible(False)
    ax[2][0].set_title("Labels from panoptic")
    ax[2][1].axis('off')

    fig.tight_layout()

    fig.savefig('MVD_plot.png')


    # PANOPTIC HANDLING

    # read in panoptic file
    if args.v1_2:
        panoptic_json_path = "training/v1.2/panoptic/panoptic_2018.json"
    else:
        panoptic_json_path = "training/v2.0/panoptic/panoptic_2020.json"
    with open(panoptic_json_path) as panoptic_file:
        panoptic = json.load(panoptic_file)

    # convert annotation infos to image_id indexed dictionary
    panoptic_per_image_id = {}
    for annotation in panoptic["annotations"]:
        panoptic_per_image_id[annotation["image_id"]] = annotation

    # convert category infos to category_id indexed dictionary
    panoptic_category_per_id = {}
    for category in panoptic["categories"]:
        panoptic_category_per_id[category["id"]] = category

    # convert segment infos to segment id indexed dictionary
    example_panoptic = panoptic_per_image_id[image_id]
    example_segments = {}
    for segment_info in example_panoptic["segments_info"]:
        example_segments[segment_info["id"]] = segment_info

    print('')
    print('Panoptic segments:')
    print('')
    panoptic_array = np.array(panoptic_image).astype(np.uint32)
    panoptic_id_array = panoptic_array[:,:,0] + (2**8)*panoptic_array[:,:,1] + (2**16)*panoptic_array[:,:,2]
    panoptic_ids_from_image = np.unique(panoptic_id_array)
    for panoptic_id in panoptic_ids_from_image:
        if panoptic_id == 0:
            # void image areas don't have segments
            continue
        segment_info = example_segments[panoptic_id]
        category = panoptic_category_per_id[segment_info["category_id"]]
        print("segment {:8d}: label {:<40}, area {:6d}, bbox {}".format(
            panoptic_id,
            category["supercategory"],
            segment_info["area"],
            segment_info["bbox"],
        ))

        # remove to show every segment is associated with an id and vice versa
        example_segments.pop(panoptic_id)

    # every positive id is associated with a segment
    # after removing every id occuring in the image, the segment dictionary is empty
    assert len(example_segments) == 0
        


if __name__ == "__main__":
    main(parser.parse_args())
