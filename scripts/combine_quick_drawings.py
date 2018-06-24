import argparse
import os
import itertools
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAX_OBJECT = 1
CANVAS_SIZE = (500, 500)
CANVAS_GRID_SIZE = (4, 4)
TOTAL_IMAGES = 100000

OBJECT_LIST = ['airplane', 'apple', 'car', 'fish', 'flower']
IMAGE_BASE_DIR = ''
OUTPUT_ANNOTATION_DIR = ''
OUTPUT_IMAGE_DIR = ''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--total_images', type=int, default=TOTAL_IMAGES)
    parser.add_argument('--image_base_dir', type=str, default=IMAGE_BASE_DIR)
    parser.add_argument('--output_annotation_dir', type=str, default=OUTPUT_ANNOTATION_DIR)
    parser.add_argument('--output_image_dir', type=str, default=OUTPUT_IMAGE_DIR)
    return parser.parse_args()


def add_drawings_to_canvas(canvas_name,
                           canvas_size,
                           canvas_grid_size,
                           object_list,
                           image_list_dict,
                           n_object_per_type=1):
    """Add single drawings to canvas"""
    canvas = np.zeros(canvas_size)

    # List of canvas grid to avoid overlaping positions
    canvas_grid = list(itertools.product(range(canvas_grid_size[0]),
                                         range(canvas_grid_size[1])))
    grid_w, grid_d = (canvas_size[0]) // canvas_grid_size[0], \
        (canvas_size[1]) // canvas_grid_size[1]

    # Allowable room to wiggle
    random_adjustment = grid_w // 3

    # Record annotation of object location
    annotation = []
    for object_name in object_list:
        sampled_image_names = np.random.choice(image_list_dict[object_name],
                                               n_object_per_type,
                                               replace=False)
        grid_locations_id = np.random.choice(range(len(canvas_grid)),
                                             len(sampled_image_names),
                                             replace=False)
        for i, image_file_name in enumerate(sampled_image_names):
            sampled_images = 255 - cv2.imread(image_file_name, 2)
            image_size_x, image_size_y = sampled_images.shape

            # Location of the left upper corner of the image
            grid_locations = canvas_grid[grid_locations_id[i]]
            random_cornor_x = grid_locations[0] * grid_w
            random_cornor_y = grid_locations[1] * grid_d

            # Random wiggle of the image
            if grid_locations[0] != (canvas_grid_size[0] - 1) and \
                    grid_locations[0] != 0:
                random_cornor_x += np.random.choice(range(-random_adjustment, random_adjustment))
            elif grid_locations[0] == (canvas_grid_size[0] - 1):
                random_cornor_x += np.random.choice(range(-random_adjustment, 1))
            else:
                random_cornor_x += np.random.choice(range(random_adjustment))

            if grid_locations[1] != (canvas_grid_size[1] - 1) and \
                    grid_locations[1] != 0:
                random_cornor_y += np.random.choice(range(-random_adjustment, random_adjustment))
            elif grid_locations[1] == (canvas_grid_size[1] - 1):
                random_cornor_y += np.random.choice(range(-random_adjustment, 1))
            else:
                random_cornor_y += np.random.choice(range(random_adjustment))

            # Add image to canvas
            canvas[random_cornor_x: random_cornor_x + image_size_x,
                   random_cornor_y: random_cornor_y + image_size_y] += sampled_images

            # Added annotation to output_data
            annotation.append(['{}.jpg'.format(canvas_name),
                               image_size_x,
                               image_size_y,
                               object_name,
                               random_cornor_y,
                               random_cornor_x,
                               random_cornor_y + image_size_y,
                               random_cornor_x + image_size_x])

        # Remove grid location to avoid overlapping position
        for l in grid_locations_id:
            canvas_grid.remove(canvas_grid[l])

    canvas[canvas > 0] = 255
    canvas = 255 - canvas
    return cv2.GaussianBlur(canvas, (5, 5), 0), annotation


if __name__ == '__main__':
    # Arguments
    args = parse_args()
    for d in [args.output_annotation_dir, args.output_image_dir]:
        if not os.path.exists(d):
            os.makedirs(d)

    # Read image directories
    image_list_dict = {}
    for object_name in OBJECT_LIST:
        image_list_dict[object_name] = [os.path.join(args.image_base_dir, object_name, f)
                                        for f in os.listdir(os.path.join(args.image_base_dir,
                                                                         object_name))]

    # Create images from single drawings and save
    file_count = 0
    total_annotation = []
    while file_count < args.total_images:
        canvas, annotation = add_drawings_to_canvas(file_count,
                                                    CANVAS_SIZE,
                                                    CANVAS_GRID_SIZE,
                                                    OBJECT_LIST,
                                                    image_list_dict)
        total_annotation += annotation
        cv2.imwrite(os.path.join(args.output_image_dir, str(file_count)) + '.jpg', canvas)
        file_count += 1
        if file_count % 100 == 0:
            logging.info("Created image {}.".format(file_count))

    # Save annotation
    COLUMN_NAMES = ['filename',
                    'width',
                    'height',
                    'class',
                    'xmin',
                    'ymin',
                    'xmax',
                    'ymax']
    pd.DataFrame(total_annotation, columns=COLUMN_NAMES).to_csv(
        os.path.join(args.output_annotation_dir, 'annotation.csv'), index=False)
