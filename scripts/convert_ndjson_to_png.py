import argparse
import logging
import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import multiprocessing

from code.tools import save_image
matplotlib.use('Agg')
logging.basicConfig(level=logging.INFO)

OBJECT_LIST = ['airplane', 'apple', 'car', 'clock', 'fish', 'flower', 'square']
IMAGE_BASE_DIR = ''
OBJECT_LIMIT = 1000
N_PROCESSES = 1
DRAWING_LIST_SHARED = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_limit', type=int, default=OBJECT_LIMIT)
    parser.add_argument('--n_processes', type=str, default=N_PROCESSES)
    parser.add_argument('--image_base_dir', type=str, default=IMAGE_BASE_DIR)
    return parser.parse_args()


def save_image_single_process(arg):
    arg = arg.split('|')
    image_id, object_name = int(arg[0]), arg[1]
    save_image(DRAWING_LIST_SHARED[image_id],
               "{}.png".format(image_id),
               os.path.join(IMAGE_BASE_DIR, object_name))


if __name__ == '__main__':
    args = parse_args()

    # Create subfolders if not exist
    for object_name in OBJECT_LIST:
        subfolder_name = os.path.join(args.image_base_dir, object_name)
        if not os.path.exists(subfolder_name):
            os.makedirs(subfolder_name)

    for object_name in OBJECT_LIST:

        logging.info('Working on object class {}.'.format(object_name))

        f_name = os.path.join(args.image_base_dir, object_name) + '.ndjson'

        # Read ndjson file
        with open(f_name, 'r') as f:
            raw_file = f.read()
        drawing_json = raw_file.split('\n')

        # Convert ndjson into a list
        DRAWING_LIST_SHARED = []
        for d in drawing_json[:args.object_limit]:
            try:
                DRAWING_LIST_SHARED.append(json.loads(d)['drawing'])
            except:
                pass

        pool = multiprocessing.Pool(processes=int(args.n_processes))
        arg_list = ['{}|{}'.format(i, object_name) for i in range(len(DRAWING_LIST_SHARED))]
        result_list = pool.map(save_image_single_process, arg_list)
        pool.close()
        pool.join()
