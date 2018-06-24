import argparse
from collections import namedtuple
import os
import io
import sys
import logging

import pandas as pd
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

from object_detection.utils import dataset_util


COMBINED_IMAGE_PATH = ''
ANNOTATION_FILE_PATH = ''
TF_RECORD_OUTPUT_FILE_PATH = ''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--combined_image_path', type=str, default=COMBINED_IMAGE_PATH)
    parser.add_argument('--annotation_file_path', type=str, default=ANNOTATION_FILE_PATH)
    parser.add_argument('--tf_record_output_file_path', type=str,
                        default=TF_RECORD_OUTPUT_FILE_PATH)
    return parser.parse_args()


def class_text_to_int(row_label):
    text_to_int_map = {'airplane': 1,
                       'apple': 2,
                       'car': 3,
                       'fish': 4,
                       'flower': 5}
    return text_to_int_map[row_label]


def create_tf_example(group, path):
    """Converting images and annotation to tfRecord"""
    image_path = os.path.join(path, '{}'.format(group.filename))
    with tf.gfile.GFile(image_path, 'rb') as fid:
        encoded_jpg = fid.read()

    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = image_path.encode('utf8')
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    difficult_obj = []
    truncated = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))
        difficult_obj.append(0)
        truncated.append(0)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


if __name__ == '__main__':
    # Parse arguments
    args = parse_args()

    drawing_df = pd.read_csv(args.annotation_file_path)
    grouped = split(drawing_df, 'filename')

    writer = tf.python_io.TFRecordWriter(args.tf_record_output_file_path)
    for group in grouped:
        tf_example = create_tf_example(group, args.combined_image_path)
        writer.write(tf_example.SerializeToString())
    writer.close()

    logging.info('Successfully created the TFRecords: {}'.format(args.tf_record_output_file_path))
