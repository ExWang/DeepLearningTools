# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#coding=utf-8

"""Loads a sample video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import i3d

import random
import re
import os
import tempfile
import cv2
import json

# Some modules to display an animation using imageio.
# import imageio
# from IPython import display

# from urllib import request


_IMAGE_SIZE = 224

_SAMPLE_VIDEO_FRAMES = None
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH90 = '/home/user/kinetics-i3d/class_name.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'

PATH_JSON = '/home/user/kinetics-i3d/data/json_result_test2/'

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_video(path, resize=(224, 224)):
    framess = []
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)


        frames = np.array(frames)
        framess.append(frames)
        # print(len(framess))
        # print(len(frames))
    finally:
        cap.release()
    framess = np.array(framess) / 255.0
    # print(len(framess))
    return framess

def load_flow_video(path):
    flow_video = np.load(path)
    flow_video = np.divide(flow_video, 255.0)
    #flow_video = flow_video[:,:,:,:,[1,0]]
    return flow_video

# dirs = os.listdir('F:/smartcity/datasets_video/test_set2')


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    eval_type = FLAGS.eval_type

    imagenet_pretrained = FLAGS.imagenet_pretrained

    NUM_CLASSES = 400
    if eval_type == 'rgb600':
        NUM_CLASSES = 600

    if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
        raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

    if eval_type == 'rgb600':
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
    else:
        kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]
        smartcity_classes = [x.strip() for x in open(_LABEL_MAP_PATH90)]

    if eval_type in ['rgb', 'rgb600', 'joint']:
        # RGB input has 3 channels.
        rgb_input = tf.placeholder(
            tf.float32,
            shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 3))

        with tf.variable_scope('RGB'):
            rgb_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            rgb_logits, _ = rgb_model(
                rgb_input, is_training=False, dropout_keep_prob=1.0)

        rgb_variable_map = {}
        for variable in tf.global_variables():

            if variable.name.split('/')[0] == 'RGB':
                if eval_type == 'rgb600':
                    rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
                else:
                    rgb_variable_map[variable.name.replace(':0', '')] = variable

        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

    if eval_type in ['flow', 'joint']:
        # Flow input has only 2 channels.
        flow_input = tf.placeholder(
            tf.float32,
            shape=(1, _SAMPLE_VIDEO_FRAMES, _IMAGE_SIZE, _IMAGE_SIZE, 2))
        with tf.variable_scope('Flow'):
            flow_model = i3d.InceptionI3d(
                NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
            flow_logits, _ = flow_model(
                flow_input, is_training=False, dropout_keep_prob=1.0)
        flow_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'Flow':
                flow_variable_map[variable.name.replace(':0', '')] = variable
        flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

    if eval_type == 'rgb' or eval_type == 'rgb600':
        model_logits = rgb_logits
    elif eval_type == 'flow':
        model_logits = flow_logits
    else:
        model_logits = rgb_logits + flow_logits
    model_predictions = tf.nn.softmax(model_logits)

    with tf.Session() as sess:
        feed_dict = {}
        if eval_type in ['rgb', 'rgb600', 'joint']:
            if imagenet_pretrained:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
            else:
                rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
            tf.logging.info('RGB checkpoint restored')

        if eval_type in ['flow', 'joint']:
            if imagenet_pretrained:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
            else:
                flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
            # tf.logging.info('Flow checkpoint restored')
            # dirs = os.listdir('/home/user/D/smartcity/datasets_video/test_set')
            dirs = os.listdir('/media/user/WorkDisk/test_set_2K_3_converted_npy')
            video_num = len(dirs)

            for i, one in enumerate(dirs):
                exists_jsonFiles = map(lambda x:x.split('/')[-1], os.listdir('/home/user/kinetics-i3d/data/json_result_test2/'))

                if one.split('.')[0] in exists_jsonFiles:
                    print("{:s} already complete. ({:d} / {:d})".format(one, i + 1, video_num))
                    continue
                print("{:s} complete. ({:d} / {:d})".format(one, i + 1, video_num))
                Mp4_path = os.path.join("/media/user/WorkDisk/smartcity/datasets_video/test_set", one.split('.')[0]+'.mp4')
                Mp4_flow_path = os.path.join('/media/user/WorkDisk/test_set_2K_3_converted_npy', one)
                Numppy_rgb = load_video(Mp4_path)
                Numppy_flow = load_flow_video(Mp4_flow_path)

                if not len(Numppy_rgb.shape) == 5:
                    print ("Error, read rgb of {:s} failed ".format(one))
                if not len(Numppy_flow.shape) == 5:
                    print("Error, read flow of {:s} failed ".format(one))

                # print (str(type(Numppy)))
                # tf.logging.info('RGB data loaded, shape=%s', str(Numppy.shape))
                if eval_type in ['rgb', 'rgb600', 'joint']:
                    feed_dict[rgb_input] = Numppy_rgb
                feed_dict[flow_input] = Numppy_flow
                out_logits, out_predictions = sess.run(
                    [model_logits, model_predictions],
                    feed_dict=feed_dict)
                out_logits = out_logits[0]
                out_predictions = out_predictions[0]
                sorted_indices = np.argsort(out_predictions)[::-1]

                # print('Norm of logits: %f' % np.linalg.norm(out_logits))
                # print('\nTop classes and probabilities')

                possibilities = []
                poss_all = 0.0
                classes = []
                count = 0
                for index in sorted_indices:
                    predicted_class = kinetics_classes[index]
                    if predicted_class in smartcity_classes:
                        count += 1
                        # print(out_predictions[index], out_logits[index], kinetics_classes[index])
                        possibilities.append(out_predictions[index])
                        poss_all += float(out_predictions[index])
                        classes.append(kinetics_classes[index])
                        if count == 5:
                            break
                for i in range(5):
                    possibilities[i] /= (poss_all + 0.00001)

                with open(PATH_JSON + one.split('.')[0], 'w') as f:
                    jsonObj = []
                    for i in range(5):
                        jsonObj.append({'label': classes[i], 'score': possibilities[i]})
                    jsonObj = {one.split('.')[0]: jsonObj}
                    f.write(json.dumps(jsonObj))


if __name__ == '__main__':
    tf.app.run(main)
