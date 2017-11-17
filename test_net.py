#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# This version is mod from Fast R-CNN provided by https://github.com/CharlesShang/TFFRCNN
"""Test a tiny face detection network on an image database."""
import sys,os
this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir)
from lib.tiny.test import test_net
from lib.tiny.config import cfg, cfg_from_file
from lib.networks.factory import get_network
import scipy.io
import argparse
import pprint
import tensorflow as tf

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Testing pre-trained tiny face detection network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU id to use',
                        default=0, type=int)
    parser.add_argument('--weights', dest='model',
                        help='model to test',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file', default=None, type=str)
    parser.add_argument('--refBox', dest='refBox',
                        help='Reference Bounding box file',
                        default='data/RefBox_N25_scaled.mat', type=str)
    parser.add_argument('--pkl_file', dest='pkl_name',
                        help='target roidb pickles',
                        default='data/pickles/wider_val_detail.pkl', type=str)
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    weights_filename = os.path.splitext(os.path.basename(args.model))[0]
    checkpoint_dir = os.path.abspath(args.model)
    
    # Read the anchor information from .mat file
    refBox_file = os.path.abspath(args.refBox)
    centers_ref = scipy.io.loadmat(refBox_file)['clusters']
    print 'Use reference box file `{:s}`.'.format(os.path.basename(refBox_file))
    
    roidb_path = os.path.abspath(args.pkl_name)
    print 'Use pkl file `{:s}` as file income'.format(os.path.basename(roidb_path))

    device_name = '/gpu:{:d}'.format(args.gpu_id)
    print device_name

    network = get_network(args.network_name)
    print 'Use network `{:s}` for testing'.format(args.network_name)

    cfg.GPU_ID = args.gpu_id

    dir_path = os.path.abspath(os.path.dirname(__file__))

    # Load a trained ckpt file, NOT compatible with .npy file (The following paragraph)
    try:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver = tf.train.Saver()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
            print 'Success to load checkpoint from {}'.format(ckpt_name)
        else:
            print 'Failed to find a checkpoint!'
    except:
        print 'Error reading checkpoint file!'
        sys.exit(1)
    test_net(sess, network, roidb_path, centers_ref, weights_filename, dir_path)

    # The following is used for loading pre-trained parameters from .npy file, NOT compatible with ckpt file.
    '''
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        test_net(sess, network, roidb_path, centers_ref, weights_filename, dir_path)
    '''