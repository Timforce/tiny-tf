
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# Mod from https://github.com/CharlesShang/TFFRCNN/blob/master/lib/fast_rcnn/train.py

"""Train a tiny face detection network based on WIDER Face dataset."""

import argparse
import pprint
import numpy as np
import sys
import os.path
import scipy.io

this_dir = os.path.dirname(__file__)
sys.path.insert(0, this_dir)

from lib.tiny.train import get_training_roidb, train_net
from lib.tiny.config import cfg, cfg_from_file
from lib.networks.factory import get_network
from time import strftime, localtime

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a tiny face detection network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--pkl_file', dest='pkl_name',
                        help='target roidb pickles',
                        default='data/pickles/wider_train_roidb_detail.pkl', type=str)
    parser.add_argument('--refBox', dest='refBox',
                        help='Reference Bounding box file',
                        default='data/RefBox_N25_scaled.mat', type=str)
    parser.add_argument('--output_dir', dest='output_dir',
                        help='specify the name for output directory',
                        default='output', type=str)
    parser.add_argument('--log_dir', dest='log_dir',
                        help='specify the name for log directory',
                        default='log', type=str)
    parser.add_argument('--restore', dest='restore',
                        help='restore or not',
                        default=0, type=int)
    parser.add_argument('--restore_dir', dest='restore_dir',
                        help='restore_directory',
                        default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def get_output_dir(output_dir, network_name):
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), output_dir))
    time = strftime("%m-%d-%H-%M", localtime())
    if network_name is not None:
        n_name = network_name.split('_')[0]
        outdir = os.path.join(outdir, (n_name + '_' + time))
    else:
        outdir = os.path.join(outdir, time)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir

def get_log_dir(log_dir):
    logdir = os.path.abspath(os.path.join(os.path.dirname(__file__), log_dir))
    logdir = os.path.join(logdir, strftime("%Y-%m-%d-%H-%M-%S", localtime()))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    print('Using config:')
    pprint.pprint(cfg)

    print 'Loaded pickles `{:s}` for training'.format(os.path.basename(args.pkl_name))
    roidb = get_training_roidb(args.pkl_name)

    if ((args.restore) and (args.restore_dir is not None)):
        output_dir = os.path.abspath(args.restore_dir)
    else:
        output_dir = get_output_dir(args.output_dir, args.network_name)

    log_dir = get_log_dir(args.log_dir)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    print 'Logs will be saved to `{:s}`'.format(log_dir)

    device_name = '/gpu:{:d}'.format(args.gpu_id)
    print device_name

    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)
    
    refBox_file = os.path.abspath(args.refBox)
    centers_read = scipy.io.loadmat(refBox_file)['clusters']
    centers_read = centers_read.astype(np.float32)
    print 'Use reference box file `{:s}` in training'.format(os.path.basename(refBox_file))
    print 'Load done!'

    train_net(network, roidb,
              output_dir=output_dir,
              log_dir=log_dir,
              ref_Box=centers_read,
              pretrained_model=args.pretrained_model,
              epochs=args.max_epochs,
              restore=bool(int(args.restore)))
