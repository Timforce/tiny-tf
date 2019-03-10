# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# Configuration for Finding tiny faces, mod from https://github.com/CharlesShang/TFFRCNN/blob/master/lib/fast_rcnn/config.py
import os
import os.path as osp
import numpy as np
from time import strftime, localtime
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

#
# Training options
#
__C.TRAIN = edict()
__C.TRAIN.BATCH_SIZE = 4
__C.TRAIN.FLIPPED = True
__C.TRAIN.SCALING = True
__C.TRAIN.SCALING_FACTER = [0.5, 1, 2]
__C.TRAIN.POS_IOU_Thresh = 0.7
__C.TRAIN.NEG_IOU_Thresh = 0.3
__C.TRAIN.NMS_Thresh = 0.3
__C.TRAIN.CONFIDENCE_Thresh = 0.5
__C.TRAIN.CROP_SIZE = 500
__C.TRAIN.VAR_SIZE = 8
__C.TRAIN.NORMALIZE = True
__C.TRAIN.SOLVER = 'Momentum'
__C.TRAIN.WEIGHT_DECAY = 0.0005
__C.TRAIN.LEARNING_RATE = 0.0001
__C.TRAIN.MOMENTUM = 0.9
__C.TRAIN.GAMMA = 0.1
__C.TRAIN.STEPSIZE = 30000
__C.TRAIN.DISPLAY = 1
__C.TRAIN.LOG_IMAGE_ITERS = 1500
__C.TRAIN.SAMPLE_LIMIT = 128
__C.TRAIN.PRUNING = True
__C.TRAIN.SNAPSHOT_PREFIX = 'VGGnet_tiny'

#
# Common options
#
__C.RGB_MEANS = np.array([119.29960, 110.54627, 101.83843], dtype=np.float32)
__C.RGB_VARIANCE = np.array([[0.74215591, -1.3868568, 0.69416434],
                            [2.6491976, 0.088368624, -2.6558025],
                            [7.3054428, 7.6848936, 7.5429802]], dtype=np.float32)

#
# Testing options
#
__C.TEST = edict()
__C.TEST.CONFIDENCE_Thresh = 0.5
__C.TEST.NMS_Thresh = 0.3
__C.TEST.RATIO_RANGE = [0.5, 1, 2]
__C.TEST.PRUNING = True
__C.TEST.GEN_PR_CURVE_TXT = True

#
# Demo options
#
__C.DEMO = edict()
__C.DEMO.CONFIDENCE_Thresh = 0.5
__C.DEMO.NMS_Thresh = 0.1
__C.DEMO.MAX_INPUT_DIM = 5000
__C.DEMO.PRUNING = True
__C.DEMO.VISUALIZE = True
__C.DEMO.DRAW_SCORE_COLORBAR = True

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.iteritems():
        # a must specify keys that are in b
        if not b.has_key(k):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                'for config key: {}').format(type(b[k]),
                                                            type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)

def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert d.has_key(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert d.has_key(subkey)
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), \
            'type {} does not match original type {}'.format(
            type(value), type(d[subkey]))
        d[subkey] = value
