import numpy as np
cimport numpy as np
cimport cython
import random as rnd
import cv2
import os
from cpython cimport bool
from ..tiny.config import cfg
from sklearn.utils import shuffle

DTYPE = np.int32
ctypedef np.int32_t DTYPE_t
DLTYPE = np.int64
ctypedef np.int64_t DLTYPE_t
FTYPE = np.float32
ctypedef np.float32_t FTYPE_t
@cython.cdivision(True)
@cython.boundscheck(False)

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline float float_max(float a, float b): return a if a >= b else b
cdef inline float float_min(float a, float b): return a if a <= b else b

def iou_ratio(np.ndarray[FTYPE_t, ndim=1] bbox, float crop_hf, float crop_wf):
    # Function for calculating iou between grount-truth bounding boxes and cropped region, will eliminate those with small value.
    cdef float x1, y1, x2, y2, iou
    x1 = float_max(0.0, bbox[0])
    y1 = float_max(0.0, bbox[1])
    x2 = float_min(bbox[2], crop_hf - 1.0)
    y2 = float_min(bbox[3], crop_wf - 1.0)
    # The following try-except is used for avoiding bad bounding annotation issue.
    try:
        iou = ((x2-x1)*(y2-y1)) / ((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
    except ZeroDivisionError:
        print('Zero division error occured!')
        iou = 0.0
    return iou

def compute_iou2d_densemap(np.ndarray[FTYPE_t, ndim=2] centers, np.ndarray[FTYPE_t, ndim=1] fst, np.ndarray[FTYPE_t, ndim=2] new_gt):
    # A numpy implementation for heatmap generation, please refer to:
    # https://github.com/peiyunh/tiny/blob/master/cnn_get_batch_hardmine.m#L378
    # https://github.com/peiyunh/tiny/blob/master/utils/compute_dense_overlap.cc
    cdef int fst_len = len(fst)
    cdef int cen_len = len(centers)
    cdef int ngt_len = len(new_gt)
    cdef np.ndarray[DTYPE_t, ndim=3] argout
    cdef np.ndarray[FTYPE_t, ndim=3] maxout
    cdef np.ndarray[DTYPE_t, ndim=1] fbest_idx
    cdef np.ndarray[FTYPE_t, ndim=1] iou_
    cdef np.ndarray[FTYPE_t, ndim=2] reshape

    cdef np.ndarray[FTYPE_t, ndim=3] center_box_small = np.stack((
                 np.tile(centers[:, 0], (fst_len, 1))+np.repeat(fst[:, np.newaxis], cen_len, axis=1),
                 np.tile(centers[:, 1], (fst_len, 1))+np.repeat(fst[:, np.newaxis], cen_len, axis=1),
                 np.tile(centers[:, 2], (fst_len, 1))+np.repeat(fst[:, np.newaxis], cen_len, axis=1),
                 np.tile(centers[:, 3], (fst_len, 1))+np.repeat(fst[:, np.newaxis], cen_len, axis=1)),axis=2)

    cdef np.ndarray[FTYPE_t, ndim=3] inter_w = np.minimum(np.repeat(center_box_small[:, :, 2, np.newaxis], len(new_gt), axis=2), np.tile(new_gt[:, 2], (len(fst), len(centers), 1)))\
              -np.maximum(np.repeat(center_box_small[:, :, 0, np.newaxis], len(new_gt), axis=2), np.tile(new_gt[:, 0], (len(fst), len(centers), 1)))
    cdef np.ndarray[FTYPE_t, ndim=3] inter_h = np.minimum(np.repeat(center_box_small[:, :, 3, np.newaxis], len(new_gt), axis=2), np.tile(new_gt[:, 3], (len(fst), len(centers), 1)))\
              -np.maximum(np.repeat(center_box_small[:, :, 1, np.newaxis], len(new_gt), axis=2), np.tile(new_gt[:, 1], (len(fst), len(centers), 1)))
    cdef np.ndarray[FTYPE_t, ndim=4] inter_area = np.repeat(inter_h[:, np.newaxis, :, :], len(fst), axis=1).clip(min=0)*np.tile(inter_w, (len(fst),1, 1, 1)).clip(min=0)
    cdef np.ndarray[FTYPE_t, ndim=3] cenbox = np.tile((centers[:, 2]-centers[:, 0]+1)*(centers[:, 3]-centers[:, 1]+1), (len(fst), len(fst), 1))
    cdef np.ndarray[FTYPE_t, ndim=4] outer_area = np.repeat(cenbox[:, :, :, np.newaxis], len(new_gt), axis=3)+\
                          np.tile((new_gt[:, 2]-new_gt[:, 0]+1)*(new_gt[:, 3]-new_gt[:, 1]+1), (len(fst), len(fst), len(centers), 1))
    cdef np.ndarray[FTYPE_t, ndim=4] iou = inter_area/(outer_area-inter_area)
    argout = np.argmax(iou, axis=3).astype(DTYPE)
    maxout = np.amax(iou, axis=3).astype(FTYPE)

    reshape = np.reshape(iou, [-1, ngt_len])
    fbest_idx = np.argmax(reshape, axis=0).astype(DTYPE)
    iou_ = np.amax(reshape, axis=0).astype(FTYPE)
    return argout, maxout, fbest_idx, iou_

def featureMap(np.ndarray[FTYPE_t, ndim=2] centers, int crop_h, int crop_w, np.ndarray[FTYPE_t, ndim=2] new_gt, int crop_size, 
               int var_size, float neg_iou_thresh, float pos_iou_thresh):
    # Will generate class and regression heatmap for training, please refer to:
    # https://github.com/peiyunh/tiny/blob/master/cnn_get_batch_hardmine.m
    cdef np.ndarray[FTYPE_t, ndim=1] fst = np.arange(-1, crop_size, var_size, dtype=FTYPE)
    cdef int cen_len = len(centers)
    cdef int ngt_len = len(new_gt)
    cdef int ind
    cdef np.ndarray[FTYPE_t, ndim=1] stuff

    fst = np.delete(fst, -1, 0)

    cdef int fst_len = len(fst)
    cdef np.ndarray[FTYPE_t, ndim=3] regmap = np.zeros([fst_len, fst_len, cen_len*4], dtype=FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=3] clsmap = -np.ones([fst_len, fst_len, cen_len], dtype=FTYPE)

    cdef np.ndarray[FTYPE_t, ndim=1] dww, dhh, fcx, fcy, fww, fhh, clsmap_div_vec
    cdef np.ndarray[FTYPE_t, ndim=3] max_iou, fbest_clsmap_r
    cdef np.ndarray[DTYPE_t, ndim=3] arg_iou
    cdef np.ndarray[DTYPE_t, ndim=2] entry_map
    cdef np.ndarray[FTYPE_t, ndim=2] tx, ty, tw, th, label_cls

    cdef np.ndarray[DTYPE_t, ndim=1] fbest_idx, search
    cdef np.ndarray[FTYPE_t, ndim=1] iou_
    cdef np.ndarray[FTYPE_t, ndim=1] fbest_clsmap = -np.ones([fst_len*fst_len*cen_len], dtype=FTYPE)

    if ngt_len > 0:
        dww = centers[:, 2]-centers[:, 0]+1
        dhh = centers[:, 3]-centers[:, 1]+1
        fcx = (new_gt[:, 0]+new_gt[:, 2])/2
        fcy = (new_gt[:, 1]+new_gt[:, 3])/2
        fww = new_gt[:, 2]-new_gt[:, 0]+1
        fhh = new_gt[:, 3]-new_gt[:, 1]+1
        
        arg_iou, max_iou, fbest_idx, iou_  = compute_iou2d_densemap(centers, fst, new_gt)
        search = fbest_idx[np.where(iou_ >= neg_iou_thresh)]
        fbest_clsmap[search] = 1
        fbest_clsmap_r = np.reshape(fbest_clsmap, [len(fst), len(fst), len(centers)])

        for ind, stuff in enumerate(centers):
            entry_map = arg_iou[:, :, ind]
            tx = (fcx[entry_map]-np.tile(fst, (fst_len, 1)))/dww[ind]
            ty = (fcy[entry_map]-np.repeat(fst[:, np.newaxis], fst_len, axis=1))/dhh[ind]
            tw = np.log((fww[entry_map])/dww[ind])
            th = np.log((fhh[entry_map])/dhh[ind])
            regmap[:, :, 0+ind] = tx
            regmap[:, :, (1*cen_len)+ind] = ty
            regmap[:, :, (2*cen_len)+ind] = tw
            regmap[:, :, (3*cen_len)+ind] = th
            label_cls = max_iou[:, :, ind].copy()
            label_cls[label_cls>=pos_iou_thresh] = 1.0
            #label_cls[label_cls<pos_iou_thresh] = 0
            label_cls[label_cls<=neg_iou_thresh] = -1.0
            label_cls[np.logical_and(label_cls<pos_iou_thresh, label_cls>neg_iou_thresh)] = 0            
            clsmap[:, :, ind] = label_cls
            if (crop_h < crop_size):
                clsmap[(crop_h/var_size):-1, :, :] = 0
                regmap[(crop_h/var_size):-1, :, :] = 0
            elif (crop_w < crop_size):
                clsmap[:, (crop_w/var_size):-1, :] = 0
                regmap[:, (crop_w/var_size):-1, :] = 0
        clsmap = np.maximum(clsmap, fbest_clsmap_r)
    return clsmap, regmap

def generate_new_bbox(np.ndarray[FTYPE_t, ndim=3] rnd_crop, np.ndarray[FTYPE_t, ndim=2] gt_bbox, int w_bias, int h_bias, float neg_iou_thresh):
    # This part efficiently keep the nice ground-truth bounding boxes
    cdef np.ndarray[FTYPE_t, ndim=2] new_gt = gt_bbox.copy()
    cdef np.ndarray[FTYPE_t, ndim=1] ax
    cdef int crop_h = rnd_crop.shape[0]
    cdef int crop_w = rnd_crop.shape[1]
    cdef int i
    cdef float crop_hf = float(crop_h)
    cdef float crop_wf = float(crop_w)
    new_gt[:,0::2] -= w_bias
    new_gt[:,1::2] -= h_bias
    index = []
    for i, ax in enumerate(new_gt):
        if ((ax[0] <= 0) and (ax[2] <= 0)): index.append(i)
        elif ((ax[1] <= 0) and (ax[3] <= 0)): index.append(i)
        elif ((ax[0] >= crop_wf) and (ax[2] >= crop_wf)): index.append(i)
        elif ((ax[1] >= crop_hf) and (ax[3] >= crop_hf)): index.append(i)
        elif (iou_ratio(ax, crop_hf, crop_wf) < neg_iou_thresh): index.append(i)
        else:
            ax[0], ax[1] = float_max(0, ax[0]), float_max(0, ax[1])
            ax[2], ax[3] = float_min(ax[2], crop_w), float_min(ax[3], crop_h)
    new_gt = np.delete(new_gt, index, axis=0)
    return new_gt

def random_crop(str img_path, np.ndarray[FTYPE_t, ndim=2] gt_bbox, bool scaling, bool flipped, int crop_size, float neg_iou_thresh):
    # Will randomly crop (Default to 500x500), and output corresponding image region and ground-truth bounding box within.
    cdef np.ndarray[FTYPE_t, ndim=3] rnd_crop
    cdef np.ndarray[FTYPE_t, ndim=2] new_gt
    cdef int cx1, cx2, cy1, cy2
    cdef float ratio
    rnd_crop = np.asarray(cv2.imread(img_path), dtype=FTYPE)

    if scaling:
        #ratio = rnd.choice([0.5, 1, 2])
        ratio = rnd.choice(cfg.TRAIN.SCALING_FACTER)
        rnd_crop = cv2.resize(rnd_crop, None, fx = ratio, fy = ratio, 
                                interpolation = cv2.INTER_LINEAR)
        gt_bbox = gt_bbox * ratio

    cdef int crop_h = rnd_crop.shape[0]
    cdef int crop_w = rnd_crop.shape[1]
    cx1 = rnd.randint(1, int_max(1, (crop_w-crop_size))) - 1
    cy1 = rnd.randint(1, int_max(1, (crop_h-crop_size))) - 1
    cx2 = int_min(crop_w, (cx1 + crop_size))
    cy2 = int_min(crop_h, (cy1 + crop_size))
    rnd_crop = rnd_crop[cy1:cy2, cx1:cx2, :]
    new_gt = generate_new_bbox(rnd_crop, gt_bbox, cx1, cy1, neg_iou_thresh)

    cdef float crop_hf = float(rnd_crop.shape[0])
    cdef float crop_wf = float(rnd_crop.shape[1])
    if flipped:
        rnd_crop = cv2.flip(rnd_crop,1)
        new_gt[:,[0, 2]] = new_gt[:, [2, 0]]
        new_gt[:, 0] = crop_wf - new_gt[:, 0] - 1
        new_gt[:, 2] = crop_wf - new_gt[:, 2] - 1
    return rnd_crop, new_gt

def normalizer(np.ndarray[FTYPE_t, ndim=3] im, np.ndarray[FTYPE_t, ndim=1] rgb_means, np.ndarray[FTYPE_t, ndim=2] rgb_variance):
    # Basically just subtracted the already cropped image with means.
    cdef int norm_h = im.shape[0]
    cdef int norm_w = im.shape[1]
    cdef np.ndarray[FTYPE_t, ndim=1] randomize = np.random.normal(0, 1, 3).astype(FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=3] norm = im.astype(np.float32, copy=True)
    cdef np.ndarray[FTYPE_t, ndim=1] offset = rgb_means + np.dot(rgb_variance, randomize)
    # Swaping offset from RGB to BGR (to match cv2 channel order)
    offset[0], offset[2] = offset[2], offset[0]
    norm = norm - np.tile(offset, (norm_h, norm_w, 1))
    return norm

def get_minibatch(roidb, np.ndarray[FTYPE_t, ndim=2] centers):
    cdef bool flipped = cfg.TRAIN.FLIPPED
    cdef bool scaling = cfg.TRAIN.SCALING
    cdef int crop_size = cfg.TRAIN.CROP_SIZE
    cdef int var_size = cfg.TRAIN.VAR_SIZE
    cdef bool normalize = cfg.TRAIN.NORMALIZE
    cdef float neg_iou_thresh = cfg.TRAIN.NEG_IOU_Thresh
    cdef float pos_iou_thresh = cfg.TRAIN.POS_IOU_Thresh
    cdef np.ndarray[FTYPE_t, ndim=1] rgb_means = cfg.RGB_MEANS
    cdef np.ndarray[FTYPE_t, ndim=2] rgb_variance = cfg.RGB_VARIANCE

    cdef np.ndarray[FTYPE_t, ndim=2] gt_bbox
    cdef np.ndarray[FTYPE_t, ndim=2] new_gt
    cdef np.ndarray[FTYPE_t, ndim=3] rnd_crop, clsmap, regmap

    cdef int crop_h, crop_w
    cdef int batch_size = len(roidb)
    cdef int mapsize = crop_size/var_size
    cdef int clustersize = len(centers)
    cdef int count
    cdef int new_gt_count = 0

    cdef np.ndarray[FTYPE_t, ndim=4] processed_data = np.zeros([batch_size, crop_size, crop_size, 3], dtype=FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=4] processed_clsmap = np.zeros([batch_size, mapsize, mapsize, clustersize], dtype=FTYPE)
    cdef np.ndarray[FTYPE_t, ndim=4] processed_regmap = np.zeros([batch_size, mapsize, mapsize, clustersize*4], dtype=FTYPE)

    for count in range(batch_size):
        new_gt_count = 0
        img_path = roidb[count]['image']
        gt_bbox = roidb[count]['boxes'].astype(FTYPE)

        rnd_crop, new_gt = random_crop(img_path, gt_bbox, scaling, flipped, crop_size, neg_iou_thresh)
        crop_h = rnd_crop.shape[0]
        crop_w = rnd_crop.shape[1]
        clsmap, regmap = featureMap(centers, crop_h, crop_w, new_gt, crop_size, var_size, neg_iou_thresh, pos_iou_thresh)

        rnd_crop = np.stack((np.pad(rnd_crop[:, :, 0], ((0, crop_size-crop_h), (0, crop_size-crop_w)), mode='constant', constant_values=rgb_means[2]), \
                             np.pad(rnd_crop[:, :, 1], ((0, crop_size-crop_h), (0, crop_size-crop_w)), mode='constant', constant_values=rgb_means[1]), \
                             np.pad(rnd_crop[:, :, 2], ((0, crop_size-crop_h), (0, crop_size-crop_w)), mode='constant', constant_values=rgb_means[0])), axis=2)

        data = rnd_crop.astype(np.float32, copy=True)
        if (normalize):
            data = normalizer(data, rgb_means, rgb_variance)

        # These will be feed into placeholder of tensorflow architecture
        processed_data[count, :, :, :] = data
        processed_clsmap[count, :, :, :] = clsmap
        processed_regmap[count, :, :, :] = regmap

    blobs = {'data': processed_data}
    blobs['clsmap'] = processed_clsmap
    blobs['regmap'] = processed_regmap
    # The following only served for image viusalization in tensorboard
    blobs['gen_bbox'] = new_gt
    blobs['im_name'] = os.path.basename(img_path)
    blobs['orig_img'] = rnd_crop
    return blobs
