import numpy as np
import cv2
import cPickle
import os
import math
import tensorflow as tf
from .config import cfg
from ..utils.timer import Timer

def normalizer_blob(im, rgb_means):
    # Basically just subtracted the already cropped image with means.
    blobs_data = {'data' : None}
    norm = im.astype(np.float32, copy=True)
    offset = rgb_means.copy()
    # Swaping offset from RGB to BGR (to match cv2 channel order)
    offset[0], offset[2] = offset[2], offset[0]
    norm = norm - np.tile(offset, (norm.shape[0], im.shape[1], 1))
    # Generating a empty np array in blob
    img_shape = norm.shape
    blob = np.zeros((1, img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    # Fitting normalized image into blob
    blob[0, 0:img_shape[0], 0:img_shape[1], :] = norm
    blobs_data['data'] = blob
    return blobs_data

def im_detect(sess, net, centers_ref, im, ratio, rgb_means, sigmoid_score=False):
    # Detecting process for each images.
    # Will do mean substracting before feeding into tensor graph
    blobs = normalizer_blob(im, rgb_means)
    feed_dict={net.data: blobs['data']}
    fetch_list = [net.get_output('score_cls'),
                   net.get_output('score_reg'),
                   net.get_output('prob_cls')]
    # score_cls_np: Score_heatmap(non-sigmoid), score_reg_np: Regression_heatmap, prob_cls_np: Score_heatmap(Sigmoid)
    score_cls_np, score_reg_np, prob_cls_np = sess.run(fetches=fetch_list, feed_dict=feed_dict)
    boxes, scores = _process_boxes_scores(prob_cls_np[0, :, :, :], score_reg_np[0, :, :, :], score_cls_np[0, :, :, :], centers_ref, sigmoid_score, ratio)
    return boxes, scores

def nms(boxes, score, overlapThresh=0.5):
    # Non-Maximum Suppression
    x1 = boxes[:, 0]  
    y1 = boxes[:, 1]  
    x2 = boxes[:, 2]  
    y2 = boxes[:, 3]  
    scores = score[:, 0]  
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    order = scores.argsort()[::-1]  
    keep = []  
    while order.size > 0: 
        i = order[0]  
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  
        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= overlapThresh)[0]  
        order = order[inds + 1]  
    # Output the result of kept index after NMS
    return keep

def _process_boxes_scores(prob_cls_np, score_reg_np, score_cls_np, centers, sigmoid_score, ratio):
    # Process both scores and regression, for details please refer:
    # https://github.com/peiyunh/tiny/blob/master/tiny_face_detector.m
    if cfg.TEST.PRUNING:
        if ratio <= 1.0:
            tids = list(range(4, 12))
        else :
            tids = list(range(4, 12)) + list(range(18, 25))
        ignoredTids = list(set(range(0,centers.shape[0]))-set(tids))
        prob_cls_np[:,:,ignoredTids] = 0.

    # Determine spatial location of face heatmap
    fy, fx, fc = np.where(prob_cls_np > cfg.TEST.CONFIDENCE_Thresh)
    # Pre-processing information for bounding-box regression
    Nt = centers.shape[0]
    tx = score_reg_np[:,:,0:Nt]
    ty = score_reg_np[:,:,Nt:2*Nt]
    tw = score_reg_np[:,:,2*Nt:3*Nt]
    th = score_reg_np[:,:,3*Nt:4*Nt]
    # Create empty array for bounding-boxes
    bbox = np.zeros([len(fc), 4])
    confidence = np.zeros([len(fc), 1])
    for ind, num in enumerate(fc):
        rcx = ((fx[ind]*8) - 1) + ((centers[num, 2]-centers[num, 0]+1)*tx[fy[ind], fx[ind], fc[ind]])
        rcy = ((fy[ind]*8) - 1) + ((centers[num, 3]-centers[num, 1]+1)*ty[fy[ind], fx[ind], fc[ind]])
        rcw = (centers[num, 2]-centers[num, 0]+1)*math.exp(tw[fy[ind], fx[ind], fc[ind]])
        rch = (centers[num, 3]-centers[num, 1]+1)*math.exp(th[fy[ind], fx[ind], fc[ind]])
        bbox[ind, :] = [rcx-rcw/2, rcy-rch/2, rcx+rcw/2, rcy+rch/2]
        # Decide if output score for PR-curve tool is sigmoid score or not (Default to non-sigmoid)
        if sigmoid_score:
            confidence[ind] = prob_cls_np[fy[ind], fx[ind], fc[ind]]
        else:
            confidence[ind] = score_cls_np[fy[ind], fx[ind], fc[ind]]
    # Fix the bounding box by ratio that match its corresponding image pyramid ratio
    bbox = bbox / float(ratio)
    return bbox, confidence

def test_net(sess, net, roidb_path, centers_ref, weights_filename, dir_path):
    """Test a tiny face detector network on an image database."""
    # Load the validation dataset
    with open(roidb_path, 'rb') as fid:
        roidb = cPickle.load(fid)
    fid.close()
    num_images = len(roidb)

    # Load mean value from cfg files
    rgb_means = cfg.RGB_MEANS
    rgb_variance = cfg.RGB_VARIANCE

    # Decide if output score for PR-curve tool is sigmoid score or not (Default to non-sigmoid)
    sigmoid_score = False

    # This part served as loading pre-trained parameters from specified .npy file
    '''
    pretrained_model = str(os.path.join(dir_path, 'data/pretrain_model/.npy')
    try:
        print ('Loading pretrained model '
            'weights from {:s}').format(pretrained_model)
        net.load(pretrained_model, sess, True)
    except:
        raise 'Check your pretrained model {:s}'.format(pretrained_model)
    '''

    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    # Iterate through whole validation dataset
    for i in xrange(num_images):
        im = cv2.imread(roidb[i]['image'])
        _t['im_detect'].tic()
        # Create empty arrays for detected bounding boxes
        final_boxes = np.zeros([0, 4])
        final_confidence = np.zeros([0, 1])
        # Iterate through different image pyramid ratio, determined by config files
        for _, ratio in enumerate(cfg.TEST.RATIO_RANGE):
            resize_h = im.shape[0]*ratio
            resize_w = im.shape[1]*ratio
            resize_im = cv2.resize(im, (int(resize_w), int(resize_h)), interpolation = cv2.INTER_CUBIC)
            # im_detect handles function that feed image into tensor graph
            boxes, scores = im_detect(sess, net, centers_ref, resize_im, ratio, rgb_means, sigmoid_score=sigmoid_score)
            # Concate detected result into arrays for detected bounding boxes
            if (len(boxes)>0):
                final_boxes = np.concatenate((final_boxes, boxes), axis=0)
                final_confidence = np.concatenate((final_confidence, scores), axis=0)
        detect_time = _t['im_detect'].toc(average=False)

        _t['misc'].tic()
        # Applying Non-Maximum Suppression
        if (len(final_boxes)>0):
            refind_idx = nms(final_boxes, final_confidence, overlapThresh=cfg.TEST.NMS_Thresh)
            res = final_boxes[refind_idx, :]
            conf = final_confidence[refind_idx, :]
            res = res.astype(np.int32)
            conf = np.reshape(conf, [-1])
        else:
            res = final_boxes
            conf = final_confidence
        nms_time = _t['misc'].toc(average=False)

        # Create txt files for WIDER dataset official PR-curve generation tool, also used for our validation.
        if (cfg.TEST.GEN_PR_CURVE_TXT):
            img_abspath = roidb[i]['image']
            plain_file_name = os.path.splitext(os.path.basename(img_abspath))[0]
            file_dir = os.path.basename(os.path.dirname(img_abspath))
            # Default output all result into "pred" directory
            out_txt_dir = os.path.join(dir_path, 'pred', file_dir)
            if not os.path.exists(out_txt_dir):
                os.makedirs(out_txt_dir)
            out_txt = os.path.join(out_txt_dir, plain_file_name+'.txt')
            with open(out_txt, "w") as text_file:
                text_file.write('{0}\n'.format(plain_file_name))
                text_file.write('{:d}\n'.format(len(res)))
                for ind, ax in enumerate(res):
                    (x1, y1, x2, y2) = ax
                    text_file.write('{:d} {:d} {:d} {:d} {:0.3f}\n'.format(int(x1), int(y1), int(x2-x1+1), int(y2-y1+1), float(conf[ind])))
            text_file.close()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, detect_time, nms_time)
