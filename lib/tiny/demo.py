import numpy as np
import cv2
import os
import math
import tensorflow as tf
import matplotlib.pyplot as plt
from .config import cfg
from ..utils.timer import Timer
from matplotlib import cm

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
    # Fitting normalized image to blob
    blob[0, 0:img_shape[0], 0:img_shape[1], :] = norm
    blobs_data['data'] = blob
    return blobs_data

def im_detect(sess, net, centers_ref, im, ratio, rgb_means):
    # Detecting process for each images.
    # Will do mean substracting before feeding into tensor graph
    blobs = normalizer_blob(im, rgb_means)
    feed_dict={net.data: blobs['data']}
    fetch_list = [net.get_output('score_cls'),
                   net.get_output('score_reg'),
                   net.get_output('prob_cls')]
    # score_cls_np: Score_heatmap(non-sigmoid), score_reg_np: Regression_heatmap, prob_cls_np: Score_heatmap(Sigmoid)
    score_cls_np, score_reg_np, prob_cls_np = sess.run(fetches=fetch_list, feed_dict=feed_dict)
    boxes, scores = _process_boxes_scores(prob_cls_np[0, :, :, :], score_reg_np[0, :, :, :], score_cls_np[0, :, :, :], centers_ref, ratio)
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

def _process_boxes_scores(prob_cls_np, score_reg_np, score_cls_np, centers, ratio):
    # Process both scores and regression, for details please refer:
    # https://github.com/peiyunh/tiny/blob/master/tiny_face_detector.m
    if cfg.DEMO.PRUNING:
        if ratio <= 1.0:
            tids = list(range(4, 12))
        else :
            tids = list(range(4, 12)) + list(range(18, 25))
        ignoredTids = list(set(range(0,centers.shape[0]))-set(tids))
        prob_cls_np[:,:,ignoredTids] = 0.

    # Determine spatial location of face heatmap
    fy, fx, fc = np.where(prob_cls_np > cfg.DEMO.CONFIDENCE_Thresh)
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
        confidence[ind] = prob_cls_np[fy[ind], fx[ind], fc[ind]]
    # Fix the bounding box by ratio that match its corresponding image pyramid ratio
    bbox = bbox / float(ratio)
    return bbox, confidence

def plot_score_colorbar(im, output_dest, dpi=80):
    height = im.shape[0]
    width = im.shape[1]
    fig, ax = plt.subplots(figsize=(width/float(dpi), height/float(dpi)), dpi=dpi)
    vis = ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    cbar = fig.colorbar(vis, ax=ax, 
                        ticks=np.linspace(0, 255, 5), 
                        pad=0.01, fraction=0.03, 
                        shrink=0.95, orientation='vertical')
    _ = cbar.ax.set_yticklabels(np.linspace(0.0, 1.0, 5), fontsize=int(1.5*height/dpi))
    _ = plt.tight_layout()
    _ = plt.axis('off')
    plt.savefig(os.path.splitext(output_dest)[0], dpi=dpi, bbox_inches='tight', transparent=True, pad_inches=0)

def vis_detections(im, boxes, conf, output_dest):
    # A visualization of detection
    output_img = im.copy()
    im_shape = im.shape
    # color = (0, 255, 255)
    cmap = cm.viridis
    for ind, box in enumerate(boxes):
        (x1, y1, x2, y2) = box
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, im_shape[1]), min(y2, im_shape[0])
        bw = x2 - x1 + 1
        bh = y2 - y1 + 1
        if (min(bw, bh) <= 20):
           lw = 1
        else:
           lw = int(max(2, min(3, min([bh/20, bw/20]))))
        color = tuple((math.ceil(ch*256)-1) for ch in cmap(conf[ind])[:3])
        color = (color[2], color[1], color[0])
        cv2.rectangle(output_img, (int(x1), int(y1)), (int(x2), int(y2)), color, lw)
    if cfg.DEMO.DRAW_SCORE_COLORBAR:
        plot_score_colorbar(output_img, output_dest)
    else:
        cv2.imwrite(output_dest, output_img)

def demo_net(sess, net, centers_ref, weights_filename, dir_path):
    # timers
    _t = {'im_detect' : Timer(), 'misc' : Timer()}

    # Load mean value from cfg files
    rgb_means = cfg.RGB_MEANS
    rgb_variance = cfg.RGB_VARIANCE

    # Will scan through demo/data directory, and output all result into demo/visualize 
    source_dir = os.path.join(dir_path, 'demo/data')
    output_dir = os.path.join(dir_path, 'demo/visualize')
    valid_images = [".jpg",".gif",".png"]

    # These parameters are used for auto-decide image pyramid ratio range
    clusters_h = centers_ref[:,3] - centers_ref[:,1] + 1
    clusters_w = centers_ref[:,2] - centers_ref[:,0] + 1
    normal_idx = np.where(centers_ref == 1)[0]

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

    det_count = 0
    for file_name in os.listdir(source_dir):
        ext = os.path.splitext(file_name)[1]
        if ext.lower() not in valid_images:
            continue
        im = cv2.imread(os.path.join(source_dir, file_name))

        # This part auto-decide image pyramid ratio range
        min_scale = min(np.floor(np.log2(max(clusters_w[normal_idx]/im.shape[1]))), \
                        np.floor(np.log2(max(clusters_h[normal_idx]/im.shape[0]))))
        max_scale = min(1, -np.log2(max(im.shape[0], im.shape[1])/cfg.DEMO.MAX_INPUT_DIM))
        scales = np.concatenate((np.arange(min_scale, 0), np.arange(0, max_scale+1e-4, 1)))
        scales = 2**scales

        _t['im_detect'].tic()
        final_boxes = np.zeros([0, 4])
        final_confidence = np.zeros([0, 1])
        for _, ratio in enumerate(scales):
            resize_h = im.shape[0]*ratio
            resize_w = im.shape[1]*ratio
            resize_im = cv2.resize(im, (int(resize_w), int(resize_h)), interpolation = cv2.INTER_CUBIC)
            # im_detect handles function that feed image into tensor graph
            boxes, scores = im_detect(sess, net, centers_ref, resize_im, ratio, rgb_means)
            # Concat detected result into arrays for detected bounding boxes
            if (len(boxes)>0):
                final_boxes = np.concatenate((final_boxes, boxes), axis=0)
                final_confidence = np.concatenate((final_confidence, scores), axis=0)
        detect_time = _t['im_detect'].toc(average=False)

        _t['misc'].tic()
        # Applying Non-Maximum Suppression
        if (len(final_boxes)>0):
            refind_idx = nms(final_boxes, final_confidence, overlapThresh=cfg.DEMO.NMS_Thresh)
            res = final_boxes[refind_idx, :]
            conf = final_confidence[refind_idx, :]
            res = res.astype(np.int32)
            conf = np.reshape(conf, [-1])
        else:
            res = final_boxes
            conf = final_confidence
        nms_time = _t['misc'].toc(average=False)

        if cfg.DEMO.VISUALIZE:
            output_dest = os.path.join(output_dir, file_name)
            im = cv2.imread(os.path.join(source_dir, file_name))
            vis_detections(im, res, conf, output_dest)

        det_count = det_count + 1
        print 'Im_detect: ({:d}) {:s}, {:.3f}s {:.3f}s' \
              .format(det_count, file_name, detect_time, nms_time)

    if cfg.DEMO.VISUALIZE:
        print 'Visualize result store at: {:s}'.format(output_dir)
