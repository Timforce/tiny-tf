# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------
# Mod from Faster R-CNN, https://github.com/CharlesShang/TFFRCNN/blob/master/lib/fast_rcnn/train.py
import numpy as np
import tensorflow as tf
import os
import cv2
import math
import cPickle

from ..roi_data_layer.layer import RoIDataLayer
from ..utils.timer import Timer
from ..tiny.config import cfg

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """

    def __init__(self, sess, network, roidb, output_dir,
                 centers, logdir, pretrained_model=None):
        """Initialize the SolverWrapper."""
        self.net = network
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model
        self.centers = centers

        # For checkpoint
        self.saver = tf.train.Saver(max_to_keep=100)
        self.writer = tf.summary.FileWriter(logdir=logdir,
                                            graph=tf.get_default_graph(),
                                            flush_secs=5)

    def snapshot(self, sess, iter, epoch):
        # This part handle the checkpoint (ckpt) generation of the network
        net = self.net

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + '_' +
                    '_epoch_{:d}'.format(epoch+1))
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename, global_step=iter)
        print 'Wrote snapshot to: {:s}'.format(filename)

    def build_image_summary(self):
        # Image visualization in tensorboard 'image', may not be accurate detection result!
        log_image_data = tf.placeholder(tf.uint8, [None, None, None, 3])
        log_image_name = tf.placeholder(tf.string)
        from tensorflow.python.ops import gen_logging_ops
        from tensorflow.python.framework import ops as _ops
        log_image = gen_logging_ops._image_summary(log_image_name, log_image_data, max_images=2)
        _ops.add_to_collection(_ops.GraphKeys.SUMMARIES, log_image)
        return log_image, log_image_data, log_image_name

    def train_model(self, sess, epochs, restore=False):
        """Network training loop."""

        # Pass to RoIDataLayer for dataset input and pre-processing
        data_layer = get_data_layer(self.roidb, self.centers, 1)

        # Refer to 'network.py' loss building function for training
        loss, det_loss, huber_loss = self.net.build_loss()

        # scalar summary
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('det_loss', det_loss)
        tf.summary.scalar('huber_loss', huber_loss)
        summary_op = tf.summary.merge_all()

        # image writer
        # NOTE: This only serve as a simple visualizer for detection result, not accurate enough though.
        log_image, log_image_data, log_image_name =\
            self.build_image_summary()

        # Define optimizer, only default to momentum optimizer. Adam and RMS required fixing before applying to training.
        if cfg.TRAIN.SOLVER == 'Adam':
            opt = tf.train.AdamOptimizer(cfg.TRAIN.LEARNING_RATE)
        elif cfg.TRAIN.SOLVER == 'RMS':
            opt = tf.train.RMSPropOptimizer(cfg.TRAIN.LEARNING_RATE)
        else:
            # lr is for training CNN network and score_res4, lr2 is for training CNN network and score_res3
            # lr and lr2 need to be different learning rate (important!)
            lr = tf.Variable(cfg.TRAIN.LEARNING_RATE, trainable=False)
            lr2 = tf.Variable(cfg.TRAIN.LEARNING_RATE * 0.1, trainable=False)
            momentum = cfg.TRAIN.MOMENTUM
            opt = tf.train.MomentumOptimizer(lr, momentum)
            opt2 = tf.train.MomentumOptimizer(lr2, momentum)

        global_step = tf.Variable(0, trainable=False)
        # We only defualt to non-clip, clip version need to be fixed before applying to training.
        with_clip = False
        if with_clip:
            tvars = tf.trainable_variables()
            grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10.0)
            train_op = opt.apply_gradients(zip(grads, tvars), global_step=global_step)
        else:
            tvars = tf.trainable_variables()
            # CNN network and score_res4, need to change to other parameter name if using different pre-trained network.
            opt_var1 = [var for var in tvars if 'score_res3' not in var.name]
            # CNN network and score_res3, need to change to other parameter name if using different pre-trained network.
            opt_var2 = [var for var in tvars if 'score_res3' in var.name]
            # Two optimizer together to train the network.
            train_op1 = opt.minimize(loss, var_list=opt_var1, global_step=global_step)
            train_op2 = opt2.minimize(loss, var_list=opt_var2, global_step=global_step)
            train_op = tf.group(train_op1, train_op2)

        # intialize variables
        sess.run(tf.global_variables_initializer())
        restore_iter = 0

        # load pre-trained CNN network.
        if self.pretrained_model is not None and not restore:
            try:
                print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
                self.net.load(self.pretrained_model, sess, True)
            except:
                raise 'Check your pretrained model {:s}'.format(self.pretrained_model)

        # Resuming a trainer if restore=True, need to specify a ckpt directory
        if restore:
            try:
                ckpt = tf.train.get_checkpoint_state(self.output_dir)
                print 'Restoring from {}...'.format(ckpt.model_checkpoint_path),
                self.saver.restore(sess, ckpt.model_checkpoint_path)
                stem = os.path.splitext(os.path.basename(ckpt.model_checkpoint_path))[0]
                restore_iter = int(stem.split('-')[-1])
                sess.run(global_step.assign(restore_iter))
                print 'done'
            except:
                print 'Failed to find a checkpoint!'
                sys.exit(1)

        last_snapshot_iter = -1
        iter_per_epoch = int(len(self.roidb)/cfg.TRAIN.BATCH_SIZE)
        restore_epoch = (restore_iter+1) / iter_per_epoch
        # Only use the following if you want learning rate drop by specified GAMMA value during training, default to not using
        '''
        for _ in range(0, int(restore_iter / cfg.TRAIN.STEPSIZE)):
            sess.run(tf.assign(lr, lr.eval() * cfg.TRAIN.GAMMA))
            sess.run(tf.assign(lr2, lr2.eval() * cfg.TRAIN.GAMMA))
        '''

        # Training Iteration
        timer = Timer()
        for epoch in range(restore_epoch, epochs):
          for e_iter in range(0, iter_per_epoch):
            timer.tic()
            iter = epoch*(iter_per_epoch) + e_iter
            # Only use the following if you want learning rate drop by specified GAMMA value during training,  default to not using
            '''
            #if iter != 0 and iter % cfg.TRAIN.STEPSIZE == 0:
            #    sess.run(tf.assign(lr, lr.eval() * cfg.TRAIN.GAMMA))
            #    sess.run(tf.assign(lr2, lr2.eval() * cfg.TRAIN.GAMMA))
            '''

            # Geting next minibatch, refer to 'roi_data_layer' directory
            blobs = data_layer.forward()

            feed_dict={
                self.net.data: blobs['data'],
                self.net.clsmap: blobs['clsmap'],
                self.net.regmap: blobs['regmap']}

            res_fetches = [self.net.get_output('score_cls'),
                           self.net.get_output('score_reg'),
                           self.net.get_output('prob_cls')]


            fetch_list = [loss, det_loss, huber_loss, summary_op, train_op] + res_fetches
            fetch_list += []

            # Where we feed input into tensor graph and obtain result
            loss_value, det_loss_value, huber_loss_value, summary_str, _, \
            score_cls_np, score_reg_np, prob_cls_np = sess.run(fetches=fetch_list, feed_dict=feed_dict)

            self.writer.add_summary(summary=summary_str, global_step=global_step.eval())

            _diff_time = timer.toc(average=False)

            # Generating image visualization in tensorboard every LOG_IMAGE_ITERS steps
            if (iter+1) % cfg.TRAIN.LOG_IMAGE_ITERS == 0:
                print 'Generating image log...'
                ori_im = np.squeeze(blobs['orig_img'])
                ori_im = ori_im.astype(dtype=np.uint8, copy=False)
                ori_im = _draw_gt_to_image(ori_im, blobs['gen_bbox'])

                boxes, scores = _process_boxes_scores(prob_cls_np[-1], score_reg_np[-1], self.centers)
                heatmap = draw_heatmap(prob_cls_np[-1], cfg.TRAIN.CROP_SIZE)

                if (len(boxes)>0):
                    refind_idx = nms(boxes, scores, overlapThresh=cfg.TRAIN.NMS_Thresh)
                    res = final_boxes[refind_idx, :]
                    conf = final_confidence[refind_idx, :]
                else:
                    res = []
                    conf = []
                image = cv2.cvtColor(_draw_boxes_to_image(ori_im, res, conf), cv2.COLOR_BGR2RGB)
                image_mux = np.stack((image, heatmap), axis=0)
                log_image_name_str = ('Epoch_%02d_' % (epoch+1) ) + '(' + blobs['im_name'] + ')'
                log_image_summary_op = sess.run(log_image, \
                                                feed_dict={log_image_name: log_image_name_str,log_image_data: image_mux})
                self.writer.add_summary(log_image_summary_op, global_step=global_step.eval())

            # Printing training information in terminal
            if (iter) % (cfg.TRAIN.DISPLAY) == 0:
                print 'Epoch: [%02d/%02d], iter: [%d/%d], Loss: %.4f, det_loss: %.4f, huber_loss: %.4f, lr: %f, lr2: %f'%\
                        ((epoch+1), epochs, (e_iter+1), iter_per_epoch, loss_value, det_loss_value, huber_loss_value, lr.eval(), lr2.eval())
                print 'speed: {:.3f}s / iter'.format(_diff_time)

            # Save a checkpoint every epoch
            if (iter+1) % iter_per_epoch == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter, epoch)

        # Saving final checkpoint before ending
        if last_snapshot_iter != iter:
            self.snapshot(sess, iter)

#==================================================================================================================================

def get_training_roidb(pkl_file):
    # This part only served as loading .pkl file
    with open(pkl_file, 'rb') as fid:
        roidb = cPickle.load(fid)
    assert (len(roidb) > 0), 'No data loaded!'
    print 'Loaded roidb done!'
    return roidb

def get_data_layer(roidb, centers, num_classes):
    # RoIDataLayer will handle both input and pre-processing of training dataset
    layer = RoIDataLayer(roidb, centers, num_classes)
    return layer

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

def _process_boxes_scores(prob_cls_np, score_reg_np, centers):
    # Process both scores and regression, for details please refer:
    # https://github.com/peiyunh/tiny/blob/master/tiny_face_detector.m
    if cfg.TRAIN.PRUNING:
        tids = list(range(4, 12)) + list(range(18, 25))
        ignoredTids = list(set(range(0,centers.shape[0]))-set(tids))
        prob_cls_np[:,:,ignoredTids] = 0.

    # Determine spatial location of face heatmap
    fy, fx, fc = np.where(prob_cls_np > cfg.TRAIN.CONFIDENCE_Thresh)

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
    return bbox, confidence

def _draw_boxes_to_image(im, res, conf):
    # Draw detected bounding box to image as yellow bounding boxes.
    # Since it does not use image pyramid, it may not represent the network detection ability.
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = np.copy(im)
    for ind, boxes in enumerate(res):
        (x1, y1, x2, y2) = boxes
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
        confidence = conf[ind][0]
        try:
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            text = 'Score={:.2f}'.format(confidence)
            cv2.putText(image, text, (x1, y1), font, 0.6, (0, 0, 0), 1)
        except:
          continue
    return image

def _draw_gt_to_image(im, gt_boxes):
    # Draw ground-truth bounding boxes to image as green bounding boxes
    image = np.copy(im)
    for i in range(0, gt_boxes.shape[0]):
        (x1, y1, x2, y2) = gt_boxes[i, :]
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
    return image

def draw_heatmap(clsmap, size):
    # This part just combine each anchor channel (Default 25) to a gray level channel, may not represent the real heatmap.
    heatmap = np.amax(clsmap, axis=-1)
    heatmap = (heatmap-np.amin(heatmap))*(255/(np.amax(heatmap)-np.amin(heatmap)))
    heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
    heatmap = cv2.resize(heatmap,(size, size), interpolation = cv2.INTER_NEAREST)
    return heatmap

def train_net(network, roidb, output_dir, log_dir, ref_Box, pretrained_model=None, epochs=20, restore=False):
    """Train a tiny face detection network."""

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sw = SolverWrapper(sess, network, roidb, output_dir, logdir=log_dir, centers=ref_Box, pretrained_model=pretrained_model)
        print 'Solving...'
        sw.train_model(sess, epochs, restore=restore)
        print 'done solving'
