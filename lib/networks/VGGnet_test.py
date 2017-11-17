import tensorflow as tf
from .network import Network
from ..tiny.config import cfg


class VGGnet_test(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.layers = dict({'data': self.data})
        self.trainable = trainable
        self.setup()

    def setup(self):

        (self.feed('data')
         .conv(3, 3, 64, 1, 1, name='conv1_1', trainable=False)
         .conv(3, 3, 64, 1, 1, name='conv1_2', trainable=False)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1', trainable=False)
         .conv(3, 3, 128, 1, 1, name='conv2_2', trainable=False)
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1')
         .conv(3, 3, 256, 1, 1, name='conv3_2')
         .conv(3, 3, 256, 1, 1, name='conv3_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1')
         .conv(3, 3, 512, 1, 1, name='conv4_2')
         .conv(3, 3, 512, 1, 1, name='conv4_3')
         .max_pool(2, 2, 2, 2, padding='VALID', name='pool4'))
        #========= FCN ============
        (self.feed('pool4')
             .conv(1, 1, 125, 1, 1, biased=True, relu=False, name='score_res4'))
        (self.feed('pool3')
             .conv(1, 1, 125, 1, 1, biased=True, relu=False, name='score_res3'))
        (self.feed('score_res4', 'score_res3')
             .upscale_bil_spec(name='score4f'))
        (self.feed('score4f', 'score_res3')
             .add(name='score_map'))
        (self.feed('score_map')
             .get_score_cls(25, name='score_cls'))
        (self.feed('score_map')
             .get_score_reg(25, name='score_reg'))
        (self.feed('score_cls')
             .sigmoid(name='prob_cls'))
