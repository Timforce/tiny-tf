import numpy as np
import tensorflow as tf

from ..tiny.config import cfg

DEFAULT_PADDING = 'SAME'

def include_original(dec):
    """ Meta decorator, which make the original function callable (via f._original() )"""
    def meta_decorator(f):
        decorated = dec(f)
        decorated._original = f
        return decorated
    return meta_decorator

@include_original
def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, ignore_missing=False):
        data_dict = np.load(data_path).item()
        for key in data_dict:
            with tf.variable_scope(key, reuse=True):
                for subkey in data_dict[key]:
                    try:
                        var = tf.get_variable(subkey)
                        session.run(var.assign(data_dict[key][subkey]))
                        print "assign pretrain model "+subkey+ " to "+key
                    except ValueError:
                        print "ignore "+key
                        if not ignore_missing:

                            raise

    def feed(self, *args):
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            if isinstance(layer, basestring):
                try:
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    def get_unique_name(self, prefix):
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        return '%s_%d'%(prefix, id)

    def make_var(self, name, shape, initializer=None, trainable=True, regularizer=None):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable, regularizer=regularizer)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, biased=True,relu=True, padding=DEFAULT_PADDING, trainable=True):
        """ contribution by miraclebiu, and biased option"""
        self.validate_padding(padding)
        c_i = input.get_shape()[-1]
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:

            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
            # init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            # init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            init_biases = tf.constant_initializer(0.0)
            kernel = self.make_var('weights', [k_h, k_w, c_i, c_o], init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            if biased:
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                conv = convolve(input, kernel)
                if relu:
                    bias = tf.nn.bias_add(conv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(conv, biases)
            else:
                conv = convolve(input, kernel)
                if relu:
                    return tf.nn.relu(conv)
                return conv

    @layer
    def upconv(self, input, shape, c_o, ksize=4, stride = 2, name = 'upconv', biased=False, relu=True, padding=DEFAULT_PADDING,
             trainable=True):
        """ up-conv"""
        self.validate_padding(padding)

        c_in = input.get_shape()[3].value
        in_shape = tf.shape(input)
        if shape is None:
            # h = ((in_shape[1] - 1) * stride) + 1
            # w = ((in_shape[2] - 1) * stride) + 1
            h = ((in_shape[1] ) * stride)
            w = ((in_shape[2] ) * stride)
            new_shape = [in_shape[0], h, w, c_o]
        else:
            new_shape = [in_shape[0], shape[1], shape[2], c_o]
        output_shape = tf.stack(new_shape)

        filter_shape = [ksize, ksize, c_o, c_in]

        with tf.variable_scope(name) as scope:
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            # init_weights = tf.contrib.layers.variance_scaling_initializer(factor=0.01, mode='FAN_AVG', uniform=False)
            filters = self.make_var('weights', filter_shape, init_weights, trainable, \
                                   regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            deconv = tf.nn.conv2d_transpose(input, filters, output_shape,
                                            strides=[1, stride, stride, 1], padding=DEFAULT_PADDING, name=scope.name)
            # coz de-conv losses shape info, use reshape to re-gain shape
            deconv = tf.reshape(deconv, new_shape)

            if biased:
                init_biases = tf.constant_initializer(0.0)
                biases = self.make_var('biases', [c_o], init_biases, trainable)
                if relu:
                    bias = tf.nn.bias_add(deconv, biases)
                    return tf.nn.relu(bias)
                return tf.nn.bias_add(deconv, biases)
            else:
                if relu:
                    return tf.nn.relu(deconv)
                return deconv

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def get_score_cls(self, input, index, name):
        with tf.variable_scope(name) as scope:
            score_cls = tf.slice(input, [0, 0, 0, 0], [-1, -1, -1, index])
            return score_cls

    @layer
    def get_score_reg(self, input, index, name):
        with tf.variable_scope(name) as scope:
            score_reg = tf.slice(input, [0, 0, 0, index], [-1, -1, -1, -1])
            return score_reg

    @layer
    def sigmoid(self, input, name):
        sig = tf.nn.sigmoid(input, name=name)
        return sig

    @layer
    def upscale_bil(self, input, name, ratio = 2):
        with tf.variable_scope(name) as scope:
            in_shape = tf.shape(input)
            h = ((in_shape[1] )* ratio)
            w = ((in_shape[2] )* ratio)
            bil_upscale = tf.image.resize_bilinear(input, [h, w])
            return bil_upscale

    @layer
    def upscale_bil_spec(self, input, name):
        # This is used for preventing spatial dimension mis-match between convolutoinal or pooling layers
        with tf.variable_scope(name) as scope:
            out_shape = tf.shape(input[1])
            h = out_shape[1]
            w = out_shape[2]
            bil_upscale = tf.image.resize_bilinear(input[0], [h, w])
            return bil_upscale

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat(axis=axis, values=inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input,[0,3,1,2]), [-1, dim])
            else:
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable, \
                                    regularizer=self.l2_regularizer(cfg.TRAIN.WEIGHT_DECAY))
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def add(self,input,name):
        """contribution by miraclebiu"""
        return tf.add(input[0],input[1], name=name)

    @layer
    def batch_normalization(self,input,name,relu=True, is_training=False):
        """contribution by miraclebiu"""
        if relu:
            #temp_layer=tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            temp_layer=tf.contrib.layers.batch_norm(input, decay=0.9, epsilon=0.00001, updates_collections=None, scale=True,center=True,is_training=is_training,scope=name)
            return tf.nn.relu(temp_layer)
        else:
            #return tf.contrib.layers.batch_norm(input,scale=True,center=True,is_training=is_training,scope=name)
            return tf.contrib.layers.batch_norm(input, decay=0.9, epsilon=0.00001, updates_collections=None, scale=True,center=True,is_training=is_training,scope=name)

    @layer
    def negation(self, input, name):
        """ simply multiplies -1 to the tensor"""
        return tf.multiply(input, -1.0, name=name)

    @layer
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name=name)

    def l2_regularizer(self, weight_decay=0.0005, scope=None):
        def regularizer(tensor):
            with tf.name_scope(scope, default_name='l2_regularizer', values=[tensor]):
                l2_weight = tf.convert_to_tensor(weight_decay,
                                       dtype=tensor.dtype.base_dtype,
                                       name='weight_decay')
                return tf.multiply(l2_weight, tf.nn.l2_loss(tensor), name='value')
        return regularizer

    def smooth_l1_dist(self, deltas, sigma2=9.0, name='smooth_l1_dist'):
        with tf.name_scope(name=name) as scope:
            deltas_abs = tf.abs(deltas)
            smoothL1_sign = tf.cast(tf.less(deltas_abs, 1.0/sigma2), tf.float32)
            return tf.square(deltas) * 0.5 * sigma2 * smoothL1_sign + \
                        (deltas_abs - 0.5 / sigma2) * tf.abs(smoothL1_sign - 1)

    def build_loss(self):
        ## Important part building loss function that train the network ##

        # Information from training architecture
        score_cls = tf.reshape(self.get_output('score_cls'), [-1])
        score_reg = tf.reshape(self.get_output('score_reg'), [-1])
        label_cls = tf.reshape(self.clsmap, [-1])
        label_reg = tf.reshape(self.regmap, [-1])

        # Logistic loss that determine face heatmap, please refer to:
        # https://github.com/peiyunh/matconvnet/blob/d972fe848900c1c56d32e7a955ca38d8e05e0669/matlab/vl_nnloss.m#L244
        a = -label_cls * score_cls
        b = tf.maximum(a, 0)
        t = b + tf.log(tf.exp(-b) + tf.exp(a-b))
        y = t * tf.abs(label_cls)   # t(:) * instanceWeights(:)
        # OHEM processing, please refer to:
        # https://arxiv.org/abs/1604.03540
        # https://github.com/peiyunh/tiny/blob/master/cnn_train_dag_hardmine.m#L293
        mine_keep = tf.where(tf.greater(y, 0.03))  # keep label_cls(loss_cls_map>0.03), discard the rest. Same as label_cls(loss_cls_map<0.03) = 0

        # We want to keep positive and negative anchors to 1:1 as possible, may not always be 1:1, thus cause loss value oscillation
        # Because negative almost always > 128 sample for each image, but positive may be a few for some images.
        # For details, please refer to pre-process file 'minibatch.pyx'
        # det_loss is our loss that determine potential face heatmap
        score_mine = tf.reshape(tf.gather(y, mine_keep), [-1])
        label_mine = tf.reshape(tf.gather(label_cls, mine_keep), [-1])
        pos_idx = tf.random_shuffle(tf.where(tf.equal(label_mine, 1.0)))
        neg_idx = tf.random_shuffle(tf.where(tf.equal(label_mine, -1.0)))
        sample_num = int(cfg.TRAIN.SAMPLE_LIMIT * cfg.TRAIN.BATCH_SIZE)
        pos_keep = tf.cond(tf.shape(pos_idx)[0] < sample_num, lambda: tf.gather(score_mine, pos_idx), \
                                                    lambda: tf.slice(tf.reshape(tf.gather(score_mine, pos_idx), [-1]), [0], [sample_num]))
        neg_keep = tf.cond(tf.shape(neg_idx)[0] < sample_num, lambda: tf.gather(score_mine, neg_idx), \
                                                    lambda: tf.slice(tf.reshape(tf.gather(score_mine, neg_idx), [-1]), [0], [sample_num]))
        det_loss = (tf.reduce_sum(pos_keep) + tf.reduce_sum(neg_keep)) / float(cfg.TRAIN.BATCH_SIZE)

        # This part process regression loss (aka. huber loss in paper)
        # huber_loss is our loss that used for bounding box regression
        positive_keep = tf.where(tf.equal(tf.reshape(tf.tile(self.clsmap, [1, 1, 1, 4]), [-1]), 1.0))
        reg_score = tf.gather(score_reg, positive_keep)
        reg_label = tf.gather(label_reg, positive_keep)
        reg_value = self.smooth_l1_dist((reg_score-reg_label), sigma2=1.0)
        huber_loss = tf.reduce_sum(reg_value) / float(cfg.TRAIN.BATCH_SIZE)

        # Combine both for our training
        loss = det_loss + huber_loss

        # This part is for parameter regulation (not regression!)
        if cfg.TRAIN.WEIGHT_DECAY > 0:
            regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            loss = tf.add_n(regularization_losses) + loss
        return loss, det_loss, huber_loss
