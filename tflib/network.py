import tflib
import tflib.ops
import tensorflow as tf
import numpy as np

def alex_net(inp,DIM=512):
    X = tf.nn.relu(tflib.ops.conv2d('conv1', inp, 11, 4, 1, 96, bias=True, batchnorm=False, pad = 'SAME'))
    X = tflib.ops.max_pool('pool1', X, k=3, s=2)
    X = tflib.ops.norm('norm1', X, lsize=5)

    X = tf.nn.relu(tflib.ops.conv2d('conv2', X, 5, 1, 96, 256, bias=True, batchnorm=False, pad = 'SAME'))
    X = tflib.ops.max_pool('pool2', X, k=3, s=2)
    X = tflib.ops.norm('norm2', X, lsize=5)

    X = tf.nn.relu(tflib.ops.conv2d('conv3', X, 3, 1, 256, 384, bias=True, batchnorm=False, pad = 'SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv4', X, 3, 1, 384, 384, bias=True, batchnorm=False, pad = 'SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv5', X, 3, 1, 384, 256, bias=True, batchnorm=False, pad = 'SAME'))
    X = tflib.ops.max_pool('pool5', X, k=3, s=2)
    X = tflib.ops.norm('norm5', X, lsize=5)

    X = tf.nn.relu(tflib.ops.Linear('fc6',tf.reshape(X,[tf.shape(X)[0],-1]),32768,4096))
    X = tf.nn.dropout(X,0.5)

    X = tf.nn.relu(tflib.ops.Linear('fc7',X,4096,4096))
    X = tf.nn.dropout(X,0.5)

    X = tflib.ops.Linear('fc8',X,4096,DIM)

    return X

def alex_net_att(inp):
    X = tf.nn.relu(tflib.ops.conv2d('conv1', inp, 11, 4, 1, 96, bias=True, batchnorm=False, pad = 'SAME'))
    X = tflib.ops.max_pool('pool1', X, k=3, s=2)
    X = tflib.ops.norm('norm1', X, lsize=5)

    X = tf.nn.relu(tflib.ops.conv2d('conv2', X, 5, 1, 96, 256, bias=True, batchnorm=False, pad = 'SAME'))
    X = tflib.ops.max_pool('pool2', X, k=3, s=2)
    X = tflib.ops.norm('norm2', X, lsize=5)

    X = tf.nn.relu(tflib.ops.conv2d('conv3', X, 3, 1, 256, 384, bias=True, batchnorm=False, pad = 'SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv4', X, 3, 1, 384, 384, bias=True, batchnorm=False, pad = 'SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv5', X, 3, 1, 384, 256, bias=True, batchnorm=False, pad = 'SAME'))

    return X


def vgg16(X,num_feats=64):
    X = tf.nn.relu(tflib.ops.conv2d('conv1_1', X, 3, 1, 1, num_feats, bias=True, batchnorm=False, pad = 'SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv1_2', X, 3, 1, num_feats, num_feats, bias=True, batchnorm=False, pad = 'SAME'))
    X = tflib.ops.max_pool('pool1', X, k=2, s=2)

    X = tf.nn.relu(tflib.ops.conv2d('conv2_1', X, 3, 1, num_feats, 2*num_feats, bias=True, batchnorm=False, pad = 'SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv2_2', X, 3, 1, 2*num_feats, 2*num_feats, bias=True, batchnorm=False, pad = 'SAME'))
    X = tflib.ops.max_pool('pool2', X, k=2, s=2)

    X = tf.nn.relu(tflib.ops.conv2d('conv3_1', X, 3, 1, 2*num_feats, 4*num_feats, bias=True, batchnorm=False, pad = 'SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv3_2', X, 3, 1, 4*num_feats, 4*num_feats, bias=True, batchnorm=False, pad = 'SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv3_3', X, 3, 1, 4*num_feats, 4*num_feats, bias=True, batchnorm=False, pad = 'SAME'))
    X = tflib.ops.max_pool('pool3', X, k=2, s=2)

    X = tf.nn.relu(tflib.ops.conv2d('conv4_1', X, 3, 1, 4*num_feats, 8*num_feats, bias=True, batchnorm=False, pad = 'SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv4_2', X, 3, 1, 8*num_feats, 8*num_feats, bias=True, batchnorm=False, pad = 'SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv4_3', X, 3, 1, 8*num_feats, 8*num_feats, bias=True, batchnorm=False, pad = 'SAME'))
    X = tflib.ops.max_pool('pool4', X, k=2, s=2)

    X = tf.nn.relu(tflib.ops.conv2d('conv5_1', X, 3, 1, 8*num_feats, 8*num_feats, bias=True, batchnorm=False, pad = 'SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv5_2', X, 3, 1, 8*num_feats, 8*num_feats, bias=True, batchnorm=False, pad = 'SAME'))
    X = tf.nn.relu(tflib.ops.conv2d('conv5_3', X, 3, 1, 8*num_feats, 8*num_feats, bias=True, batchnorm=False, pad = 'SAME'))

    return X

def im2latex_cnn(X, num_feats, bn, train_mode='True'):
    X = X-128.
    X = X/128.

    X = tf.nn.relu(tflib.ops.conv2d('conv1', X, 3, 1, 1, num_feats, pad = 'SAME', bias=False)) 
    X = tflib.ops.max_pool('pool1', X, k=2, s=2)

    X = tf.nn.relu(tflib.ops.conv2d('conv2', X, 3, 1, num_feats, num_feats*2, pad = 'SAME', bias=False))
    X = tflib.ops.max_pool('pool2', X, k=2, s=2)

    X = tf.nn.relu(tflib.ops.conv2d('conv3', X, 3, 1, num_feats*2, num_feats*4,  batchnorm=bn, is_training=train_mode, pad = 'SAME', bias=False))

    X = tf.nn.relu(tflib.ops.conv2d('conv4', X, 3, 1, num_feats*4, num_feats*4, pad = 'SAME', bias=False))
    X = tflib.ops.max_pool('pool4', X, k=(1,2), s=(1,2))

    X = tf.nn.relu(tflib.ops.conv2d('conv5', X, 3, 1, num_feats*4, num_feats*8, batchnorm=bn, is_training=train_mode, pad = 'SAME', bias=False))
    X = tflib.ops.max_pool('pool5', X, k=(2,1), s=(2,1))

    X = tf.nn.relu(tflib.ops.conv2d('conv6', X, 3, 1, num_feats*8, num_feats*8, batchnorm=bn, is_training=train_mode, pad = 'SAME', bias=False))

    return X
