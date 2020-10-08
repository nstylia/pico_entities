import tensorflow as tf
from functools import reduce
from operator import mul
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python.util import nest


#source https://github.com/IsaacChanghau/Dense_BiLSTM/blob/master/models/nns.py

def dense(inputs, hidden_dim, use_bias=True, scope='dense'):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(len(inputs.get_shape().as_list()) - 1)] + [hidden_dim]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        w = tf.get_variable("weight", shape=[dim, hidden_dim], dtype=tf.float32)
        output = tf.matmul(flat_inputs, w)
        if use_bias:
            b = tf.get_variable("bias", shape=[hidden_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.))
            output = tf.nn.bias_add(output, b)
        output = tf.reshape(output, out_shape)
        return output

def highway_layer(arg, bias, bias_start=0.0, scope=None, keep_prob=None, is_train=None):
    with tf.variable_scope(scope or "highway_layer"):
        d = arg.get_shape()[-1]
        trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', keep_prob=keep_prob, is_train=is_train)
        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', keep_prob=keep_prob, is_train=is_train)
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, keep_prob=None, is_train=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx),
                                keep_prob=keep_prob, is_train=is_train)
            prev = cur
        return cur

def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, keep_prob=None, is_train=None):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("args must be specified")
    if not nest.is_sequence(args):
        args = [args]
    flat_args = [flatten(arg, 1) for arg in args]
    if keep_prob is not None and is_train is not None:
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, keep_prob), lambda: arg) for arg in flat_args]
    with tf.variable_scope(scope or 'linear'):
        flat_out = _linear(flat_args, output_size, bias, bias_initializer=tf.constant_initializer(bias_start))
    out = reconstruct(flat_out, args[0], 1)
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
    return out


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out

def attention(inputs):
    # Trainable parameters
    hidden_size = inputs.shape[2].value
    u_omega = tf.get_variable("u_omega", [hidden_size], initializer=tf.keras.initializers.glorot_normal())

    with tf.name_scope('v'):
        v = tf.tanh(inputs)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    # Final output with tanh
    output = tf.tanh(output)

    return output, alphas