import math

import tensorflow as tf

from .CQAModel import CQAModel
from layers.BiGRU import NativeGRU as BiGRU, Dropout


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


class Baseline5_4(CQAModel):
    def build_model(self):
        with tf.variable_scope('ai_cnn', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, C = self.QS, self.CT
            Q_len, C_len = self.Q_len, self.C_len

            with tf.variable_scope('encode'):
                Q = add_timing_signal_1d(Q)
                C = add_timing_signal_1d(C)
                Q = Dropout(Q, keep_prob=self.dropout_keep_prob, is_train=self._is_train)
                C = Dropout(C, keep_prob=self.dropout_keep_prob, is_train=self._is_train)

                Q_sequence = tf.layers.conv1d(Q, filters=200, kernel_size=3, padding='same')
                C_sequence = tf.layers.conv1d(C, filters=200, kernel_size=3, padding='same')

            with tf.variable_scope('interaction'):
                Q_ = tf.expand_dims(Q_sequence, axis=2)  # (B, L1, 1, dim)
                C_ = tf.expand_dims(C_sequence, axis=1)  # (B, 1, L2, dim)
                hQ = tf.tile(Q_, [1, 1, self.C_maxlen, 1])
                hC = tf.tile(C_, [1, self.Q_maxlen, 1, 1])
                H = tf.concat([hQ, hC], axis=-1)
                # H = tf.concat([tf.abs(hQ - hC)], axis=-1)
                units = 200
                A = tf.layers.dense(H, units=units, activation=tf.tanh)  # (B, L1, L2, dim)
                v = tf.get_variable('v', [units, 1], dtype=tf.float32)

                # convolution
                info = tf.layers.conv2d(A, units, 5, activation=tf.nn.relu)  # (B, NL1, NL2, units)
                score = tf.keras.backend.dot(info, v)  # (B, NL1, NL2, 1)

                eroll_info = tf.reshape(score, [self.N, -1])
                eroll_A = tf.reshape(A, [self.N, -1, units])

                info_top_k, indices = tf.nn.top_k(eroll_info, k=10)  # (B, k)
                A_top_k = tf.batch_gather(eroll_A, indices)

                alpha = tf.expand_dims(tf.nn.softmax(info_top_k, axis=1), axis=-1)
                res = tf.reduce_sum(alpha * A_top_k, axis=1)

            info = res
            info = Dropout(info, keep_prob=self.dropout_keep_prob, is_train=self._is_train)
            median = tf.layers.dense(info, 300, activation=tf.tanh)
            output = tf.layers.dense(median, self.args.categories_num, activation=tf.identity)

            return output

