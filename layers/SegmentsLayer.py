import collections

import tensorflow as tf
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import init_ops, array_ops, nn_ops, math_ops
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell, GRUCell
from tensorflow.python.layers import base as base_layer

_SegGRUStateTuple = collections.namedtuple("SegGRUStateTuple", ("h", "p"))
_SegInferStateTuple = collections.namedtuple("SegInferStateTuple", ("h", ))


class SegGRUStateTuple(_SegGRUStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (h, p) = self
        if h.dtype != p.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s" %
                            (str(h.dtype), str(p.dtype)))
        return h.dtype


class SegInferStateTuple(_SegInferStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        # (h, inner_fake) = self
        # if h.dtype != inner_fake.dtype:
        #     raise TypeError("Inconsistent internal state: %s vs %s" %
        #                     (str(h.dtype), str(inner_fake.dtype)))
        h, = self
        return h.dtype


class SegGRUCell(GRUCell):
    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units,
                 word_dim=300,
                 epsilon=0,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 sent_len=None,
                 sent_memory=None,
                 sent_mask=None,
                 name=None):
        super(SegGRUCell, self).__init__(num_units=num_units,
                                         activation=activation,
                                         reuse=reuse,
                                         kernel_initializer=kernel_initializer,
                                         bias_initializer=bias_initializer,
                                         name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)
        self._epsilon = epsilon
        self._activation = activation or tf.tanh
        assert sent_len is not None
        self.sent_len = sent_len
        self._word_dim = word_dim
        self.sent_memory = sent_memory
        self.sent_mask = sent_mask
        self.sent_eye = None

    def set_sentence(self, sent_memory, sent_mask, sent_len):
        self.sent_len = sent_len
        self.sent_memory = sent_memory
        self.sent_mask = sent_mask
        self.sent_eye = tf.eye(sent_len)

    @property
    def state_size(self):
        return SegGRUStateTuple(self._num_units, 1)

    @property
    def output_size(self):
        return self._word_dim + 1

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
          raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                           % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self._num_units

        # ========================================================================

        super(SegGRUCell, self).build(inputs_shape)

    def hard_softmax(self, memory, mask, vector):
        copy_vec = tf.tile(tf.expand_dims(vector, axis=1), [1, self.sent_len, 1])
        concate_median = tf.concat([memory, copy_vec])
        res = tf.matmul(concate_median, self._mem_kernel)  # (B, L, new_dim)
        res = tf.nn.bias_add(res, self._mem_bias)
        weight_score = tf.matmul(self._activation(res), self._v)  # (B, L, 1)

        if self.sent_mask:
            weight_score -= (1 - tf.expand_dims(self.sent_mask, axis=2)) * 1e30

        alpha = tf.nn.softmax(weight_score, axis=1)
        return alpha, tf.nn.top_k(alpha, 1)

    def MLP(self, cur_vec, cur_h):
        info = tf.matmul(tf.concat([cur_vec, cur_h], 1), self._mlp_kernel)
        return tf.nn.bias_add(info, self._mlp_bias)


    def call(self, inputs, state, scope=None):
        h_tm, p_tm = state

        # =======================================================================
        alpha, (values, indices) = self.hard_softmax(self.sent_memory, self.sent_mask, h_tm)
        m = tf.gather(self.sent_eye, indices, axis=0)
        self.sent_mask *= (1 - m)

        word = tf.batch_gather(self.sent_memory, indices)
        cur_vec = tf.reduce_sum(alpha * self.sent_memory, axis=1)
        h_t = self.call(cur_vec, h_tm)

        drop_p = tf.sigmoid(self.MLP(cur_vec, h_tm))
        p_t = tf.maximum(0., p_tm - drop_p)

        m = tf.cast(tf.greater(p_t, 0.), tf.float32)
        word_m = tf.concat([word, cur_vec, m], axis=1)

        new_state = SegGRUStateTuple(h_t, p_t)
        return word_m, new_state


class SegInferCell(LayerRNNCell):
    def compute_output_shape(self, input_shape):
        pass

    def __init__(self, num_units,
                 word_dim=300,
                 epsilon=0,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None,
                 inner_fake=None,
                 sent_len=None,
                 sent_memory=None,
                 sent_mask=None,
                 name=None):
        super(SegInferCell, self).__init__(_reuse=reuse, name=name)

        # Inputs must be 2-dimensional.
        self.input_spec = base_layer.InputSpec(ndim=2)
        self.word_dim = word_dim
        self.inner_fake = inner_fake
        self.p_0 = tf.ones([tf.shape(inner_fake)[0], 1], dtype=tf.float32)
        self.inner_cell = SegGRUCell(num_units,
                                     word_dim,
                                     epsilon,
                                     activation,
                                     reuse,
                                     kernel_initializer,
                                     bias_initializer,
                                     sent_len,
                                     sent_memory,
                                     sent_mask,
                                     name)

    def set_sentence(self, sent_memory, sent_mask, sent_len):
        self.inner_cell.sent_len = sent_len
        self.inner_cell.sent_memory = sent_memory
        self.inner_cell.sent_mask = sent_mask
        self.inner_cell.sent_eye = tf.eye(sent_len)

    @property
    def state_size(self):
        return SegInferStateTuple(self.word_dim, )

    @property
    def output_size(self):
        return self.word_dim

    def build(self, inputs_shape):
        if inputs_shape[1].value is None:
            raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                             % inputs_shape)

        input_depth = inputs_shape[1].value
        h_depth = self.word_dim

        # ========================================================================

        self.built = True

    def call(self, inputs, state, scope=None):
        h_tm, = state

        # =======================================================================

        inner_state = SegGRUStateTuple(h_tm, self.p_0)
        segments, final_states = tf.nn.dynamic_rnn(self.inner_cell, self.inner_fake, initial_state=inner_state)
        h_t, p_t = final_states
        alpha = self.att(segments, self.r)
        word_m = tf.reduce_sum(alpha * segments, axis=1)

        new_state = SegGRUStateTuple(h_t, )
        return word_m, new_state
