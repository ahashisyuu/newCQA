import collections
import tensorflow as tf

from tensorflow.contrib.rnn import LayerRNNCell, RNNCell
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.keras import initializers

_DoubleStateTuple = collections.namedtuple("DoubleStateTuple", ("s1", "s2",
                                                                "values",
                                                                "r_h"))


class DoubleStateTuple(_DoubleStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        return self[0].dtype


class DoubleCell(LayerRNNCell):
    def __init__(self,
                 num_units,
                 sent_cell: RNNCell,
                 r_cell: RNNCell,
                 sentence_transformer,
                 sent1,
                 sent2,
                 sent1_length,
                 sent2_length,
                 dim,
                 keys,
                 sent1_mask=None,
                 sent2_mask=None,
                 initializer=None,
                 activation=tf.nn.relu,
                 use_bias=False,
                 reuse=None, name="Double", dtype=None, **kwargs):
        super(DoubleCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)

        self.num_units = num_units

        self.sent_cell = sent_cell
        self.r_cell = r_cell
        self.sent_transformer = sentence_transformer

        self.sent1 = sent1
        self.sent2 = sent2
        self.sent1_length = sent1_length
        self.sent2_length = sent2_length
        self.sent1_mask = sent1_mask
        self.sent2_mask = sent2_mask

        self.dim = dim
        self.initializer = initializers.get(initializer)
        self.keys = keys
        self.activation = activation  # \phi
        self.use_bias = use_bias
        self.initializer = initializer

        self.keys_num, _ = keys.get_shape()

        self._state_size = DoubleStateTuple(self.sent1_length * self.dim,  # sentences
                                            self.sent2_length * self.dim,

                                            self.keys_num * self.dim,  # values, relation vectors

                                            self.keys_num * self.dim,  # relation hidden states
                                            )

        self._output_size = self.keys_num * self.dim + 1
        # "1" stands for the probability that finally we select this time step as output

    def zero_state(self, batch_size, dtype):
        """Initialize the memory to the key values."""

        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            sent1 = tf.reshape(self.sent1, [-1, self.sent1_length * self.dim])
            sent2 = tf.reshape(self.sent2, [-1, self.sent2_length * self.dim])

            values = _zero_state_tensors([self.keys_num * self.dim], batch_size, dtype)

            rh = tf.reshape(self.keys, [self.keys_num * self.dim])
            rh = tf.tile(tf.expand_dims(rh, axis=0), [batch_size, 1])

            state_list = [sent1, sent2, values[0], rh]

            return DoubleStateTuple(*state_list)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):

        bias_init = 1.0  # or 0.0
        # parameters which relation updating needs
        self.r_kernel = self.add_variable(name="r_kernel",
                                          shape=[2 * self.dim, self.num_units],
                                          initializer=self.initializer)
        self.r_s_kernel = self.r_kernel[:self.dim]
        self.r_k_kernel = self.r_kernel[self.dim:]

        self.r_gate_kernel = self.add_variable(name="r_gate_kernel",
                                               shape=[3 * self.dim, 1],
                                               initializer=self.initializer)

        # all bias
        if self.use_bias:
            self.r_bias = self.add_variable(name="r_bias",
                                            shape=[2 * self.num_units],
                                            initializer=tf.constant_initializer(bias_init, dtype=tf.float32))
            self.r_s_bias = self.r_bias[:self.num_units]
            self.r_k_bias = self.r_bias[self.num_units:]

            self.r_gate_bias = self.add_variable(name="r_gate_bias",
                                                 shape=[1],
                                                 initializer=tf.constant_initializer(bias_init, dtype=tf.float32))

        self.built = True

    def call(self, inputs, state=None):
        s1_tm, s2_tm, values_tm, rh_tm = state
        # s: (B, Lx * dim), values: (B, keys_num * dim),
        # r_h: (B, keys_num * dim)

        # updating relation
        values, rh = self.update_relation(values_tm,
                                          s1_tm, self.sent1_length, self.sent1_mask,
                                          s2_tm, self.sent2_length, self.sent2_mask,
                                          rh_tm)

        # updating sentence presentation
        s1 = self.sent_transformer(s1_tm, self.sent1_mask)
        s2 = self.sent_transformer(s2_tm, self.sent2_mask)

        score = self.pro_compute(rh)  # (B, 1)
        m = tf.concat([values, score], axis=1)

        state = [s1, s2, values, rh]

        return m, DoubleStateTuple(*state)

    def update_relation(self, values_tm,
                        s1, s1_len, s1_mask,
                        s2, s2_len, s2_mask,
                        r_h_tm):
        # values_tm: (B, keys_num * dim),
        # s1, s2: (B, Lx * dim);  s1_len, s2_len: scalar;  s1_mask, s2_mask: (B, Lx) or None
        # r_h_tm: (B, keys_num * dim)

        values_tm = tf.reshape(values_tm, [-1, self.keys_num, self.dim])  # (keys_num, dim)
        s1 = tf.reshape(s1, [-1, s1_len, self.dim])                       # (B, L1, dim)
        s2 = tf.reshape(s2, [-1, s2_len, self.dim])                       # (B, L2, dim)
        r_h_tm = tf.reshape(r_h_tm, [-1, self.keys_num, self.dim])        # (B, keys_num, dim)

        # <1> GET new_values (new relation values)
        s_cat = tf.concat([s1, s2], axis=1)  # (B, L1 + L2, dim) -> (B, L, dim)
        if s1_mask is not None:
            s_mask = tf.concat([s1_mask, s2_mask], axis=1)
        else:
            s_mask = None

        batch_size = tf.shape(s1)[0]

        res_s = tf.keras.backend.dot(s_cat, self.r_s_kernel)  # (B, L, units)
        res_k = tf.keras.backend.dot(r_h_tm, self.r_k_kernel)  # (B, keys_num, units)

        if self.use_bias:
            res_s = tf.nn.bias_add(res_s, self.r_s_bias)
            res_k = tf.nn.bias_add(res_k, self.r_k_bias)

        res_s = self.activation(res_s)
        res_k = self.activation(res_k)

        score = tf.keras.backend.batch_dot(res_s, res_k, axes=[2, 2])  # (B, L, keys_num)
        # score = tf.expand_dims(score, axis=-1)  # (B, L, keys_num, 1)

        if s_mask is not None:
            s_mask = tf.expand_dims(s_mask, axis=-1)
            score -= (1 - s_mask) * 1e10
        alpha = tf.nn.softmax(score, axis=1)

        new_values = tf.keras.backend.batch_dot(alpha, s_cat, axes=[1, 1])  # (B, keys_nums, dim)

        # <2> GET gate
        temp_tensor = tf.concat([values_tm, new_values, r_h_tm], axis=-1)  # (B, keys_nums, 3*dim)
        temp_array = tf.keras.backend.dot(temp_tensor, self.r_gate_kernel)  # (B, keys_nums, 3*dim) or (B, keys_nums, 1)
        if self.use_bias:
            temp_array = tf.nn.bias_add(temp_array, self.r_gate_bias)  # bias: (3*dim,) or (1,)
        gate = tf.sigmoid(temp_array)

        # <3> UPDATE
        values_t = values_tm + gate * new_values
        values = tf.nn.l2_normalize(values_t)
        values = tf.reshape(values, [-1, self.keys_num * self.dim])

        r_cell_input = tf.reshape(new_values, [-1, self.dim])
        r_h_tm_reshaped = tf.reshape(r_h_tm, [-1, self.dim])
        r_h_reshaped, _ = self.r_cell(r_cell_input, r_h_tm_reshaped)
        r_h = tf.reshape(r_h_reshaped, [-1, self.keys_num * self.dim])

        return values, r_h

    def pro_compute(self, rh):
        # coming soon
        B = tf.shape(rh)[0]

        output = tf.ones([B, 1], dtype=tf.float32)

        return output

