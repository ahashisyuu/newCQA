import collections
import tensorflow as tf

from tensorflow.contrib.rnn import LayerRNNCell, RNNCell
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.keras import initializers

_TriangularStateTuple = collections.namedtuple("TriangularStateTuple", ("s1", "s2", "s3",
                                                                        "values1", "values2", "values3",
                                                                        "r1_h", "r2_h", "r3_h"))


class TriangularStateTuple(_TriangularStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        return self[0].dtype


class TriangularCell(LayerRNNCell):
    def __init__(self,
                 num_units,
                 sent_cell: RNNCell,
                 r_cell: RNNCell,
                 sentence_transformer,
                 sent1,
                 sent2,
                 sent3,
                 sent1_length,
                 sent2_length,
                 sent3_length,
                 dim,
                 keys,
                 sent1_mask=None,
                 sent2_mask=None,
                 sent3_mask=None,
                 initializer=None,
                 activation=tf.nn.relu,
                 use_bias=False,
                 reuse=None, name="Triangular", dtype=None, **kwargs):
        super(TriangularCell, self).__init__(_reuse=reuse, name=name, dtype=dtype, **kwargs)

        self.num_units = num_units

        self.sent_cell = sent_cell
        self.r_cell = r_cell
        self.sent_transformer = sentence_transformer

        self.sent1 = sent1
        self.sent2 = sent2
        self.sent3 = sent3
        self.sent1_length = sent1_length
        self.sent2_length = sent2_length
        self.sent3_length = sent3_length
        self.sent1_mask = sent1_mask
        self.sent2_mask = sent2_mask
        self.sent3_mask = sent3_mask

        self.dim = dim
        self.initializer = initializers.get(initializer)
        self.keys = keys
        self.activation = activation  # \phi
        self.use_bias = use_bias
        self.initializer = initializer

        self.keys_num, _ = keys.get_shape()

        self._state_size = TriangularStateTuple(self.sent1_length * self.dim,  # sentences
                                                self.sent2_length * self.dim,
                                                self.sent3_length * self.dim,

                                                self.keys_num * self.dim,  # values, relation vectors
                                                self.keys_num * self.dim,
                                                self.keys_num * self.dim,

                                                self.keys_num * self.dim,  # relation hidden states
                                                self.keys_num * self.dim,
                                                self.keys_num * self.dim)

        self._output_size = self.keys_num * self.dim * 3 + 1
        # "1" stands for the probability that finally we select this time step as output

    def zero_state(self, batch_size, dtype):
        """Initialize the memory to the key values."""

        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            sent1 = tf.reshape(self.sent1, [-1, self.sent1_length * self.dim])
            sent2 = tf.reshape(self.sent2, [-1, self.sent2_length * self.dim])
            sent3 = tf.reshape(self.sent3, [-1, self.sent3_length * self.dim])

            values = _zero_state_tensors([self.keys_num * self.dim] * 3, batch_size, dtype)

            rh = tf.reshape(self.keys, [self.keys_num * self.dim])
            rh = tf.tile(tf.expand_dims(rh, axis=0), [batch_size, 1])
            r1_h = r2_h = r3_h = rh

            state_list = [sent1, sent2, sent3] + values + [r1_h, r2_h, r3_h]

            return TriangularStateTuple(*state_list)

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
                                          shape=[3 * self.dim, self.num_units],
                                          initializer=self.initializer)
        self.r_v = self.add_variable(name="r_v",
                                     shape=[self.num_units, 1],
                                     initializer=self.initializer)

        self.r_gate_kernel = self.add_variable(name="r_gate_kernel",
                                               shape=[3 * self.dim, 1],
                                               initializer=self.initializer)

        # parameters which sentence updating needs
        # self.att_r_kernel = self.add_variable(name="att_r_kernel",
        #                                       shape=[2 * self.dim, self.num_units],
        #                                       initializer=self.initializer)
        # self.att_r_v = self.r_v
        #
        # self.s_kernel = self.add_variable(name="s_kernel",
        #                                   shape=[4 * self.dim, self.num_units],
        #                                   initializer=self.initializer)
        # self.s_v = self.r_v
        #
        # self.att_s_kernel = self.add_variable(name="att_s_kernel",
        #                                       shape=[2 * self.dim, self.num_units],
        #                                       initializer=self.initializer)
        # self.att_s_v = self.r_v

        # all bias
        if self.use_bias:
            self.r_bias = self.add_variable(name="r_bias",
                                            shape=[self.num_units],
                                            initializer=tf.constant_initializer(bias_init, dtype=tf.float32))
            self.r_gate_bias = self.add_variable(name="r_gate_bias",
                                                 shape=[1],
                                                 initializer=tf.constant_initializer(bias_init, dtype=tf.float32))

        self.built = True

    def call(self, inputs, state=None):
        s1_tm, s2_tm, s3_tm, \
            values1_tm, values2_tm, values3_tm, \
            r1_h_tm, r2_h_tm, r3_h_tm = state
        # s: (B, Lx * dim), values: (B, keys_num * dim),
        # r_h: (B, keys_num * dim)

        # updating relation
        values1, r1_h = self.update_relation(values1_tm,
                                             s1_tm, self.sent1_length, self.sent1_mask,
                                             s2_tm, self.sent2_length, self.sent2_mask,
                                             r1_h_tm)
        values2, r2_h = self.update_relation(values2_tm,
                                             s2_tm, self.sent2_length, self.sent2_mask,
                                             s3_tm, self.sent3_length, self.sent3_mask,
                                             r2_h_tm)
        values3, r3_h = self.update_relation(values3_tm,
                                             s3_tm, self.sent3_length, self.sent3_mask,
                                             s1_tm, self.sent1_length, self.sent1_mask,
                                             r3_h_tm)

        # updating sentence presentation
        s1 = s1_tm
        s2 = s2_tm
        s3 = s3_tm

        output = tf.concat([values1, values2, values3], axis=1)  # (B, 3 * keys_num * dim)
        score = self.pro_compute(r1_h, r2_h, r3_h)  # (B, 1)
        m = tf.concat([output, score], axis=1)

        state = [s1, s2, s3,
                 values1, values2, values3,
                 r1_h, r2_h, r3_h]

        return m, TriangularStateTuple(*state)

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
        r_h_tm = tf.reshape(r_h_tm, [-1, self.keys_num, self.dim])        # (keys_num, dim)

        # <1> GET new_values (new relation values)
        s_cat = tf.concat([s1, s2], axis=1)  # (B, L1 + L2, dim) -> (B, L, dim)
        if s1_mask is not None:
            s_mask = tf.concat([s1_mask, s2_mask], axis=1)
        else:
            s_mask = None

        batch_size = tf.shape(s1)[0]
        keys = tf.tile(tf.expand_dims(self.keys, axis=0), [batch_size, 1, 1])
        key_cat = tf.concat([keys, r_h_tm], axis=2)  # (B, keys_num, 2*dim)

        # #===========
        # score function:
        #   1):
        #     s = tf.tile(tf.expand_dims(s_cat, axis=2), [1, 1, self.keys_num, 1])  # (B, L, keys_num, dim)
        #     key = tf.tile(tf.expand_dims(key_cat, axis=1), [1, s1_len + s2_len, 1, 1])  # (B, L, keys_num, 2*dim)
        #
        #     temp_tensor = tf.concat([s, key], axis=3)  # (B, L, keys_num, 3*dim)
        #     temp_array = tf.matmul(temp_tensor, self.r_kernel)  # (B, L, keys_num, units)
        #     if self.use_bias:
        #         temp_array = tf.nn.bias_add(temp_array, self.r_bias)
        #     score = tf.matmul(tf.tanh(temp_array), self.r_v)  # (B, L, keys_num, 1)
        #
        #   2):
        #     res_s = tf.matmul(s_cat, self.r_s_kernel)    # (B, L, units)
        #     res_k = tf.matmul(key_cat, self.r_k_kernel)  # (B, keys_num, units)
        #
        #     if self.use_bias:
        #         res_s = tf.nn.bias_add(res_s, self.r_s_bias)
        #         res_k = tf.nn.bias_add(res_k, self.r_k_bias)
        #
        #     res_s = self.activation(res_s)
        #     res_k = self.activation(res_k)
        #
        #     score = tf.matmul(res_s, res_k, transpose_b=True)  # (B, L, keys_num)
        #     score = tf.expand_dims(score, axis=-1)  # (B, L, keys_num, 1)
        #
        # #===========

        s = tf.tile(tf.expand_dims(s_cat, axis=2), [1, 1, self.keys_num, 1])  # (B, L, keys_num, dim)
        key = tf.tile(tf.expand_dims(key_cat, axis=1), [1, s1_len + s2_len, 1, 1])  # (B, L, keys_num, 2*dim)

        temp_tensor = tf.concat([s, key], axis=3)  # (B, L, keys_num, 3*dim)
        temp_array = tf.keras.backend.dot(temp_tensor, self.r_kernel)  # (B, L, keys_num, units)
        if self.use_bias:
            temp_array = tf.nn.bias_add(temp_array, self.r_bias)
        score = tf.keras.backend.dot(tf.tanh(temp_array), self.r_v)  # (B, L, keys_num, 1)

        if s_mask is not None:
            s_mask = tf.expand_dims(tf.expand_dims(s_mask, axis=-1), axis=-1)
            score -= (1 - s_mask) * 1e10
        alpha = tf.nn.softmax(score, axis=1)

        new_values = tf.reduce_sum(alpha * s, axis=1)  # (B, keys_nums, dim)

        # <2> GET gate

        # #===========
        # gate function:
        #     1):
        #       temp_tensor = tf.concat([values_tm, new_values, r_h_tm], axis=-1)  # (B, keys_nums, 3*dim)
        #       temp_array = tf.matmul(temp_tensor, self.r_gate_kernel)
        #       if self.use_bias:
        #           temp_array = tf.nn.bias_add(temp_array, self.r_gate_bias)  # bias: (units,)
        #       temp_array = tf.matmul(tf.tanh(temp_array), self.r_gate_v)  # (B, keys_nums, 3*dim) or (B, keys_nums, 1)
        #       gate = tf.sigmoid(temp_array)
        #
        #     2):
        #       temp_tensor = tf.concat([values_tm, new_values, r_h_tm], axis=-1)  # (B, keys_nums, 3*dim)
        #       temp_array = tf.matmul(temp_tensor, self.r_gate_kernel)  # (B, keys_nums, 3*dim) or (B, keys_nums, 1)
        #       if self.use_bias:
        #           temp_array = tf.nn.bias_add(temp_array, self.r_gate_bias)  # bias: (3*dim,) or (1,)
        #       gate = tf.sigmoid(temp_array)
        # #===========

        temp_tensor = tf.concat([values_tm, new_values, r_h_tm], axis=-1)  # (B, keys_nums, 3*dim)
        temp_array = tf.keras.backend.dot(temp_tensor, self.r_gate_kernel)  # (B, keys_nums, 3*dim) or (B, keys_nums, 1)
        if self.use_bias:
            temp_array = tf.nn.bias_add(temp_array, self.r_gate_bias)  # bias: (3*dim,) or (1,)
        gate = tf.sigmoid(temp_array)

        # <3> UPDATE
        values_t = values_tm + gate * new_values
        values = tf.nn.l2_normalize(values_t, axis=2)
        values = tf.reshape(values, [-1, self.keys_num * self.dim])

        r_cell_input = tf.reshape(new_values, [-1, self.dim])
        r_h_tm_reshaped = tf.reshape(r_h_tm, [-1, self.dim])
        r_h_reshaped, _ = self.r_cell(r_cell_input, r_h_tm_reshaped)
        r_h = tf.reshape(r_h_reshaped, [-1, self.keys_num * self.dim])

        return values, r_h

    def pro_compute(self, r1_h, r2_h, r3_h):
        # coming soon
        B = tf.shape(r1_h)[0]

        output = tf.ones([B, 1], dtype=tf.float32)

        return output

    def relation_att(self, vector, relation):
        # vector: (B, dim), relation: (B, keys_num, dim)
        vector_expanded = tf.tile(tf.expand_dims(vector, axis=1), [1, self.keys_num, 1])

        # #==========
        # attention score function
        #   1):
        #     temp_tensor = tf.concat([vector_expanded, relation], axis=-1)
        #     temp_array = tf.matmul(temp_tensor, self.att_r_kernel)  # (B, keys_num, units)
        #     if self.use_bias:
        #         temp_array = tf.nn.bias_add(temp_array, self.att_r_bias)
        #     score = tf.matmul(tf.tanh(temp_array), self.att_r_v)  # (B, keys_num, 1)
        #     alpha = tf.nn.softmax(score, axis=1)
        #
        #     output = tf.reduce_sum(alpha * relation, axis=1)
        #
        #   2):
        #     temp_tensor = tf.concat([vector_expanded, relation], axis=-1)
        #     temp_array = tf.matmul(temp_tensor, self.att_r_kernel)  # (B, keys_num, 1)
        #     if self.use_bias:
        #         temp_array = tf.nn.bias_add(temp_array, self.att_r_bias)
        #     score = self.activation(temp_array)  # (B, keys_num, 1)
        #     alpha = tf.nn.softmax(score, axis=1)
        #
        #     output = tf.reduce_sum(alpha * relation, axis=1)
        #
        #   3):
        #     res0 = tf.matmul(vector, self.att_r_kernel0)  # (B, units)
        #     res1 = tf.matmul(relation, self.att_r_kernel1)  # (B, keys_num, units)
        #     if self.use_bias:
        #         res0 = tf.nn.bias_add(res0, self.att_r_bias0)
        #         res1 = tf.nn.bias_add(res1, self.att_r_bias1)
        #     res0 = self.activation(res0)
        #     res1 = self.activation(res1)
        #
        #     score = tf.matmul(res1, tf.expand_dims(res0, axis=-1))  # (B, keys_num, 1)
        #     alpha = tf.nn.softmax(score, axis=1)
        #
        #     output = tf.reduce_sum(alpha * relation, axis=1)
        #
        # #==========

        batch_size = tf.shape(vector)[0]
        keys = tf.tile(tf.expand_dims(self.keys, axis=0), [batch_size, 1, 1])
        temp_tensor = tf.concat([vector_expanded, keys], axis=-1)
        temp_array = tf.keras.backend.dot(temp_tensor, self.att_r_kernel)  # (B, keys_num, units)
        if self.use_bias:
            temp_array = tf.nn.bias_add(temp_array, self.att_r_bias)
        score = tf.keras.backend.dot(tf.tanh(temp_array), self.att_r_v)  # (B, keys_num, 1)
        alpha = tf.nn.softmax(score, axis=1)

        output = tf.reduce_sum(alpha * relation, axis=1)

        return output

    def sent_att(self, vector, sentence):
        sent_len = sentence.get_shape().as_list()[1]
        vector_expanded = tf.tile(tf.expand_dims(vector, axis=1), [1, sent_len, 1])

        # #==========
        # attention score function
        #   1):
        #     temp_tensor = tf.concat([vector_expanded, sentence], axis=-1)
        #     temp_array = tf.matmul(temp_tensor, self.att_s_kernel)  # (B, keys_num, units)
        #     if self.use_bias:
        #         temp_array = tf.nn.bias_add(temp_array, self.att_s_bias)
        #     score = tf.matmul(tf.tanh(temp_array), self.att_s_v)  # (B, keys_num, 1)
        #     alpha = tf.nn.softmax(score, axis=1)
        #
        #     output = tf.reduce_sum(alpha * sentence, axis=1)
        #
        #   2):
        #     temp_tensor = tf.concat([vector_expanded, sentence], axis=-1)
        #     temp_array = tf.matmul(temp_tensor, self.att_s_kernel)  # (B, keys_num, 1)
        #     if self.use_bias:
        #         temp_array = tf.nn.bias_add(temp_array, self.att_s_bias)
        #     score = self.activation(temp_array)  # (B, keys_num, 1)
        #     alpha = tf.nn.softmax(score, axis=1)
        #
        #     output = tf.reduce_sum(alpha * sentence, axis=1)
        #
        #   3):
        #     res0 = tf.matmul(vector, self.att_s_kernel0)  # (B, units)
        #     res1 = tf.matmul(sentence, self.att_s_kernel1)  # (B, keys_num, units)
        #     if self.use_bias:
        #         res0 = tf.nn.bias_add(res0, self.att_s_bias0)
        #         res1 = tf.nn.bias_add(res1, self.att_s_bias1)
        #     res0 = self.activation(res0)
        #     res1 = self.activation(res1)
        #
        #     score = tf.matmul(res1, tf.expand_dims(res0, axis=-1))  # (B, keys_num, 1)
        #     alpha = tf.nn.softmax(score, axis=1)
        #
        #     output = tf.reduce_sum(alpha * sentence, axis=1)
        #
        # #==========

        temp_tensor = tf.concat([vector_expanded, sentence], axis=-1)
        temp_array = tf.keras.backend.dot(temp_tensor, self.att_s_kernel)  # (B, keys_num, units)
        if self.use_bias:
            temp_array = tf.nn.bias_add(temp_array, self.att_s_bias)
        score = tf.keras.backend.dot(tf.tanh(temp_array), self.att_s_v)  # (B, keys_num, 1)
        alpha = tf.nn.softmax(score, axis=1)

        output = tf.reduce_sum(alpha * sentence, axis=1)

        return output
