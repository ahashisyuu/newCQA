import collections
import tensorflow as tf

from tensorflow.contrib.rnn import LayerRNNCell, RNNCell
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.keras import initializers

_TriangularStateTuple = collections.namedtuple("TriangularStateTuple", ("s1", "s2", "s3",
                                                                        "r1_h", "r2_h", "r3_h"))


class TriangularStateTuple(_TriangularStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        return self[0].dtype


class TriangularCell(LayerRNNCell):
    def __init__(self,
                 num_units,
                 r_cell: RNNCell,
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
        self.r_cell = r_cell

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

        # self._state_size = TriangularStateTuple(self.sent1_length * self.dim,  # sentences
        #                                         self.sent2_length * self.dim,
        #                                         self.sent3_length * self.dim,
        #
        #                                         self.num_units,  # relation hidden states
        #                                         self.num_units,
        #                                         self.num_units)
        self._state_size = [self.sent1_length * self.dim,  # sentences
                            self.sent2_length * self.dim,
                            self.sent3_length * self.dim,

                            self.num_units,  # relation hidden states
                            self.num_units,
                            self.num_units]

        self._output_size = self.num_units * 3 + 1
        # "1" stands for the probability that finally we select this time step as output

    def zero_state(self, batch_size, dtype):
        """Initialize the memory to the key values."""

        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            sent1 = tf.reshape(self.sent1, [-1, self.sent1_length * self.dim])
            sent2 = tf.reshape(self.sent2, [-1, self.sent2_length * self.dim])
            sent3 = tf.reshape(self.sent3, [-1, self.sent3_length * self.dim])

            # rh = _zero_state_tensors([self.num_units]*3, batch_size, dtype)
            rh = [tf.tile(tf.expand_dims(self.keys[i], axis=0), [batch_size, 1])
                  for i in range(3)]

            state_list = [sent1, sent2, sent3] + rh

            return state_list

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def build(self, inputs_shape):
        bias_init = 1.0  # or 0.0
        # parameters which relation updating needs
        self.matrix_kernel = self.add_variable(name="matrix_kernel",
                                               shape=[3 * self.dim + self.num_units, 1],
                                               initializer=self.initializer)

        # self.fuse_kernel = self.add_variable(name="fuse_kernel",
        #                                      shape=[4 * self.dim, self.dim],
        #                                      initializer=self.initializer)
        # self.fuse_kernel1 = self.fuse_kernel[:2*self.dim]
        # self.fuse_kernel2 = self.fuse_kernel[2*self.dim:]
        self.r_kernel = self.add_variable(name="r_kernel",
                                          shape=[2 * self.dim, self.num_units])

        # all bias
        if self.use_bias:
            self.matrix_bias = self.add_variable(name="matrix_bias",
                                                 shape=[1],
                                                 initializer=tf.constant_initializer(bias_init, dtype=tf.float32))

            # self.fuse_bias = self.add_variable(name="fuse_bias",
            #                                    shape=[2 * self.dim],
            #                                    initializer=tf.constant_initializer(bias_init, dtype=tf.float32))
            # self.fuse_bias1 = self.fuse_bias[:self.dim]
            # self.fuse_bias2 = self.fuse_bias[self.dim:]
            self.r_bias = self.add_variable(name="r_bias",
                                            shape=[self.num_units],
                                            initializer=tf.constant_initializer(bias_init, dtype=tf.float32))

        self.built = True

    def call(self, inputs, state=None):
        s1_tm, s2_tm, s3_tm, \
            r1_h_tm, r2_h_tm, r3_h_tm = state
        # s: (B, Lx * dim), values: (B, keys_num * dim),
        # r_h: (B, keys_num * dim)

        # updating sentence with rh_tm
        s1_2, s2_1 = self.update_sentence(s1_tm, self.sent1_length, self.sent1_mask,
                                          s2_tm, self.sent2_length, self.sent2_mask,
                                          r1_h_tm)
        s2_3, s3_2 = self.update_sentence(s2_tm, self.sent2_length, self.sent2_mask,
                                          s3_tm, self.sent3_length, self.sent3_mask,
                                          r2_h_tm)
        s3_1, s1_3 = self.update_sentence(s3_tm, self.sent3_length, self.sent3_mask,
                                          s1_tm, self.sent1_length, self.sent1_mask,
                                          r3_h_tm)

        s1 = (s1_2 + s1_3) / 2.0
        s2 = (s2_1 + s2_3) / 2.0
        s3 = (s3_1 + s3_2) / 2.0

        # updating relation
        s1_vec = tf.reduce_max(s1, axis=1)
        s2_vec = tf.reduce_max(s2, axis=1)
        s3_vec = tf.reduce_max(s3, axis=1)

        r1_h = self.update_relation(s1_vec, s2_vec, r1_h_tm)
        r2_h = self.update_relation(s2_vec, s3_vec, r2_h_tm)
        r3_h = self.update_relation(s3_vec, s1_vec, r3_h_tm)

        output = tf.concat([r1_h, r2_h, r3_h], axis=1)  # (B, 3 * units)
        score = self.pro_compute(r1_h, r2_h, r3_h)  # (B, 1)
        m = tf.concat([output, score], axis=1)

        s1 = tf.reshape(s1, [-1, self.sent1_length * self.dim])
        s2 = tf.reshape(s2, [-1, self.sent2_length * self.dim])
        s3 = tf.reshape(s3, [-1, self.sent3_length * self.dim])

        state = [s1, s2, s3, r1_h, r2_h, r3_h]

        return m, state

    @staticmethod
    def top_k_att(matrix, sentence, k=1, transpose=False):
        """
        sentence: (B, L1, L2, dim)

        return: (B, L1, dim)

        NOTE: if transpose is False, return (B, L2, dim)

        """
        matrix = tf.squeeze(matrix, axis=3)
        if transpose is True:
            matrix = tf.transpose(matrix, perm=[0, 2, 1])
            sentence = tf.transpose(sentence, perm=[0, 2, 1, 3])
        values, indices = tf.nn.top_k(matrix, k=k, sorted=False)  # (B, L, k)
        s_select = tf.batch_gather(sentence, indices)
        alpha = tf.expand_dims(tf.nn.softmax(values, 2), 3)
        return tf.reduce_sum(alpha * s_select, axis=2)  # (B, L, DIM)

    def update_sentence(self,
                        s1, s1_len, s1_mask,
                        s2, s2_len, s2_mask,
                        rh_tm):
        # values_tm: (B, keys_num * dim),
        # s1, s2: (B, Lx * dim);  s1_len, s2_len: scalar;  s1_mask, s2_mask: (B, Lx) or None
        # rh_tm: (B, dim)

        s1 = tf.reshape(s1, [-1, s1_len, self.dim])                       # (B, L1, dim)
        s2 = tf.reshape(s2, [-1, s2_len, self.dim])                       # (B, L2, dim)

        # <1>
        s1_exp = tf.expand_dims(s1, axis=2)
        s2_exp = tf.expand_dims(s2, axis=1)
        rh_tm_exp = tf.expand_dims(tf.expand_dims(rh_tm, axis=1), axis=1)

        s1_tile = tf.tile(s1_exp, [1, 1, s2_len, 1])
        s2_tile = tf.tile(s2_exp, [1, s1_len, 1, 1])
        rh_tm_tile = tf.tile(rh_tm_exp, [1, s1_len, s2_len, 1])

        infor_cat = tf.concat([s1_tile, s2_tile, s1_tile * s2_tile, rh_tm_tile], axis=-1)   # (B, L1, L2, 3*dim + rdim)
        res_matrix = tf.keras.backend.dot(infor_cat, self.matrix_kernel)

        if self.use_bias:
            res_matrix = tf.nn.bias_add(res_matrix, self.matrix_bias)

        res_matrix = self.activation(res_matrix)

        if s1_mask is not None:
            s1_mask_exp = tf.expand_dims(tf.expand_dims(s1_mask, axis=2), axis=2)  # (B, L1, 1, 1)
            s2_mask_exp = tf.expand_dims(tf.expand_dims(s2_mask, axis=2), axis=1)  # (B, 1, L2, 1)
            res_matrix1 = res_matrix - (1 - s1_mask_exp) * 1e20
            res_matrix2 = res_matrix - (1 - s2_mask_exp) * 1e20
        else:
            res_matrix1 = res_matrix
            res_matrix2 = res_matrix

        # adding TOP K
        m_s2 = self.top_k_att(res_matrix1, s1_tile, k=10, transpose=True)
        m_s1 = self.top_k_att(res_matrix2, s2_tile, k=10, transpose=False)

        # score1 = tf.nn.softmax(res_matrix1, axis=1)
        # score2 = tf.nn.softmax(res_matrix2, axis=2)
        #
        # m_s2 = tf.reduce_sum(score1 * s1_tile, axis=1)  # (B, L2, dim)
        # m_s1 = tf.reduce_sum(score2 * s2_tile, axis=2)  # (B, L1, dim)

        # # <2>[]
        # s1_cat = tf.concat([m_s1 - s1, m_s1 * s1], axis=2)
        # s2_cat = tf.concat([m_s2 - s2, m_s2 * s2], axis=2)
        # m_s1 = tf.keras.backend.dot(s1_cat, self.fuse_kernel1)
        # m_s2 = tf.keras.backend.dot(s2_cat, self.fuse_kernel2)
        #
        # if self.use_bias:
        #     m_s1 = tf.nn.bias_add(m_s1, self.fuse_bias1)
        #     m_s2 = tf.nn.bias_add(m_s2, self.fuse_bias2)
        #
        # m_s1 = self.activation(m_s1)
        # m_s2 = self.activation(m_s2)

        if s1_mask is not None:
            m_s1 = m_s1 * tf.expand_dims(s1_mask, axis=2)
            m_s2 = m_s2 * tf.expand_dims(s2_mask, axis=2)

        # n_s1 = tf.reshape(m_s1, [-1, s1_len * self.dim])
        # n_s2 = tf.reshape(m_s2, [-1, s2_len * self.dim])

        return m_s1, m_s2

    def update_relation(self, s1_vec, s2_vec, rh_tm):
        # s1_vec, s2_vec: (B, dim)
        # rh_tm: (B, units)

        # <1> GET medium
        vec_cat = tf.concat([s1_vec, s2_vec], axis=1)  # (B, 2dim)
        medium = tf.matmul(vec_cat, self.r_kernel)  # (2dim, units) -> (B, UNITS)
        if self.use_bias:
            medium = tf.nn.bias_add(medium, self.r_bias)
        medium = self.activation(medium)

        # <2> UPDATE
        rh, _ = self.r_cell(medium, rh_tm)

        return rh

    def pro_compute(self, r1_h, r2_h, r3_h):
        # coming soon
        B = tf.shape(r1_h)[0]

        output = tf.ones([B, 1], dtype=tf.float32)

        return output

