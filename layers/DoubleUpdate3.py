import collections
import tensorflow as tf

from tensorflow.contrib.rnn import LayerRNNCell, RNNCell
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.keras import initializers

_DoubleStateTuple = collections.namedtuple("DoubleStateTuple", ("s1", "s2",
                                                                # "values",
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

        self._state_size = DoubleStateTuple(self.sent1_length * self.dim,  # sentences
                                            self.sent2_length * self.dim,

                                            # self.keys_num * self.dim,  # values, relation vectors

                                            self.num_units   # relation hidden states
                                            )

        self._output_size = self.num_units + 1
        # "1" stands for the probability that finally we select this time step as output

    def zero_state(self, batch_size, dtype):
        """Initialize the memory to the key values."""

        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            sent1 = tf.reshape(self.sent1, [-1, self.sent1_length * self.dim])
            sent2 = tf.reshape(self.sent2, [-1, self.sent2_length * self.dim])

            rh = _zero_state_tensors([self.num_units], batch_size, dtype=tf.float32)

            state_list = [sent1, sent2, rh[0]]

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
        # self.matrix_kernel = self.add_variable(name="matrix_kernel",
        #                                        shape=[3 * self.dim + self.num_units, 1],
        #                                        initializer=self.initializer)

        self.att_weight = self.add_variable(name="att_weight",
                                            shape=[self.dim*2, self.num_units],
                                            initializer=self.initializer)
        self.v = self.add_variable(name="v",
                                   shape=[self.num_units, 1],
                                   initializer=self.initializer)

        # self.fuse_kernel = self.add_variable(name="fuse_kernel",
        #                                      shape=[4 * self.dim, self.dim],
        #                                      initializer=self.initializer)
        # self.fuse_kernel1 = self.fuse_kernel[:2*self.dim]
        # self.fuse_kernel2 = self.fuse_kernel[2*self.dim:]

        # all bias
        if self.use_bias:
            # self.matrix_bias = self.add_variable(name="matrix_bias",
            #                                      shape=[1],
            #                                      initializer=tf.constant_initializer(bias_init, dtype=tf.float32))

            self.att_bias = self.add_variable(name="att_bias",
                                                   shape=[3 * self.dim + self.num_units, 1],
                                                   initializer=self.initializer)

            # self.fuse_bias = self.add_variable(name="fuse_bias",
            #                                    shape=[2 * self.dim],
            #                                    initializer=tf.constant_initializer(bias_init, dtype=tf.float32))
            # self.fuse_bias1 = self.fuse_bias[:self.dim]
            # self.fuse_bias2 = self.fuse_bias[self.dim:]

        self.built = True

    def call(self, inputs, state=None):
        s1_tm, s2_tm, rh_tm = state
        # s: (B, Lx * dim), values: (B, keys_num * dim),
        # r_h: (B, dim)

        # updating sentence with rh_tm
        s1, s2, rh = self.update_sentence(s1_tm, self.sent1_length, self.sent1_mask,
                                          s2_tm, self.sent2_length, self.sent2_mask,
                                          rh_tm)

        score = self.pro_compute(rh)  # (B, 1)
        m = tf.concat([rh, score], axis=1)  # (B, rdim + 1)

        state = [s1, s2, rh]

        return m, DoubleStateTuple(*state)

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

    def attention(self, sent, sent_mask, r_vec):
        r_vec_exp = tf.tile(tf.expand_dims(r_vec, axis=1),
                            [1, sent.get_shape().as_list()[1], 1])
        sent_cat = tf.concat([sent, r_vec_exp], axis=2)
        temp_array = tf.keras.backend.dot(sent_cat, self.att_weight)
        if self.use_bias:
            temp_array = tf.nn.bias_add(temp_array, self.att_bias)
        temp_array = self.activation(temp_array)

        score = tf.squeeze(tf.keras.backend.dot(temp_array,  self.v), 2)
        if sent_mask is not None:
            score -= (1 - tf.expand_dims(self.C_mask, 2)) * 10000

        # s_value, s_indices = tf.nn.top_k(score, k=10, sorted=False)  # (B, k), (B, k)
        # sent_select = tf.batch_gather(sent, s_indices)
        alpha = tf.expand_dims(tf.nn.softmax(score, 1), 2)
        sent_vec = tf.reduce_sum(alpha * sent, 1)

        return sent_vec

    def update_sentence(self,
                        s1, s1_len, s1_mask,
                        s2, s2_len, s2_mask,
                        rh_tm):
        # values_tm: (B, keys_num * dim),
        # s1, s2: (B, Lx * dim);  s1_len, s2_len: scalar;  s1_mask, s2_mask: (B, Lx) or None
        # rh_tm: (B, dim)

        s1 = tf.reshape(s1, [-1, s1_len, self.dim])                       # (B, L1, dim)
        s2 = tf.reshape(s2, [-1, s2_len, self.dim])                       # (B, L2, dim)

        # <1> Q_vec

        s1_vec = self.attention(s1, s1_mask, rh_tm)
        s2_vec = self.attention(s2, s2_mask, s1_vec)

        temp = tf.concat([s1_vec, s2_vec], axis=1)
        rh, _ = self.r_cell(temp, rh_tm)

        # <3> UPDATE

        s1 = tf.reshape(s1, [-1, s1_len * self.dim])
        s2 = tf.reshape(s2, [-1, s2_len * self.dim])

        return s1, s2, rh

    def pro_compute(self, rh):
        # coming soon
        B = tf.shape(rh)[0]

        output = tf.ones([B, 1], dtype=tf.float32)

        return output

