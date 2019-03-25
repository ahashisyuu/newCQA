import tensorflow as tf

from layers.DoubleUpdate2 import DoubleCell
from .CQAModel import CQAModel
from layers.BiGRU import NativeGRU as BiGRU
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.contrib.rnn import DropoutWrapper


def wrapper(is_train, sr_cell, keep_prob, dim):
    def _true_func():
        _sr_cell = DropoutWrapper(cell=sr_cell,
                                  input_keep_prob=keep_prob,
                                  output_keep_prob=keep_prob,
                                  state_keep_prob=keep_prob,
                                  variational_recurrent=True,
                                  input_size=dim,
                                  dtype=tf.float32)
        return _sr_cell

    def _false_func():
        return sr_cell

    return tf.cond(is_train, _true_func, _false_func)


class BaselineCell2(CQAModel):
    def build_model(self):
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, C = self.QS, self.CT
            Q_len, C_len = self.Q_len, self.C_len
            batch_size = tf.shape(self.QS)[0]

            update_num = 2
            keys_num = 6

            with tf.variable_scope('encode'):
                rnn1 = BiGRU(num_layers=1, num_units=units,
                             batch_size=batch_size, input_size=self.CT.get_shape()[-1],
                             keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='q')
                rnn2 = BiGRU(num_layers=1, num_units=units,
                             batch_size=batch_size, input_size=self.CT.get_shape()[-1],
                             keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='c')
                Q_sequence = rnn1(Q, seq_len=Q_len, return_type=1)
                C_sequence = rnn2(C, seq_len=C_len, return_type=1)
                # C_sequence = tf.reshape(C_sequence, [-1, self.C_maxlen, 2*units])

            with tf.variable_scope("memory"):
                rdim = 256
                # keys_embedding = tf.get_variable("keys_embedding",
                #                                  [rdim],
                #                                  dtype=tf.float32,
                #                                  trainable=True)

            with tf.variable_scope("inferring_module"):
                sr_cell = GRUCell(num_units=rdim, activation=tf.nn.relu)

                # sr_cell = wrapper(self._is_train, sr_cell, self.dropout_keep_prob, dim)
                sent_cell = r_cell = sr_cell

                # sent_transformer = self.sent_transformer(hidden_size=dim)
                # highway = MultiLayerHighway(dim, 1, keep_prob=1.0, is_train=is_train)

                def _trans(_sent, _mask):
                    # return highway(_sent)
                    return _sent

                sent_transformer = _trans

                tri_cell = DoubleCell(num_units=rdim,
                                      sent_cell=sent_cell, r_cell=r_cell, sentence_transformer=sent_transformer,
                                      sent1=Q_sequence, sent2=C_sequence,
                                      sent1_length=self.Q_maxlen,
                                      sent2_length=self.C_maxlen,
                                      dim=2*units,
                                      use_bias=False, activation=tf.nn.selu,
                                      keys=None,
                                      sent1_mask=None, sent2_mask=None,
                                      initializer=None, dtype=tf.float32)

                fake_input = tf.tile(tf.expand_dims(Q_sequence[:, 0, :], axis=1), [1, update_num, 1])
                self.init_state = tri_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

                # print(fake_input)
                # tri_cell = wrapper(self._is_train, tri_cell, self.dropout_keep_prob, dim)

                self.double_output, last_state = dynamic_rnn(cell=tri_cell,
                                                             inputs=fake_input,
                                                             initial_state=self.init_state)
                refer_output = last_state[2]  # (B, dim)

            info = refer_output
            median = tf.layers.dense(info, 300, activation=tf.tanh)
            output = tf.layers.dense(median, self.args.categories_num, activation=tf.identity)

            return output

