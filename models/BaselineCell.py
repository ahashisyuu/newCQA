import tensorflow as tf

from layers.DoubleUpdate import DoubleCell
from .CQAModel import CQAModel
from layers.BiGRU import NativeGRU as BiGRU
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.contrib.rnn import DropoutWrapper


class Baseline3_4(CQAModel):
    def build_model(self):
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, C = self.QS, self.CT
            Q_len, C_len = self.Q_len, self.C_len
            batch_size = tf.shape(self.QS)[0]

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

            with tf.variable_scope("inferring_module"):
                dim = 2 * units
                sr_cell = GRUCell(num_units=dim, activation=tf.nn.relu)

                if self.is_training:
                    sr_cell = DropoutWrapper(cell=sr_cell,
                                             input_keep_prob=self.dropout_keep_prob,
                                             output_keep_prob=self.dropout_keep_prob,
                                             state_keep_prob=self.dropout_keep_prob,
                                             variational_recurrent=True,
                                             input_size=dim,
                                             dtype=tf.float32)
                sent_cell = r_cell = sr_cell

                # sent_transformer = self.sent_transformer(hidden_size=dim)
                # highway = MultiLayerHighway(dim, 1, keep_prob=1.0, is_train=is_train)

                def _trans(_sent, _mask):
                    # return highway(_sent)
                    return _sent

                sent_transformer = _trans

                tri_cell = DoubleCell(num_units=256,
                                      sent_cell=sent_cell, r_cell=r_cell, sentence_transformer=sent_transformer,
                                      sent1=self.sent1, sent2=self.sent2,
                                      sent1_length=s1_len,
                                      sent2_length=s2_len,
                                      dim=self.encoding_size,
                                      keys=self._keys_embedding,
                                      sent1_mask=self.mask1, sent2_mask=self.mask2,
                                      initializer=None, dtype=tf.float32)

                fake_input = tf.tile(tf.expand_dims(self.mark0, axis=1), [1, self.config.update_num, 1])
                init_state = tri_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

                if self.is_training:
                    tri_cell = DropoutWrapper(cell=tri_cell,
                                              input_keep_prob=self.dropout_keep_prob,
                                              output_keep_prob=self.dropout_keep_prob,
                                              state_keep_prob=self.dropout_keep_prob,
                                              variational_recurrent=True,
                                              input_size=fake_input.get_shape()[2],
                                              dtype=tf.float32)

                output, last_state = dynamic_rnn(cell=tri_cell,
                                                 inputs=fake_input,
                                                 initial_state=init_state)
                self.output = [tf.reshape(a, [-1, self.config.keys_num, self.encoding_size]) for a in last_state[3:6]]


            Q_vec = tf.reduce_mean(Q_sequence, axis=1)  # (B, 2H)

            with tf.variable_scope('attention'):
                q_att = tf.tile(tf.expand_dims(Q_vec, 1), [1, self.C_maxlen, 1])
                att_pre = tf.layers.dense(tf.concat([q_att, C_sequence], axis=2), units, tf.tanh)
                v = tf.get_variable('v', [units, 1], tf.float32)
                score = tf.squeeze(tf.keras.backend.dot(att_pre, v), 2)  # (B, L)
                # score -= (1 - tf.expand_dims(self.C_mask, 2)) * 10000
                print(score, C_sequence)
                s_value, s_indices = tf.nn.top_k(score, k=10, sorted=False)  # (B, k), (B, k)
                C_select = tf.batch_gather(C_sequence, s_indices)
                alpha = tf.expand_dims(tf.nn.softmax(s_value, 1), 2)
                print(alpha)
                C_vec = tf.reduce_sum(alpha * C_select, 1)

            info = tf.concat([Q_vec, C_vec, Q_vec * C_vec], axis=1)
            median = tf.layers.dense(info, 300, activation=tf.tanh)
            output = tf.layers.dense(median, self.args.categories_num, activation=tf.identity)

            return output

