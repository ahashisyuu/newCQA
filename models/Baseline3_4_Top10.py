import tensorflow as tf

from .CQAModelTop10 import CQAModel
from layers.BiGRU import NativeGRU as BiGRU
import keras.losses


class Baseline3_4_Top10(CQAModel):
    def build_model(self):
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, C = self.QS, self.CT
            Q_len, C_len = self.Q_len, self.C_len

            C = tf.reshape(C, [-1, self.C_maxlen, 300])
            C_len = tf.reshape(C_len, [-1])

            with tf.variable_scope('encode'):
                rnn1 = BiGRU(num_layers=1, num_units=units,
                             batch_size=tf.shape(self.QS)[0], input_size=self.CT.get_shape()[-1],
                             keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='q')
                rnn2 = BiGRU(num_layers=1, num_units=units,
                             batch_size=tf.shape(C)[0], input_size=C.get_shape()[-1],
                             keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='c')
                Q_sequence = rnn1(Q, seq_len=Q_len, return_type=1)
                C_sequence = rnn2(C, seq_len=C_len, return_type=1)
                C_sequence = tf.reshape(C_sequence, [-1, 10, self.C_maxlen, 2*units])

            Q_vec = tf.reduce_mean(Q_sequence, axis=1)  # (B, 2H)
            Q_vec = tf.tile(tf.expand_dims(Q_vec, 1), [1, 10, 1])  # (B, 10, 2H)

            with tf.variable_scope('attention'):
                q_att = tf.tile(tf.expand_dims(Q_vec, 2), [1, 1, self.C_maxlen, 1])
                att_pre = tf.layers.dense(tf.concat([q_att, C_sequence], axis=3), units, tf.tanh)
                v = tf.get_variable('v', [units, 1], tf.float32)
                score = tf.squeeze(tf.keras.backend.dot(att_pre, v), 3)  # (B, 10, L)
                # score -= (1 - tf.expand_dims(self.C_mask, 2)) * 10000
                print(score, C_sequence)
                s_value, s_indices = tf.nn.top_k(score, k=10, sorted=False)  # (B, 10, k), (B, 10, k)
                C_select = tf.batch_gather(C_sequence, s_indices)
                alpha = tf.expand_dims(tf.nn.softmax(s_value, 2), 3)
                print(alpha)
                C_vec = tf.reduce_sum(alpha * C_select, 2)  # (B, 10, dim)

            print(Q_vec, C_vec)
            info = tf.concat([Q_vec, C_vec, Q_vec * C_vec], axis=2)
            median = tf.layers.dense(info, 300, activation=tf.tanh)
            output = tf.layers.dense(median, self.args.categories_num, activation=tf.identity)  # (B, 10, 2)

            return output

