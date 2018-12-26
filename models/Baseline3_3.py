import tensorflow as tf

from .CQAModel import CQAModel
from layers.BiGRU import NativeGRU as BiGRU
import keras.losses


class Baseline3_3(CQAModel):
    def build_model(self):
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, C = self.QS, self.CT
            Q_len, C_len = self.Q_len, self.C_len

            with tf.variable_scope('encode'):
                rnn1 = BiGRU(num_layers=1, num_units=units,
                             batch_size=tf.shape(self.QS)[0], input_size=self.CT.get_shape()[-1],
                             keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='q')
                rnn2 = BiGRU(num_layers=1, num_units=units,
                             batch_size=tf.shape(self.QS)[0], input_size=self.CT.get_shape()[-1],
                             keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='c')
                Q_sequence = rnn1(Q, seq_len=Q_len, return_type=1)
                C_sequence = rnn2(C, seq_len=C_len, return_type=1)

            Q_vec = tf.reduce_mean(Q_sequence, axis=1)  # (B, 2H)

            with tf.variable_scope('attention'):
                q_att = tf.tile(tf.expand_dims(Q_vec, 1), [1, self.C_maxlen, 1])
                att_pre = tf.layers.dense(tf.concat([q_att, C_sequence], axis=2), units, tf.tanh)
                v = tf.get_variable('v', [units, 1], tf.float32)
                score = tf.keras.backend.dot(att_pre, v)  # (B, L, 1)
                # score -= (1 - tf.expand_dims(self.C_mask, 2)) * 10000
                alpha = tf.nn.softmax(score, 1)
                C_vec = tf.reduce_sum(alpha * C_sequence, 1)

            info = tf.concat([Q_vec, C_vec, Q_vec * C_vec], axis=1)
            median = tf.layers.dense(info, 300, activation=tf.tanh)
            output = tf.layers.dense(median, self.args.categories_num, activation=tf.identity)

            return output

