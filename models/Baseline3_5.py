import tensorflow as tf

from .CQAModel import CQAModel
from layers.BiGRU import NativeGRU as BiGRU
import keras.losses


class Baseline3_5(CQAModel):
    def build_model(self):
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, C = self.QS, self.CT
            Q_len, C_len = self.Q_len, self.C_len

            # with tf.variable_scope('encode'):
            #     rnn1 = BiGRU(num_layers=1, num_units=units,
            #                  batch_size=tf.shape(self.QS)[0], input_size=self.CT.get_shape()[-1],
            #                  keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='q')
            #     rnn2 = BiGRU(num_layers=1, num_units=units,
            #                  batch_size=tf.shape(self.QS)[0], input_size=self.CT.get_shape()[-1],
            #                  keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='c')
            #     Q_sequence = rnn1(Q, seq_len=Q_len, return_type=1)
            #     C_sequence = rnn2(C, seq_len=C_len, return_type=1)
            #     # C_sequence = tf.reshape(C_sequence, [-1, self.C_maxlen, 2*units])

            # Q_vec = tf.reduce_mean(Q_sequence, axis=1)  # (B, 2H)
            Q_sequence = Q
            C_sequence = C

            v = tf.get_variable('v', [units, 1], tf.float32)
            with tf.variable_scope('q_attention'):
                cate_att = tf.tile(tf.expand_dims(self.cate_f, 1), [1, self.Q_maxlen, 1])
                att_pre = tf.layers.dense(tf.concat([cate_att, Q_sequence], 2), units, tf.tanh)
                score = tf.keras.backend.dot(att_pre, v)
                alpha = tf.nn.softmax(score, 1)
                Q_vec = tf.reduce_sum(alpha * Q_sequence, 1)

            with tf.variable_scope('attention'):
                q_att = tf.tile(tf.expand_dims(Q_vec, 1), [1, self.C_maxlen, 1])
                att_pre = tf.layers.dense(tf.concat([q_att, C_sequence], axis=2), units, tf.tanh)

                score = tf.squeeze(tf.keras.backend.dot(att_pre, v), 2)  # (B, L)
                # score -= (1 - tf.expand_dims(self.C_mask, 2)) * 10000
                s_value, s_indices = tf.nn.top_k(score, k=10, sorted=False)  # (B, k), (B, k)
                C_select = tf.batch_gather(C_sequence, s_indices)
                alpha = tf.expand_dims(tf.nn.softmax(s_value, 1), 2)

                C_vec = tf.reduce_sum(alpha * C_select, 1)

            info = tf.concat([Q_vec, C_vec, Q_vec * C_vec], axis=1)
            median = tf.layers.dense(info, 300, activation=tf.tanh)
            output = tf.layers.dense(median, self.args.categories_num, activation=tf.identity)

            return output

