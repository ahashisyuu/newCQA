import tensorflow as tf

from .CQAModel_margin import CQAModel
from layers.BiGRU import NativeGRU as BiGRU
import keras.losses


class Baseline4_margin(CQAModel):
    def MLP(self, Q_vec, C_vec, reuse=None):
        info = tf.concat([Q_vec, C_vec, Q_vec * C_vec], axis=1)
        median = tf.layers.dense(info, 300, activation=tf.tanh, name='median', reuse=reuse)
        output = tf.layers.dense(median, 1, activation=tf.sigmoid, name='output', reuse=reuse)
        return output

    def build_model(self):
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, posC, negC = self.QS, self.pos_CT, self.neg_CT
            Q_len, posC_len, negC_len = self.Q_len, self.PosC_len, self.NegC_len

            with tf.variable_scope('encode'):
                rnn1 = BiGRU(num_layers=1, num_units=units,
                             batch_size=tf.shape(self.QS)[0], input_size=self.QS.get_shape()[-1],
                             keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='q')
                rnn2 = BiGRU(num_layers=1, num_units=units,
                             batch_size=tf.shape(self.QS)[0], input_size=self.QS.get_shape()[-1],
                             keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='c')
                Q_sequence = rnn1(Q, seq_len=Q_len, return_type=1)

                posC_sequence = rnn2(posC, seq_len=posC_len, return_type=1)
                negC_sequence = rnn2(negC, seq_len=negC_len, return_type=1)

            with tf.variable_scope('interaction'):
                F1 = tf.layers.dense(Q_sequence, units=1, activation=tf.tanh, name='inter_dense')

                posF2 = tf.layers.dense(posC_sequence, units=1, activation=tf.tanh, name='inter_dense', reuse=True)
                negF2 = tf.layers.dense(negC_sequence, units=1, activation=tf.tanh, name='inter_dense', reuse=True)

                pos_matrix = tf.keras.backend.batch_dot(F1, posF2, axes=[2, 2])  # (B, L1, L2)
                neg_matrix = tf.keras.backend.batch_dot(F1, negF2, axes=[2, 2])  # (B, L1, L2)

                pos_m2 = tf.nn.softmax(pos_matrix, axis=2)
                neg_m2 = tf.nn.softmax(neg_matrix, axis=2)

                pos_m1 = tf.nn.softmax(pos_matrix, axis=1)
                neg_m1 = tf.nn.softmax(neg_matrix, axis=1)

                Q_pos = tf.keras.backend.batch_dot(pos_m2, posC_sequence, axes=[2, 1])
                Q_neg = tf.keras.backend.batch_dot(neg_m2, negC_sequence, axes=[2, 1])

                C_pos = tf.keras.backend.batch_dot(pos_m1, Q_sequence, axes=[1, 1])
                C_neg = tf.keras.backend.batch_dot(neg_m1, Q_sequence, axes=[1, 1])

            Q_vec_pos = tf.reduce_mean(Q_pos, axis=1)
            Q_vec_neg = tf.reduce_mean(Q_neg, axis=1)

            C_vec_pos = tf.reduce_mean(C_pos, axis=1)
            C_vec_neg = tf.reduce_mean(C_neg, axis=1)

            # pos_output = self.MLP(Q_vec_pos, C_vec_pos)
            # neg_output = self.MLP(Q_vec_neg, C_vec_neg, reuse=True)

            pos_output = tf.reduce_sum(tf.nn.l2_normalize(Q_vec_pos, 1) * tf.nn.l2_normalize(C_vec_pos, 1), 1)
            neg_output = tf.reduce_sum(tf.nn.l2_normalize(Q_vec_neg, 1) * tf.nn.l2_normalize(C_vec_neg, 1), 1)

            return pos_output, neg_output

