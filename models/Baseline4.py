import tensorflow as tf

from .CQAModel import CQAModel
from layers.BiGRU import NativeGRU as BiGRU
import keras.losses


class Baseline4(CQAModel):
    def dropout_func(self, tensor):
        return tf.cond(self._is_train, lambda: tf.nn.dropout(tensor, self.dropout_keep_prob), lambda: tensor)

    def build_model(self):
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):
            units = 256
            Q, C = self.QS, self.CT
            Q_len, C_len = self.Q_len, self.C_len

            with tf.variable_scope('encode'):
                rnn1 = BiGRU(num_layers=1, num_units=units,
                             batch_size=tf.shape(self.QS)[0], input_size=self.CT.get_shape()[-1],
                             keep_prob=0.8, is_train=self._is_train, scope='q')
                rnn2 = BiGRU(num_layers=1, num_units=units,
                             batch_size=tf.shape(self.QS)[0], input_size=self.CT.get_shape()[-1],
                             keep_prob=0.8, is_train=self._is_train, scope='c')
                Q_sequence = rnn1(Q, seq_len=Q_len, return_type=1)
                C_sequence = rnn2(C, seq_len=C_len, return_type=1)

            with tf.variable_scope('interaction'):
                # Q_ = tf.expand_dims(Q_sequence, axis=2)  # (B, L1, 1, dim)
                # C_ = tf.expand_dims(C_sequence, axis=1)  # (B, 1, L2, dim)
                Q_sequence = self.dropout_func(Q_sequence)
                C_sequence = self.dropout_func(C_sequence)
                F1 = tf.layers.dense(Q_sequence, units=1, activation=tf.tanh, name='inter_dense')
                F2 = tf.layers.dense(C_sequence, units=1, activation=tf.tanh, name='inter_dense', reuse=True)
                matrix = tf.keras.backend.batch_dot(F1, F2, axes=[2, 2])  # (B, L1, L2)

                m2 = tf.nn.softmax(matrix, axis=2)
                m1 = tf.nn.softmax(matrix, axis=1)
                Q_ = tf.keras.backend.batch_dot(m2, C_sequence, axes=[2, 1])
                C_ = tf.keras.backend.batch_dot(m1, Q_sequence, axes=[1, 1])

            Q_vec = tf.reduce_mean(Q_, axis=1)
            C_vec = tf.reduce_mean(C_, axis=1)

            info = tf.concat([Q_vec, C_vec, Q_vec * C_vec], axis=1)
            info = tf.cond(self._is_train, lambda: tf.nn.dropout(info, 0.7), lambda: info)
            median = tf.layers.dense(info, units, activation=tf.tanh)
            output = tf.layers.dense(median, self.args.categories_num, activation=tf.identity)

            return output

