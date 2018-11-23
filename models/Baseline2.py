import tensorflow as tf

from .CQAModel import CQAModel
from layers.BiGRU import NativeGRU as BiGRU
import keras.losses


class Baseline2(CQAModel):
    def build_model(self):
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, C = self.QS, self.CT
            Q_len, C_len = self.Q_len, self.C_len

            with tf.variable_scope('encode'):
                rnn = BiGRU(num_layers=1, num_units=units,
                            batch_size=tf.shape(self.QS)[0], input_size=self.CT.get_shape()[-1],
                            keep_prob=self.dropout_keep_prob, is_train=self._is_train)

                Q_sequence = rnn(Q, seq_len=Q_len, return_type=1)
                C_sequence = rnn(C, seq_len=C_len, return_type=1)

            with tf.variable_scope('conv'):
                Q_conv = tf.layers.conv1d(Q_sequence, padding='same', filters=units, kernel_size=5, name='con1d_Q')
                C_conv = tf.layers.conv1d(C_sequence, padding='same', filters=units, kernel_size=5, name='con1d_C')

                Q_w = tf.layers.conv1d(Q_sequence, padding='same', filters=1, kernel_size=5, name='con1d_wQ')
                C_w = tf.layers.conv1d(C_sequence, padding='same', filters=1, kernel_size=5, name='con1d_wC')
                Q_w = tf.nn.softmax(Q_w, axis=1)
                C_w = tf.nn.softmax(C_w, axis=1)
                # Q_w = tf.squeeze(Q_w, axis=-1)
                # C_w = tf.squeeze(C_w, axis=-1)
                #
                # parse, indice = tf.nn.top_k(Q_w, 10)
                # weight = parse / tf.reduce_sum(parse, axis=1, keepdims=True)
                # vector = tf.batch_gather(Q_conv, indice)
                # Q_vec = tf.reduce_sum(tf.expand_dims(weight, axis=-1) * vector, axis=1)
                #
                # parse, indice = tf.nn.top_k(C_w, 10)
                # weight = parse / tf.reduce_sum(parse, axis=1, keepdims=True)
                # vector = tf.batch_gather(C_conv, indice)
                # C_vec = tf.reduce_sum(tf.expand_dims(weight, axis=-1) * vector, axis=1)
                Q_vec = tf.reduce_sum(Q_conv * Q_w, axis=1)
                C_vec = tf.reduce_sum(C_conv * C_w, axis=1)

            info = tf.concat([Q_vec, C_vec, Q_vec * C_vec], axis=1)
            median = tf.layers.dense(info, 300, activation=tf.tanh)
            output = tf.layers.dense(median, self.args.categories_num, activation=tf.identity)

            return output

