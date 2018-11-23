import tensorflow as tf

from .CQAModel import CQAModel
from layers.BiGRU import NativeGRU as BiGRU


class Baseline(CQAModel):

    def transformer(self, QS, QB, trans_type=0):
        # QS: (B, L1, dim), QB: (B, L2, dim)
        with tf.variable_scope('transformer'):
            QS_exp = tf.tile(tf.expand_dims(QS, axis=2), [1, 1, self.Q_maxlen, 1])  # (B, L1, L2, dim)
            QB_exp = tf.tile(tf.expand_dims(QB, axis=1), [1, self.Q_maxlen, 1, 1])  # (B, L1, L2, dim)

            infomation = tf.concat([QS_exp, QB_exp, QS_exp - QB_exp, QS_exp * QB_exp], axis=3)
            infomation = tf.nn.dropout(infomation, keep_prob=self.dropout_keep_prob)
            score_matrix = tf.layers.dense(infomation, 1, activation=tf.tanh, use_bias=False)  # (B, L1, L2, 1)

            if trans_type == 0:
                mask = tf.expand_dims(tf.expand_dims(self.Q_mask, axis=1), axis=3)
                score_matrix -= (1 - mask) * 1e30
                alpha = tf.nn.softmax(score_matrix, axis=2)  # L2
                newQ = tf.reduce_sum(alpha * QB_exp, axis=2)  # (B, L1, dim)
            else:
                mask = tf.expand_dims(tf.expand_dims(self.Q_mask, axis=2), axis=3)
                score_matrix -= (1 - mask) * 1e30
                alpha = tf.nn.softmax(score_matrix, axis=1)  # L2
                newQ = tf.reduce_sum(alpha * QB_exp, axis=1)  # (B, L2, dim)

            return newQ

    def build_model(self):
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):
            units = 300
            with tf.variable_scope('encode'):
                rnn = BiGRU(num_layers=1, num_units=units,
                            batch_size=tf.shape(self.QS)[0], input_size=self.QS.get_shape()[-1],
                            keep_prob=self.dropout_keep_prob, is_train=self._is_train)

                QS_encode = rnn(self.QS, seq_len=self.QS_len)
                QB_encode = rnn(self.QB, seq_len=self.QS_len)

            newQS = self.transformer(QS_encode, QB_encode, trans_type=0)

            gate = tf.layers.dense(tf.concat([self.QS, newQS], axis=2), 1, activation=tf.sigmoid)  # (B, L1, 2*dim)
            Q = gate * QS_encode + (1 - gate) * newQS
            Q = tf.layers.dense(Q, self.CT.get_shape()[-1], activation=tf.tanh)

            with tf.variable_scope('extract'):
                rnn = BiGRU(num_layers=1, num_units=units,
                            batch_size=tf.shape(self.QS)[0], input_size=self.CT.get_shape()[-1],
                            keep_prob=self.dropout_keep_prob, is_train=self._is_train)

                Q_vec = rnn(Q, seq_len=self.QS_len, return_type=1)
                C_vec = rnn(self.CT, seq_len=self.CT_len, return_type=1)
            Q_vec_avg = tf.reduce_sum(
                Q_vec * tf.expand_dims(
                    self.QS_mask, axis=-1), axis=1) / tf.expand_dims(tf.cast(self.QS_len, tf.float32), axis=-1)
            C_vec_avg = tf.reduce_sum(
                C_vec * tf.expand_dims(
                    self.CT_mask, axis=-1), axis=1) / tf.expand_dims(tf.cast(self.CT_len, tf.float32), axis=-1)
            # Q_vec_max = tf.reduce_max(Q_vec - 10 * tf.expand_dims(1 - self.QS_mask, axis=-1), axis=1)
            # C_vec_max = tf.reduce_max(C_vec - 10 * tf.expand_dims(1 - self.CT_mask, axis=-1), axis=1)
            info = tf.concat([Q_vec_avg, C_vec_avg], axis=1)
            median = tf.layers.dense(info, 300, activation=tf.tanh)
            output = tf.layers.dense(median, self.args.categories_num, activation=tf.identity)

            return output

