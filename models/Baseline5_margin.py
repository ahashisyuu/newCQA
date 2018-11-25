import tensorflow as tf

from .CQAModel_margin import CQAModel
from layers.BiGRU import NativeGRU as BiGRU, Dropout


class Baseline5(CQAModel):
    def build_model(self):
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, pos_C, neg_C = self.QS, self.pos_CT, self.neg_CT
            Q_len, PosC_len, NegC_len = self.Q_len, self.PosC_len, self.NegC_len

            with tf.variable_scope('encode'):
                # Q = Dropout(Q, keep_prob=self.dropout_keep_prob, is_train=self._is_train)
                # pos_C = Dropout(pos_C, keep_prob=self.dropout_keep_prob, is_train=self._is_train, name='C')
                # neg_C = Dropout(neg_C, keep_prob=self.dropout_keep_prob, is_train=self._is_train, name='C')
                Q_sequence = tf.layers.conv1d(Q, filters=200, kernel_size=3, padding='same')
                pos_C_sequence = tf.layers.conv1d(pos_C, filters=200, kernel_size=3, padding='same', name='C_sequence')
                neg_C_sequence = tf.layers.conv1d(neg_C, filters=200, kernel_size=3, padding='same', name='C_sequence', reuse=True)

            with tf.variable_scope('interaction'):
                Q_ = tf.expand_dims(Q_sequence, axis=2)  # (B, L1, 1, dim)
                posC_ = tf.expand_dims(pos_C_sequence, axis=1)  # (B, 1, L2, dim)
                negC_ = tf.expand_dims(neg_C_sequence, axis=1)  # (B, 1, L2, dim)
                poshQ = tf.tile(Q_, [1, 1, self.PosC_maxlen, 1])
                neghQ = tf.tile(Q_, [1, 1, self.NegC_maxlen, 1])
                poshC = tf.tile(posC_, [1, self.Q_maxlen, 1, 1])
                neghC = tf.tile(negC_, [1, self.Q_maxlen, 1, 1])
                posH = tf.concat([poshQ, poshC], axis=-1)
                negH = tf.concat([neghQ, neghC], axis=-1)


                posA = tf.layers.dense(posH, units=200, activation=tf.tanh, name='A')  # (B, L1, L2, dim)
                negA = tf.layers.dense(negH, units=200, activation=tf.tanh, name='A', reuse=True)  # (B, L1, L2, dim)

                pos_rQ = tf.reduce_max(posA, axis=2)
                pos_rC = tf.reduce_max(posA, axis=1)

                neg_rQ = tf.reduce_max(negA, axis=2)
                neg_rC = tf.reduce_max(negA, axis=1)

            with tf.variable_scope('attention'):
                # concate
                cate_f_ = tf.expand_dims(self.cate_f, axis=1)
                posQ_m = tf.concat([Q_sequence, pos_rQ, tf.tile(cate_f_, [1, self.Q_maxlen, 1])], axis=-1)
                negQ_m = tf.concat([Q_sequence, neg_rQ, tf.tile(cate_f_, [1, self.Q_maxlen, 1])], axis=-1)
                posC_m = tf.concat([pos_C_sequence, pos_rC, tf.tile(cate_f_, [1, self.PosC_maxlen, 1])], axis=-1)
                negC_m = tf.concat([neg_C_sequence, neg_rC, tf.tile(cate_f_, [1, self.NegC_maxlen, 1])], axis=-1)

                # posQ_m = Dropout(posQ_m, keep_prob=self.dropout_keep_prob, is_train=self._is_train, name='Q_m')
                # negQ_m = Dropout(negQ_m, keep_prob=self.dropout_keep_prob, is_train=self._is_train, name='Q_m')
                # posC_m = Dropout(posC_m, keep_prob=self.dropout_keep_prob, is_train=self._is_train, name='C_m')
                # negC_m = Dropout(negC_m, keep_prob=self.dropout_keep_prob, is_train=self._is_train, name='C_m')

                posQ_m = tf.layers.dense(posQ_m, units=300, activation=tf.tanh, name='fw1')
                negQ_m = tf.layers.dense(negQ_m, units=300, activation=tf.tanh, name='fw1', reuse=True)
                posC_m = tf.layers.dense(posC_m, units=300, activation=tf.tanh, name='fw1', reuse=True)
                negC_m = tf.layers.dense(negC_m, units=300, activation=tf.tanh, name='fw1', reuse=True)

                # posQ_m = Dropout(posQ_m, keep_prob=self.dropout_keep_prob, is_train=self._is_train, name='Q_m2')
                # negQ_m = Dropout(negQ_m, keep_prob=self.dropout_keep_prob, is_train=self._is_train, name='Q_m2')
                # posC_m = Dropout(posC_m, keep_prob=self.dropout_keep_prob, is_train=self._is_train, name='C_m2')
                # negC_m = Dropout(negC_m, keep_prob=self.dropout_keep_prob, is_train=self._is_train, name='C_m2')

                posQ_m = tf.layers.dense(posQ_m, units=1, activation=tf.tanh, name='fw2')
                negQ_m = tf.layers.dense(negQ_m, units=1, activation=tf.tanh, name='fw2', reuse=True)
                posC_m = tf.layers.dense(posC_m, units=1, activation=tf.tanh, name='fw2', reuse=True)
                negC_m = tf.layers.dense(negC_m, units=1, activation=tf.tanh, name='fw2', reuse=True)

                posQ_m -= (1 - tf.expand_dims(self.Q_mask, axis=-1)) * 1e30
                negQ_m -= (1 - tf.expand_dims(self.Q_mask, axis=-1)) * 1e30
                posC_m -= (1 - tf.expand_dims(self.PosC_mask, axis=-1)) * 1e30
                negC_m -= (1 - tf.expand_dims(self.NegC_mask, axis=-1)) * 1e30

                posQalpha = tf.nn.softmax(posQ_m, axis=1)
                negQalpha = tf.nn.softmax(negQ_m, axis=1)
                posCalpha = tf.nn.softmax(posC_m, axis=1)
                negCalpha = tf.nn.softmax(negC_m, axis=1)

                posQ_vec = tf.reduce_sum(posQalpha * pos_rQ, axis=1)
                negQ_vec = tf.reduce_sum(negQalpha * neg_rQ, axis=1)
                posC_vec = tf.reduce_sum(posCalpha * pos_rC, axis=1)
                negC_vec = tf.reduce_sum(negCalpha * neg_rC, axis=1)

            with tf.name_scope('info'):
                pos_vec = tf.concat([posQ_vec, posC_vec], axis=-1)
                neg_vec = tf.concat([negQ_vec, negC_vec], axis=-1)
                pos_info = tf.layers.dense(pos_vec, 200, activation=tf.tanh, name='info')
                neg_info = tf.layers.dense(neg_vec, 200, activation=tf.tanh, name='info', reuse=True)

            with tf.name_scope('similarity'):
                # norm_pos_q_pool = tf.nn.l2_normalize(posQ_vec, axis=1)
                # norm_neg_q_pool = tf.nn.l2_normalize(negQ_vec, axis=1)
                # norm_pos_c_pool = tf.nn.l2_normalize(posC_vec, axis=1)
                # norm_neg_c_pool = tf.nn.l2_normalize(negC_vec, axis=1)
                # pos_similarity = tf.reduce_sum(tf.multiply(norm_pos_q_pool, norm_pos_c_pool), 1)
                # neg_similarity = tf.reduce_sum(tf.multiply(norm_neg_q_pool, norm_neg_c_pool), 1)
                pos_similarity = tf.layers.dense(pos_info, 1, activation=tf.sigmoid, use_bias=False, name='similarity')
                neg_similarity = tf.layers.dense(neg_info, 1, activation=tf.sigmoid, use_bias=False, name='similarity', reuse=True)

            return pos_similarity, neg_similarity

