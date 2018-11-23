import tensorflow as tf

from .CQAModel import CQAModel


class RandomModel(CQAModel):
    def build_model(self):
        with tf.variable_scope('CT_random', initializer=tf.glorot_uniform_initializer()):
            score = tf.layers.dense(self.CT, 1, activation=tf.tanh)
            score -= (1-tf.expand_dims(self.CT_mask, -1))*1e30
            alpha = tf.nn.softmax(score)
            r = tf.reduce_sum(alpha * self.CT, axis=1)
            r = tf.layers.dense(r, 100, activation=tf.tanh)
            return tf.layers.dense(r, 3, activation=tf.identity)

