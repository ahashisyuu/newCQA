import tensorflow as tf

from .CQAModel_margin import CQAModel
from layers.HyperQALayer import *
from layers.BiGRU import NativeGRU as BiGRU, Dropout


class HyperQA(CQAModel):

    def build_glove(self, embed, lens, max_len):
        embed = mask_zeros_1(embed, lens, max_len)
        return tf.reduce_sum(embed, 1)

    def learn_repr(self, q1_embed, q2_embed, q1_len, q2_len, q1_max,
                   q2_max, force_model=None, score=1,
                   reuse=None, extract_embed=False,
                   side=''):

        translate_act = tf.nn.relu
        use_mode = 'FC'
        num_outputs = 1
        rnn_size = 300
        num_proj = 3
        initializer = tf.contrib.layers.xavier_initializer()
        # initializer = tf.random_normal_initializer(0.0, self.args.init)
        # initializer = tf.random_uniform_initializer(maxval=self.args.init, minval=-self.args.init)

        q1_embed = projection_layer(
            q1_embed,
            rnn_size,
            name='trans_proj',
            activation=translate_act,
            initializer=initializer,
            dropout=self.dropout_keep_prob,
            reuse=reuse,
            use_mode=use_mode,
            num_layers=num_proj
        )
        q2_embed = projection_layer(
            q2_embed,
            rnn_size,
            name='trans_proj',
            activation=translate_act,
            initializer=initializer,
            dropout=self.dropout_keep_prob,
            reuse=True,
            use_mode=use_mode,
            num_layers=num_proj
        )

        q1_output = self.build_glove(q1_embed, q1_len, q1_max)
        q2_output = self.build_glove(q2_embed, q2_len, q2_max)

        try:
            self.max_norm = tf.reduce_max(tf.norm(q1_output,
                                                  ord='euclidean',
                                                  keep_dims=True, axis=1))
        except:
            self.max_norm = 0

        if extract_embed:
            self.q1_extract = q1_output
            self.q2_extract = q2_output

        q1_output = tf.nn.dropout(q1_output, self.dropout_keep_prob)
        q2_output = tf.nn.dropout(q2_output, self.dropout_keep_prob)

        # This constraint is important
        _q1_output = tf.clip_by_norm(q1_output, 1.0, axes=1)
        _q2_output = tf.clip_by_norm(q2_output, 1.0, axes=1)
        output = hyperbolic_ball(_q1_output, _q2_output)

        representation = output
        activation = None

        with tf.variable_scope('fl', reuse=reuse) as scope:
            last_dim = output.get_shape().as_list()[1]
            weights_linear = tf.get_variable('final_weights',
                                             [last_dim, num_outputs],
                                             initializer=initializer)
            bias_linear = tf.get_variable('bias',
                                          [num_outputs],
                                          initializer=tf.zeros_initializer())

            final_layer = tf.nn.xw_plus_b(output, weights_linear,
                                          bias_linear)
            output = final_layer

        return output, representation

    def build_model(self):
        with tf.variable_scope('build_model'):
            units = 300
            B = tf.shape(self.QS)[0]
            Q, pos_C, neg_C = self.QS, self.pos_CT, self.neg_CT
            Q_len, PosC_len, NegC_len = self.Q_len, self.PosC_len, self.NegC_len
            Q_maxlen, PosC_maxlen, NegC_maxlen = self.Q_maxlen, self.PosC_maxlen, self.NegC_maxlen

            Q = tf.nn.dropout(Q, self.dropout_keep_prob)
            pos_C = tf.nn.dropout(pos_C, self.dropout_keep_prob)
            neg_C = tf.nn.dropout(neg_C, self.dropout_keep_prob)

            repr_fun = self.learn_repr

            output_pos, _ = repr_fun(Q, pos_C,
                                     Q_len, PosC_len,
                                     Q_maxlen, PosC_maxlen,
                                     score=1, reuse=None, side='POS',
                                     )

            output_neg, _ = repr_fun(Q, neg_C,
                                     Q_len, NegC_len,
                                     Q_maxlen, NegC_maxlen,
                                     score=1, reuse=True, side='NEG',
                                     )

            return tf.squeeze(output_pos), tf.squeeze(output_neg)









