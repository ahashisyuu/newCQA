import math

import tensorflow as tf

from .CQAModel import CQAModel
from layers.BiGRU import NativeGRU as BiGRU, Dropout
from layers.BiLSTM import NativeLSTM as BiLSTM
from layers.BertLayer import create_attention_mask_from_input_mask, attention_layer, gelu, reshape_to_matrix, \
    create_initializer, dropout, layer_norm, embedding_postprocessor


def add_timing_signal_1d(x, min_timescale=1.0, max_timescale=1.0e4):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    x: a Tensor with shape [batch, length, channels]
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor the same shape as x.
    """
    length = tf.shape(x)[1]
    channels = tf.shape(x)[2]
    signal = get_timing_signal_1d(length, channels, min_timescale, max_timescale)
    return x + signal


def get_timing_signal_1d(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """Gets a bunch of sinusoids of different frequencies.
    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.
    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.
    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).
    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.
    Args:
    length: scalar, length of timing signal sequence.
    channels: scalar, size of timing embeddings to create. The number of
        different timescales is equal to channels / 2.
    min_timescale: a float
    max_timescale: a float
    Returns:
    a Tensor of timing signals [1, length, channels]
    """
    position = tf.to_float(tf.range(length))
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
            (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.reshape(signal, [1, length, channels])
    return signal


def projection(attention_output, hidden_size, initializer_range, hidden_dropout_prob, layer_input, reuse=None, name=None):
    with tf.variable_scope("output"):
        attention_output = tf.layers.dense(
            attention_output,
            hidden_size,
            reuse=reuse,
            kernel_initializer=create_initializer(initializer_range))
        attention_output = dropout(attention_output, hidden_dropout_prob)
        attention_output = layer_norm(attention_output + layer_input, name=name)
    return attention_output


class Baseline7(CQAModel):
    def build_model(self):
        with tf.variable_scope('baseline7', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, C = self.QS, self.CT
            Q_len, C_len = self.Q_len, self.C_len

            with tf.variable_scope('position'):
                q_attention_mask = create_attention_mask_from_input_mask(self.QText, self.Q_mask)
                c_attention_mask = create_attention_mask_from_input_mask(self.CText, self.C_mask)
                hidden_size = 300
                num_hidden_layers = 1
                num_attention_heads = 6
                hidden_dropout = 0.1 if self.is_training else 0.0
                attention_dropout = 0.1 if self.is_training else 0.0
                initializer_range = 0.02

                full_position_embeddings = tf.get_variable(name='position',
                                                           shape=[200, units],
                                                           initializer=create_initializer(initializer_range))
                Q_sequence = embedding_postprocessor(Q,
                                                     full_position_embeddings=full_position_embeddings,
                                                     dropout_prob=hidden_dropout)
                C_sequence = embedding_postprocessor(C,
                                                     full_position_embeddings=full_position_embeddings,
                                                     dropout_prob=hidden_dropout)

            with tf.variable_scope('phrase_level'):
                if hidden_size % num_attention_heads != 0:
                    raise ValueError(
                        "The hidden size (%d) is not a multiple of the number of attention "
                        "heads (%d)" % (hidden_size, num_attention_heads))

                attention_head_size = int(hidden_size / num_attention_heads)
                input_width = units

                if input_width != hidden_size:
                    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                                     (input_width, hidden_size))

                q_prev_output = reshape_to_matrix(Q_sequence)
                c_prev_output = reshape_to_matrix(C_sequence)

                q_all_layers_outputs = []
                c_all_layers_outputs = []
                for layer_idx in range(num_hidden_layers):
                    with tf.variable_scope("layer_%d" % layer_idx):
                        q_layer_input = q_prev_output
                        c_layer_input = c_prev_output

                        q_all_layers = attention_layer(from_tensor=q_layer_input, to_tensor=q_layer_input,
                                                       attention_mask=q_attention_mask,
                                                       num_attention_heads=num_attention_heads,
                                                       size_per_head=attention_head_size,
                                                       attention_probs_dropout_prob=attention_dropout,
                                                       initializer_range=initializer_range,
                                                       do_return_2d_tensor=True,
                                                       batch_size=self.N,
                                                       from_seq_length=self.Q_maxlen,
                                                       to_seq_length=self.Q_maxlen)
                        c_all_layers = attention_layer(from_tensor=c_layer_input, to_tensor=c_layer_input,
                                                       attention_mask=c_attention_mask,
                                                       num_attention_heads=num_attention_heads,
                                                       size_per_head=attention_head_size,
                                                       attention_probs_dropout_prob=attention_dropout,
                                                       initializer_range=initializer_range,
                                                       do_return_2d_tensor=True,
                                                       batch_size=self.N,
                                                       from_seq_length=self.C_maxlen,
                                                       to_seq_length=self.C_maxlen,
                                                       reuse=True)

                        q_output = projection(q_all_layers, hidden_size, initializer_range,
                                              hidden_dropout, q_layer_input, name='q')
                        c_output = projection(c_all_layers, hidden_size, initializer_range,
                                              hidden_dropout, c_layer_input, reuse=True, name='c')

                        q_prev_output = q_output
                        c_prev_output = c_output

                        q_all_layers_outputs.append(q_output)
                        c_all_layers_outputs.append(c_output)

            with tf.variable_scope('interaction'):
                q_prev_output = tf.reshape(q_prev_output, [self.N, -1, hidden_size])
                c_prev_output = tf.reshape(c_prev_output, [self.N, -1, hidden_size])
                Q_ = tf.expand_dims(q_prev_output, axis=2)  # (B, L1, 1, dim)
                C_ = tf.expand_dims(c_prev_output, axis=1)  # (B, 1, L2, dim)
                hQ = tf.tile(Q_, [1, 1, self.C_maxlen, 1])
                hC = tf.tile(C_, [1, self.Q_maxlen, 1, 1])
                H = tf.concat([hQ, hC], axis=-1)
                # H = tf.concat([tf.abs(hQ - hC)], axis=-1)
                A = tf.layers.dense(H, units=200, activation=tf.tanh)  # (B, L1, L2, dim)

                rQ = tf.reduce_max(A, axis=2)
                rC = tf.reduce_max(A, axis=1)

            with tf.variable_scope('attention'):
                # concate
                cate_f_ = tf.expand_dims(self.cate_f, axis=1)
                Q_m = tf.concat([Q_sequence, rQ, tf.tile(cate_f_, [1, self.Q_maxlen, 1])], axis=-1)
                C_m = tf.concat([C_sequence, rC, tf.tile(cate_f_, [1, self.C_maxlen, 1])], axis=-1)

                Q_m = Dropout(Q_m, keep_prob=self.dropout_keep_prob, is_train=self._is_train)
                C_m = Dropout(C_m, keep_prob=self.dropout_keep_prob, is_train=self._is_train)

                Q_m = tf.layers.dense(Q_m, units=300, activation=tf.tanh, name='fw1')
                C_m = tf.layers.dense(C_m, units=300, activation=tf.tanh, name='fw1', reuse=True)

                Q_m = Dropout(Q_m, keep_prob=0.8, is_train=self._is_train)
                C_m = Dropout(C_m, keep_prob=0.8, is_train=self._is_train)

                Q_m = tf.layers.dense(Q_m, units=1, activation=tf.tanh, name='fw2')
                C_m = tf.layers.dense(C_m, units=1, activation=tf.tanh, name='fw2', reuse=True)

                Q_m -= (1 - tf.expand_dims(self.Q_mask, axis=-1)) * 1e30
                C_m -= (1 - tf.expand_dims(self.C_mask, axis=-1)) * 1e30
                Qalpha = tf.nn.softmax(Q_m, axis=1)
                Calpha = tf.nn.softmax(C_m, axis=1)

                Q_vec = tf.reduce_sum(Qalpha * rQ, axis=1)
                C_vec = tf.reduce_sum(Calpha * rC, axis=1)

            info = tf.concat([Q_vec, C_vec], axis=1)
            info = Dropout(info, keep_prob=self.dropout_keep_prob, is_train=self._is_train)
            median = tf.layers.dense(info, 300, activation=tf.tanh)
            output = tf.layers.dense(median, self.args.categories_num, activation=tf.identity)

            return output

