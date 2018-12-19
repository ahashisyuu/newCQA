import math

import tensorflow as tf

from .CQAModel import CQAModel
from layers.BiGRU import NativeGRU as BiGRU, Dropout
from layers.BiLSTM import NativeLSTM as BiLSTM
from layers.BertLayer import create_attention_mask_from_input_mask, attention_layer, gelu, reshape_to_matrix, \
    create_initializer, dropout, layer_norm, embedding_postprocessor


def projection(attention_output, hidden_size, initializer_range, hidden_dropout_prob, layer_input, reuse=None,
               name=None):
    with tf.variable_scope("output"):
        attention_output = tf.layers.dense(
            attention_output,
            hidden_size,
            reuse=reuse,
            kernel_initializer=create_initializer(initializer_range))
        attention_output = dropout(attention_output, hidden_dropout_prob)
        attention_output = layer_norm(attention_output + layer_input, name=name)
    return attention_output


class Baseline9(CQAModel):
    def build_model(self):
        with tf.variable_scope('baseline6', initializer=tf.glorot_uniform_initializer()):
            units = 256
            Q, C = self.QS, self.CT
            Q_len, C_len = self.Q_len, self.C_len

            with tf.variable_scope('position'):
                q_attention_mask = create_attention_mask_from_input_mask(self.QText, self.Q_mask)
                c_attention_mask = create_attention_mask_from_input_mask(self.CText, self.C_mask)
                hidden_size = 512
                num_hidden_layers = 1
                num_attention_heads = 8
                hidden_dropout = 0.1 if self.is_training else 0.0
                attention_dropout = 0.1 if self.is_training else 0.0
                initializer_range = 0.02

                full_position_embeddings = tf.get_variable(name='position',
                                                           shape=[200, 2 * units],
                                                           initializer=create_initializer(initializer_range))
                Q_sequence = embedding_postprocessor(Q,
                                                     full_position_embeddings=full_position_embeddings,
                                                     dropout_prob=hidden_dropout)
                C_sequence = embedding_postprocessor(C,
                                                     full_position_embeddings=full_position_embeddings,
                                                     dropout_prob=hidden_dropout)
                Q_sequence = tf.layers.conv1d(Q_sequence, units, 5, padding='same', name='encode')
                C_sequence = tf.layers.conv1d(C_sequence, units, 5, padding='same', name='encode', reuse=True)

            with tf.variable_scope('phrase_level'):
                if hidden_size % num_attention_heads != 0:
                    raise ValueError(
                        "The hidden size (%d) is not a multiple of the number of attention "
                        "heads (%d)" % (hidden_size, num_attention_heads))

                attention_head_size = int(hidden_size / num_attention_heads)
                input_width = 2 * units

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
                A = tf.layers.dense(H, units=1, activation=tf.tanh)  # (B, L1, L2, 1)
                A = tf.squeeze(A, axis=3)
                q_weight, q_indices = tf.nn.top_k(tf.transpose(A, [0, 1, 2]), k=5)  # (B, L1, k), (B, L1, k)
                c_weight, c_indices = tf.nn.top_k(tf.transpose(A, [0, 2, 1]), k=5)  # (B, L2, k), (B, L2, k)
                tf.batch_gather()

            info = tf.concat([Q_vec, C_vec], axis=1)
            info = Dropout(info, keep_prob=self.dropout_keep_prob, is_train=self._is_train)
            median = tf.layers.dense(info, 300, activation=tf.tanh)
            output = tf.layers.dense(median, self.args.categories_num, activation=tf.identity)

            return output

