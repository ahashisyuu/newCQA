import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from layers.TriangularWOupsent import TriangularCell
from layers.BiGRU import Dropout
from modeling import transformer_model, gelu, create_attention_mask_from_input_mask


class TextCNN:
    def __init__(self, input_dim, filter_sizes, filter_num):
        self.filter_sizes = filter_sizes
        self.filters = []
        self.bs = []
        for filter_size in filter_sizes:
            with tf.variable_scope("conv{}".format(filter_size)):
                filter_shape = tf.convert_to_tensor([filter_size, input_dim, 1, filter_num])
                fil = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='filter')
                b = tf.Variable(tf.constant(0.1, shape=[filter_num]))

                self.filters.append(fil)
                self.bs.append(b)

    def __call__(self, inputs, mask=None):
        pooled_outputs = []
        inputs = inputs * tf.expand_dims(mask, axis=-1) if mask is not None else inputs
        input_expand = tf.expand_dims(inputs, -1)  # (b,m,d,1)
        for filter_size, fil, b in zip(self.filter_sizes, self.filters, self.bs):
            with tf.variable_scope("conv{}".format(filter_size)):
                conv = tf.nn.conv2d(input_expand, fil, [1] * 4, 'VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b))
                pooled = tf.reduce_max(h, 1, True)
                pooled_outputs.append(tf.squeeze(pooled, axis=[1, 2]))
        return tf.concat(pooled_outputs, 1)


class MultiLayerHighway:
    def __init__(self, units, layers_num,
                 activation=tf.nn.elu, keep_prob=1.0, is_train=True, need_bias=True, scope=None):
        self.units = units
        self.layers_num = layers_num
        self.activation = activation
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.need_bias = need_bias
        self.first_use = True
        self.scope = scope or 'MulHighway'

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            output = inputs
            reuse = True if self.first_use is False else None
            for i in range(self.layers_num):
                output_line = Dropout(output, self.keep_prob, self.is_train)
                output_gate = Dropout(output, self.keep_prob, self.is_train)
                output_tran = Dropout(output, self.keep_prob, self.is_train)

                line = tf.layers.dense(output_line, self.units, tf.identity,
                                       self.need_bias, name='line_%d' % i, reuse=reuse)
                gate = tf.layers.dense(output_gate, self.units, tf.sigmoid,
                                       self.need_bias, name='gate_%d' % i, reuse=reuse)
                tran = tf.layers.dense(output_tran, self.units, self.activation,
                                       self.need_bias, name='tran_%d' % i, reuse=reuse)
                output = gate * tran + (1 - gate) * line
            self.first_use = False
            return output


class BertCQAModel:
    def __init__(self, is_training: bool, features: dict, initializer=xavier_initializer(), config=None):
        self.is_training = is_training
        self.initializer = initializer
        self.config = config

        self.input_keep_prob = 0.9
        self.output_keep_prob = 0.9
        self.state_keep_prob = 0.9

        with tf.variable_scope("CQAModel"):
            self.sent1 = features["sent1"]
            self.sent2 = features["sent2"]
            self.sent3 = features["sent3"]

            self.mask1 = features["mask1"]
            self.mask2 = features["mask2"]
            self.mask3 = features["mask3"]

            self.mark0 = features["mark0"]
            self.mark1 = features["mark1"]
            self.mark2 = features["mark2"]
            self.mark3 = features["mark3"]

            self.q_type = features["q_type"]
            self.labels = features["labels"]

            self.output = None

            self.encoding_size = self.sent1.get_shape()[-1]
            self._keys_embedding = None
            self.create_memory()
            self.build_model()

    def create_memory(self):
        with tf.variable_scope("memory"):
            self._keys_embedding = tf.get_variable("keys_embedding", [self.config.keys_num, self.encoding_size],
                                                   dtype=tf.float32,
                                                   initializer=self.initializer,
                                                   trainable=True)

    def sent_transformer(self,
                         hidden_size=768,
                         num_hidden_layers=1,
                         num_attention_heads=12,
                         intermediate_size=768,
                         intermediate_act_fn=gelu,
                         hidden_dropout_prob=0.1,
                         attention_probs_dropout_prob=0.1,
                         initializer_range=0.02,
                         do_return_all_layers=False):
        # def _trans_v1(sent, mask):
        #     attention_mask = create_attention_mask_from_input_mask(
        #         sent, mask)
        #
        #     with tf.variable_scope("sent_transformer", reuse=tf.AUTO_REUSE):
        #
        #         return transformer_model(sent,
        #                                  attention_mask=attention_mask,
        #                                  hidden_size=hidden_size,
        #                                  num_hidden_layers=num_hidden_layers,
        #                                  num_attention_heads=num_attention_heads,
        #                                  intermediate_size=intermediate_size,
        #                                  intermediate_act_fn=intermediate_act_fn,
        #                                  hidden_dropout_prob=hidden_dropout_prob,
        #                                  attention_probs_dropout_prob=attention_probs_dropout_prob,
        #                                  initializer_range=initializer_range,
        #                                  do_return_all_layers=do_return_all_layers)

        def _trans_v2(sent, mask):
            with tf.variable_scope("sent_transformer", reuse=tf.AUTO_REUSE):
                return tf.layers.dense(sent, hidden_size, activation=tf.tanh, name="trans_v2")

        def _trans_v3(sent, mask):
            with tf.variable_scope("sent_transformer", reuse=tf.AUTO_REUSE):
                return None

        return _trans_v2

    def build_model(self):
        with tf.variable_scope("inferring_module"):
            batch_size = tf.shape(self.sent1)[0]
            s1_len, dim = self.sent1.get_shape().as_list()[1:]
            s2_len, _ = self.sent2.get_shape()[1:]
            s3_len, _ = self.sent3.get_shape()[1:]

            sr_cell = GRUCell(num_units=dim, activation=tf.nn.relu)

            if self.is_training:
                sr_cell = DropoutWrapper(cell=sr_cell,
                                         input_keep_prob=self.input_keep_prob,
                                         output_keep_prob=self.output_keep_prob,
                                         state_keep_prob=self.state_keep_prob,
                                         variational_recurrent=True,
                                         input_size=dim,
                                         dtype=tf.float32)
            sent_cell = r_cell = sr_cell

            # sent_transformer = self.sent_transformer(hidden_size=dim)
            highway = MultiLayerHighway(dim, 1, keep_prob=1.0, is_train=tf.constant(self.is_training,
                                                                                    dtype=tf.bool,
                                                                                    shape=[]))

            def _trans(_sent, _mask):
                # return highway(_sent)
                return _sent

            sent_transformer = _trans

            tri_cell = TriangularCell(num_units=256,
                                      sent_cell=sent_cell, r_cell=r_cell, sentence_transformer=sent_transformer,
                                      sent1=self.sent1, sent2=self.sent2, sent3=self.sent3,
                                      sent1_length=s1_len,
                                      sent2_length=s2_len,
                                      sent3_length=s3_len,
                                      dim=self.encoding_size,
                                      keys=self._keys_embedding,
                                      sent1_mask=self.mask1, sent2_mask=self.mask2, sent3_mask=self.mask3,
                                      initializer=None, dtype=tf.float32)

            fake_input = tf.tile(tf.expand_dims(self.mark0, axis=1), [1, self.config.update_num, 1])
            init_state = tri_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            if self.is_training:
                tri_cell = DropoutWrapper(cell=tri_cell,
                                          input_keep_prob=self.input_keep_prob,
                                          output_keep_prob=self.output_keep_prob,
                                          state_keep_prob=self.state_keep_prob,
                                          variational_recurrent=True,
                                          input_size=fake_input.get_shape()[2],
                                          dtype=tf.float32)

            output, last_state = dynamic_rnn(cell=tri_cell,
                                             inputs=fake_input,
                                             initial_state=init_state)
            self.output = [tf.reshape(a, [-1, self.config.keys_num, self.encoding_size]) for a in last_state[3:6]]
            # self.output = output

    def get_output(self):
        # return self.output
        with tf.variable_scope("relation_output"):
            num_units = 256
            use_bias = True
            weight = tf.get_variable(name="r_self_weight",
                                     shape=[self.encoding_size, num_units],
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            v = tf.get_variable(name="r_self_v", shape=[num_units, 1],
                                initializer=xavier_initializer())

            temp_tensor = tf.matmul(self._keys_embedding, weight)  # (keys_num, units)
            if use_bias:
                bias = tf.get_variable(name="r_self_bias", shape=[num_units],
                                       initializer=tf.zeros_initializer(dtype=tf.float32))
                temp_tensor = tf.nn.bias_add(temp_tensor, bias)
            temp_array = tf.matmul(tf.tanh(temp_tensor), v)  # (keys_num, 1)
            alpha = tf.nn.softmax(temp_array, axis=0)
            alpha = tf.expand_dims(alpha, axis=0)  # (1, keys_num, 1)

            output = [tf.reduce_sum(alpha * values, axis=1) for values in self.output]
            output = tf.concat(output, axis=1)

            return output
