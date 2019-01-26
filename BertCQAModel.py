import tensorflow as tf

from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.python.ops.rnn import dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import GRUCell
from layers.TriangularUpdate import TriangularCell


class SentenceTransformer:
    def __init__(self, num_units):
        self.num_units = num_units


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

            self.encoding_size = self.sent1.get_shape()[-1]
            self._keys_embedding = None
            self.create_memory()
            self.build_model()

            self.output = None

    def create_memory(self):
        with tf.variable_scope("memory"):
            self._keys_embedding = tf.get_variable("keys_embedding", [self.config.keys_num, self.encoding_size],
                                                   dtype=tf.float32,
                                                   initializer=self.initializer,
                                                   trainable=True)

    def build_model(self):
        with tf.variable_scope("inferring_module"):
            batch_size, s1_len, dim = tf.shape(self.sent1)
            _, s2_len, _ = tf.shape(self.sent2)
            _, s3_len, _ = tf.shape(self.sent3)

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
            self.output = [tf.reshape(a, [-1, self.config.key_num, self.encoding_size]) for a in last_state[3:6]]
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
            output = tf.concat([output], axis=1)

            return output



