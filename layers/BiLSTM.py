import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn


def Dropout(args, keep_prob, is_train, mode="recurrent", name=None):

    def _dropout():
        _args = args
        noise_shape = None
        scale = 1.0
        shape = tf.shape(_args)
        if mode == "embedding":
            noise_shape = [shape[0], 1]
            scale = keep_prob
        if mode == "recurrent" and len(_args.get_shape().as_list()) == 3:
            noise_shape = [shape[0], 1, shape[-1]]
        _args = tf.cond(is_train, lambda: tf.nn.dropout(
            _args, keep_prob, noise_shape=noise_shape, name=name) * scale, lambda: _args)
        return _args

    return tf.cond(tf.less(keep_prob, 1.0), _dropout, lambda: args)


class NativeLSTM:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.nn.rnn_cell.LSTMCell(num_units)
            gru_bw = tf.nn.rnn_cell.LSTMCell(num_units)
            mask_fw = Dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode='')
            mask_bw = Dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode='')
            self.grus.append((gru_fw, gru_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [inputs]
        states = []
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, state_fw = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_axis=1, batch_axis=0)
                    out_bw, state_bw = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
                # states.append(tf.concat([state_fw[0], state_bw[0]], axis=1))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]

        return res