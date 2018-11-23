import tensorflow as tf


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


def bidirectional_rnn(cell_fw, cell_bw, inputs, sequence_length=None,
                      dropout=0., is_train=None,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, parallel_iterations=None,
                      swap_memory=False, time_major=False, scope=None):
    with tf.variable_scope(scope or "bidirectional_rnn"):
        # dropout mask
        shape = inputs.get_shape()
        mask_fw = Dropout(tf.ones(shape[0], 1, shape[2]), 1-dropout, is_train, mode='')
        mask_bw = Dropout(tf.ones(shape[0], 1, shape[2]), 1-dropout, is_train, mode='')

        # Forward direction
        with tf.variable_scope("fw") as fw_scope:
            output_fw, output_state_fw = tf.nn.dynamic_rnn(
                cell=cell_fw, inputs=inputs * mask_fw, sequence_length=sequence_length,
                initial_state=initial_state_fw, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, scope=fw_scope)

        # Backward direction
        if not time_major:
            time_dim = 1
            batch_dim = 0
        else:
            time_dim = 0
            batch_dim = 1

        def _reverse(input_, seq_lengths, seq_dim, batch_axis):
            if seq_lengths is not None:
                return tf.reverse_sequence(
                    input=input_ * mask_bw, seq_lengths=seq_lengths,
                    seq_axis=seq_dim, batch_axis=batch_axis)
            else:
                return tf.reverse(input_, axis=[seq_dim])

        with tf.variable_scope("bw") as bw_scope:
            inputs_reverse = _reverse(
                inputs, seq_lengths=sequence_length,
                seq_dim=time_dim, batch_axis=batch_dim)
            tmp, output_state_bw = tf.nn.dynamic_rnn(
                cell=cell_bw, inputs=inputs_reverse, sequence_length=sequence_length,
                initial_state=initial_state_bw, dtype=dtype,
                parallel_iterations=parallel_iterations, swap_memory=swap_memory,
                time_major=time_major, scope=bw_scope)

    output_bw = _reverse(
        tmp, seq_lengths=sequence_length,
        seq_dim=time_dim, batch_axis=batch_dim)

    outputs = (output_fw, output_bw)
    output_states = (output_state_fw, output_state_bw)

    return outputs, output_states


def BiGRU(sentence, units, dropout=0., is_train=None, activation=tf.tanh, length=None, return_type=0):
    with tf.variable_scope('BiGRU'):
        cell_fw = tf.nn.rnn_cell.GRUCell(num_units=units, activation=activation)
        cell_bw = tf.nn.rnn_cell.GRUCell(num_units=units, activation=activation)

        outputs, states = bidirectional_rnn(
            cell_fw, cell_bw, sentence, sequence_length=length,
            dropout=dropout, is_train=is_train, dtype=tf.float32)
        if return_type == 0:
            return tf.concat(outputs, axis=2)
        elif return_type == 1:
            return tf.concat(states, axis=1)
        elif return_type == 2:
            return tf.concat(outputs, axis=2), tf.concat(states, axis=1)
        else:
            raise ValueError('incorrect return type')


class CudnnGRU:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope=None):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            gru_bw = tf.contrib.cudnn_rnn.CudnnGRU(1, num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, 1, num_units])), [1, batch_size, 1])
            mask_fw = Dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode='')
            mask_bw = Dropout(tf.ones([1, batch_size, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode='')
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True):
        outputs = [tf.transpose(inputs, [1, 0, 2])]
        for layer in range(self.num_layers):
            gru_fw, gru_bw = self.grus[layer]
            init_fw, init_bw = self.inits[layer]
            mask_fw, mask_bw = self.dropout_mask[layer]
            with tf.variable_scope("fw_{}".format(layer)):
                out_fw, _ = gru_fw(
                    outputs[-1] * mask_fw, initial_state=(init_fw, ))
            with tf.variable_scope("bw_{}".format(layer)):
                inputs_bw = tf.reverse_sequence(
                    outputs[-1] * mask_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
                out_bw, _ = gru_bw(inputs_bw, initial_state=(init_bw, ))
                out_bw = tf.reverse_sequence(
                    out_bw, seq_lengths=seq_len, seq_dim=0, batch_dim=1)
            outputs.append(tf.concat([out_fw, out_bw], axis=2))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]
        res = tf.transpose(res, [1, 0, 2])
        return res


class NativeGRU:

    def __init__(self, num_layers, num_units, batch_size, input_size, keep_prob=1.0, is_train=None, scope="native_gru"):
        self.num_layers = num_layers
        self.grus = []
        self.inits = []
        self.dropout_mask = []
        self.scope = scope
        for layer in range(num_layers):
            input_size_ = input_size if layer == 0 else 2 * num_units
            gru_fw = tf.contrib.rnn.GRUCell(num_units)
            gru_bw = tf.contrib.rnn.GRUCell(num_units)
            init_fw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            init_bw = tf.tile(tf.Variable(
                tf.zeros([1, num_units])), [batch_size, 1])
            mask_fw = Dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode='')
            mask_bw = Dropout(tf.ones([batch_size, 1, input_size_], dtype=tf.float32),
                              keep_prob=keep_prob, is_train=is_train, mode='')
            self.grus.append((gru_fw, gru_bw, ))
            self.inits.append((init_fw, init_bw, ))
            self.dropout_mask.append((mask_fw, mask_bw, ))

    def __call__(self, inputs, seq_len, keep_prob=1.0, is_train=None, concat_layers=True, return_type=1):
        outputs = [inputs]
        states = []
        with tf.variable_scope(self.scope):
            for layer in range(self.num_layers):
                gru_fw, gru_bw = self.grus[layer]
                init_fw, init_bw = self.inits[layer]
                mask_fw, mask_bw = self.dropout_mask[layer]
                with tf.variable_scope("fw_{}".format(layer)):
                    out_fw, state_fw = tf.nn.dynamic_rnn(
                        gru_fw, outputs[-1] * mask_fw, seq_len, initial_state=init_fw, dtype=tf.float32)
                with tf.variable_scope("bw_{}".format(layer)):
                    inputs_bw = tf.reverse_sequence(
                        outputs[-1] * mask_bw, seq_lengths=seq_len, seq_axis=1, batch_axis=0)
                    out_bw, state_bw = tf.nn.dynamic_rnn(
                        gru_bw, inputs_bw, seq_len, initial_state=init_bw, dtype=tf.float32)
                    out_bw = tf.reverse_sequence(
                        out_bw, seq_lengths=seq_len, seq_dim=1, batch_dim=0)
                outputs.append(tf.concat([out_fw, out_bw], axis=2))
                states.append(tf.concat([state_fw, state_bw], axis=1))
        if concat_layers:
            res = tf.concat(outputs[1:], axis=2)
        else:
            res = outputs[-1]

        if return_type == 0:
            return states[-1]
        elif return_type == 1:
            return res
        else:
            return res, states[-1]











