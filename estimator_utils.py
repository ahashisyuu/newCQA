import os
import pickle as pkl
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from tensorflow.contrib.layers import xavier_initializer
# from models.CBertCQAModel import BertCQAModel
# from layers.optimization import create_optimizer
from models.BertCQAModelAlign import BertCQAModel


def _create_model(is_training, features, num_labels, config):

    # build model
    model = BertCQAModel(is_training=is_training, features=features, config=config)

    relation_vec = model.get_output()  # (B, D)
    hidden_size = relation_vec.shape[-1].value

    relation_weights = tf.get_variable(
        "relation_weights", [hidden_size, num_labels],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    relation_bias = tf.get_variable(
        "relation_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            relation_vec = tf.nn.dropout(relation_vec, keep_prob=0.9)

        logits = tf.keras.backend.dot(relation_vec, relation_weights)
        logits = tf.nn.bias_add(logits, relation_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(features["labels"], num_labels, dtype=tf.float32)

        per_loss = - tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        total_loss = tf.reduce_mean(per_loss)

    return total_loss, logits, per_loss


def _create_model_v2(is_training, features, num_labels, config):

    # build model
    model = BertCQAModel(is_training=is_training, features=features, config=config)

    output = model.get_output()  # (B, update_num, 3*dim*keys_num + 1)

    batch_size, update_num, _ = tf.shape(output)
    dim = model.encoding_size
    keys_num = config.keys_num
    beta = 0.5

    relation_values = tf.slice(output,
                               [0, 0, 0],
                               [batch_size, update_num, 3 * keys_num * dim])  # (B, U, 3dim*keys_num)
    score = tf.slice(output,
                     [0, 0, 3 * keys_num * dim],
                     [batch_size, update_num, 1])  # (B, U, 1)

    # ready for assistant loss
    subloss_log_probs = tf.nn.log_softmax(tf.squeeze(score, axis=2), axis=1)

    # ready for main loss
    with tf.variable_scope("values_norm"):
        num_units = 256
        use_bias = True

        relation_values = tf.reshape(relation_values, [-1, update_num, 3, keys_num, dim])

        weight = tf.get_variable(name="r_self_weight",
                                 shape=[dim, num_units],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02))
        v = tf.get_variable(name="r_self_v", shape=[num_units, 1],
                            initializer=xavier_initializer())

        temp_tensor = tf.matmul(model._keys_embedding, weight)  # (keys_num, units)
        if use_bias:
            bias = tf.get_variable(name="r_self_bias", shape=[num_units],
                                   initializer=tf.zeros_initializer(dtype=tf.float32))
            temp_tensor = tf.nn.bias_add(temp_tensor, bias)
        temp_array = tf.matmul(tf.tanh(temp_tensor), v)  # (keys_num, 1)

        alpha = tf.nn.softmax(temp_array, axis=0)
        alpha = tf.expand_dims(tf.expand_dims(tf.expand_dims(alpha, axis=0), axis=0), axis=0)  # (1, 1, 1, keys_num, 1)
        res_vec = tf.reduce_sum(alpha * relation_values, axis=3)  # (B, U, 3, dim)
        res_vec = tf.reshape(res_vec, [batch_size, update_num, 3 * dim])

    relation_weights = tf.get_variable(
        "relation_weights", [num_labels, dim],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    relation_bias = tf.get_variable(
        "relation_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            res_vec = tf.nn.dropout(res_vec, keep_prob=0.9)

        logits = tf.matmul(res_vec, relation_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, relation_bias)  # (B, U, 2)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(features["labels"], num_labels, dtype=tf.float32)  # (B, 2)
        one_hot_labels = tf.expand_dims(one_hot_labels, axis=1)  # (B, 1, 2)

        per_loss = - tf.reduce_sum(one_hot_labels * log_probs, axis=-1)  # (B, U)

        indices = tf.argmin(per_loss, axis=1)  # (B,)
        per_loss = tf.reduce_min(per_loss, axis=1)  # (B,)

        # MAIN loss
        main_loss = tf.reduce_mean(per_loss)

        # sub loss
        one_hot_sub = tf.one_hot(indices, depth=update_num, dtype=tf.float32)  # (B, U)
        sub_per_loss = - tf.reduce_sum(one_hot_sub * subloss_log_probs, axis=1)  # (B,)
        sub_loss = tf.reduce_mean(sub_per_loss)

        total_loss = main_loss + beta * sub_loss

    logits_indices = tf.argmax(score, axis=1)  # (B,)
    output_logits = tf.batch_gather(logits, logits_indices)  # (B, 2)

    return total_loss, output_logits


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def model_fn_builder(num_labels, learning_rate, num_train_steps, config):

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        # sent1 = features["sent1"]
        # sent2 = features["sent2"]
        # sent3 = features["sent3"]
        #
        # mask1 = features["mask1"]
        # mask2 = features["mask2"]
        # mask3 = features["mask3"]
        #
        # mark0 = features["mark0"]
        # mark1 = features["mark1"]
        # mark2 = features["mark2"]
        # mark3 = features["mark3"]
        #
        # q_type = features["q_type"]
        # labels = features["labels"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, logits, per_loss = _create_model(is_training, features, num_labels, config)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_warmup_steps = int(num_train_steps * 0.1)
            # train_op = create_optimizer(total_loss, learning_rate,
            #                             num_train_steps, num_warmup_steps,
            #                             False)
            global_step = tf.train.get_or_create_global_step()
            optimizer = tf.train.AdamOptimizer(learning_rate)
            grads = optimizer.compute_gradients(total_loss)
            gradients, tvars = zip(*grads)
            (clip_grads, _) = tf.clip_by_global_norm(gradients, clip_norm=5.0)

            train_op = optimizer.apply_gradients(
                zip(clip_grads, tvars), global_step=global_step)

            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {"logits": logits, "labels": features["labels"], "per_loss": per_loss}
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        else:
            raise ValueError("\n\n=======================\n\tMODE error!!!\n=======================\n\n")

        return output_spec

    return model_fn


def pad_examples(filename, statistic):

    def _loop_load(_fr):
        while True:
            try:
                yield pkl.load(_fr)
            except EOFError:
                break

    def _pad():
        tag = True
        with open(filename, "rb") as fr:
            for input_extract, q_type, label_ids, layer_output in _loop_load(fr):
                sent1 = layer_output[input_extract == 1]
                sent2 = layer_output[input_extract == 2]
                sent3 = layer_output[input_extract == 3]

                mark0 = layer_output[input_extract == 4]
                mark1 = layer_output[input_extract == 5]
                mark2 = layer_output[input_extract == 6]
                mark3 = layer_output[input_extract == 7]

                mark0 = np.squeeze(mark0)
                mark1 = np.squeeze(mark1)
                mark2 = np.squeeze(mark2)
                mark3 = np.squeeze(mark3)

                mask1 = [1. for _ in range(sent1.shape[0])]
                mask1 += [0.] * (statistic["sent1_length"] - sent1.shape[0])
                mask2 = [1. for _ in range(sent2.shape[0])]
                mask2 += [0.] * (statistic["sent2_length"] - sent2.shape[0])
                mask3 = [1. for _ in range(sent3.shape[0])]
                mask3 += [0.] * (statistic["sent3_length"] - sent3.shape[0])

                # if tag:
                #     print("\n================")
                #     print(type(sent1), type(sent1[0]), type(sent1[0][0]))
                #     print("sent1: ", sent1.shape)
                #     print("sent2: ", sent2.shape)
                #     print("sent3: ", sent3.shape)
                #     print("mark0: ", mark0.shape)
                #     print("mark1: ", mark1.shape)
                #     print("mark2: ", mark2.shape)
                #     print("mark3: ", mark3.shape)
                #     print("qtype: ", q_type.shape, q_type, type(q_type))
                #     print("label: ", label_ids.shape, label_ids, type(label_ids))
                #     print("================\n")
                #     tag = False

                def _add_zero_vec(sent, max_len):
                    _len, _dim = sent.shape
                    if _len >= max_len:
                        return list(sent)

                    zero_vec = np.zeros([max_len - _len, _dim], dtype=np.float32)
                    res = np.concatenate([sent, zero_vec], axis=0)
                    assert res.shape[0] == max_len

                    return res.tolist()

                sent1 = _add_zero_vec(sent1, statistic["sent1_length"])
                sent2 = _add_zero_vec(sent2, statistic["sent2_length"])
                sent3 = _add_zero_vec(sent3, statistic["sent3_length"])

                yield {"sent1": sent1, "sent2": sent2, "sent3": sent3,
                       "mask1": mask1, "mask2": mask2, "mask3": mask3,
                       "mark0": mark0, "mark1": mark1, "mark2": mark2, "mark3": mark3,
                       "q_type": q_type, "labels": label_ids}
                # yield [sent1, sent2, sent3,
                #        mask1, mask2, mask3,
                #        mar
                # k0, mark1, mark2, mark3,
                #        q_type, label_ids]

    return _pad


def input_fn_builder(filenames,
                     sent1_length, sent2_length, sent3_length,
                     is_training=True):

    all_sent1 = []
    all_sent2 = []
    all_sent3 = []
    all_mask1 = []
    all_mask2 = []
    all_mask3 = []
    all_mark0 = []
    all_mark1 = []
    all_mark2 = []
    all_mark3 = []
    all_qtype = []
    all_label = []

    for filename in filenames:
        with open(filename, "rb") as fr:
            file_inputs = pkl.load(fr)
            all_sent1 += file_inputs[0]
            all_sent2 += file_inputs[1]
            all_sent3 += file_inputs[2]
            all_mask1 += file_inputs[3]
            all_mask2 += file_inputs[4]
            all_mask3 += file_inputs[5]
            all_mark0 += file_inputs[6]
            all_mark1 += file_inputs[7]
            all_mark2 += file_inputs[8]
            all_mark3 += file_inputs[9]
            all_qtype += file_inputs[10]
            all_label += file_inputs[11]

    all_sent1 = np.concatenate(all_sent1, axis=0)
    all_sent2 = np.concatenate(all_sent2, axis=0)
    all_sent3 = np.concatenate(all_sent3, axis=0)
    all_mask1 = np.concatenate(all_mask1, axis=0)
    all_mask2 = np.concatenate(all_mask2, axis=0)
    all_mask3 = np.concatenate(all_mask3, axis=0)
    all_mark0 = np.concatenate(all_mark0, axis=0)
    all_mark1 = np.concatenate(all_mark1, axis=0)
    all_mark2 = np.concatenate(all_mark2, axis=0)
    all_mark3 = np.concatenate(all_mark3, axis=0)

    assert all_sent1.shape[0] == all_sent2.shape[0] & all_sent2.shape[0] == all_sent3.shape[0]
    num_examples = all_sent1.shape[0]

    def input_fn(params):
        batch_size = params["batch_size"]
        dim = params["dim"]

        name_to_features = {
            "sent1": tf.constant(all_sent1, shape=[num_examples, sent1_length, dim], dtype=tf.float32),
            "sent2": tf.constant(all_sent2, shape=[num_examples, sent2_length, dim], dtype=tf.float32),
            "sent3": tf.constant(all_sent3, shape=[num_examples, sent3_length, dim], dtype=tf.float32),

            "mask1": tf.constant(all_mask1, shape=[num_examples, sent1_length], dtype=tf.float32),
            "mask2": tf.constant(all_mask2, shape=[num_examples, sent2_length], dtype=tf.float32),
            "mask3": tf.constant(all_mask3, shape=[num_examples, sent3_length], dtype=tf.float32),

            "mark0": tf.constant(all_mark0, shape=[num_examples, dim], dtype=tf.float32),
            "mark1": tf.constant(all_mark1, shape=[num_examples, dim], dtype=tf.float32),
            "mark2": tf.constant(all_mark2, shape=[num_examples, dim], dtype=tf.float32),
            "mark3": tf.constant(all_mark3, shape=[num_examples, dim], dtype=tf.float32),

            "q_type": tf.constant(all_qtype, [num_examples, 1], dtype=tf.int32),
            "labels": tf.constant(all_label, [num_examples, 1], dtype=tf.int32)
        }

        d = tf.data.Dataset.from_tensor_slices(name_to_features)
        # d = tf.data.Dataset.from_generator(generator, output_types, output_shapes)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
            drop_remainder = True
        else:
            drop_remainder = False
        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)

        return d

    return input_fn


def input_fn_builder_v2(filenames,
                        sent1_length, sent2_length, sent3_length,
                        is_training=True):

    statistic = {"sent1_length": sent1_length, "sent2_length": sent2_length, "sent3_length": sent3_length}

    def input_fn(params):
        batch_size = params["batch_size"]
        dim = params["dim"]

        generator = pad_examples(filenames, statistic)
        # output_types = (tf.float32,)*10 + (tf.int32,)*2
        output_types = {"sent1": tf.float32, "sent2": tf.float32, "sent3": tf.float32,
                        "mask1": tf.float32, "mask2": tf.float32, "mask3": tf.float32,
                        "mark0": tf.float32, "mark1": tf.float32, "mark2": tf.float32, "mark3": tf.float32,
                        "q_type": tf.int32, "labels": tf.int32}
        # output_shapes = (tf.TensorShape([sent1_length, dim]),
        #                  tf.TensorShape([sent2_length, dim]),
        #                  tf.TensorShape([sent3_length, dim]),
        #
        #                  tf.TensorShape([sent1_length]),
        #                  tf.TensorShape([sent2_length]),
        #                  tf.TensorShape([sent3_length]),
        #
        #                  tf.TensorShape([dim]),
        #                  tf.TensorShape([dim]),
        #                  tf.TensorShape([dim]),
        #                  tf.TensorShape([dim]),
        #
        #                  tf.TensorShape([1]),
        #                  tf.TensorShape([1]))

        output_shapes = {"sent1": tf.TensorShape([sent1_length, dim]),
                         "sent2": tf.TensorShape([sent2_length, dim]),
                         "sent3": tf.TensorShape([sent3_length, dim]),

                         "mask1": tf.TensorShape([sent1_length]),
                         "mask2": tf.TensorShape([sent2_length]),
                         "mask3": tf.TensorShape([sent3_length]),

                         "mark0": tf.TensorShape([dim]),
                         "mark1": tf.TensorShape([dim]),
                         "mark2": tf.TensorShape([dim]),
                         "mark3": tf.TensorShape([dim]),

                         "q_type": tf.TensorShape([]),
                         "labels": tf.TensorShape([])}
        d = tf.data.Dataset.from_generator(generator, output_types, output_shapes)

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
            drop_remainder = True
        else:
            drop_remainder = False
        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)

        return d

    return input_fn
