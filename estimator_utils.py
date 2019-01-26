import os
import tensorflow as tf

from tensorflow.contrib import data
from tensorflow.contrib.layers import xavier_initializer

from BertCQAModel import BertCQAModel
from layers.optimization import create_optimizer


def _create_model(is_training, features, num_labels, config):

    # build model
    model = BertCQAModel(is_training=is_training, features=features, config=config)

    relation_vec = model.get_output()  # (B, D)
    hidden_size = relation_vec.shape[-1].value

    relation_weights = tf.get_variable(
        "relation_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    relation_bias = tf.get_variable(
        "relation_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            relation_vec = tf.nn.dropout(relation_vec, keep_prob=0.9)

        logits = tf.matmul(relation_vec, relation_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, relation_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(features["labels"], num_labels, dtype=tf.float32)

        per_loss = - tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        total_loss = tf.reduce_mean(per_loss)

    return total_loss, logits


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


def model_fn_builder(num_labels, learning_rate, num_train_steps):

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

        total_loss, logits = _create_model(is_training, features, num_labels)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_warmup_steps = int(num_train_steps * 0.1)
            train_op = create_optimizer(total_loss, learning_rate,
                                        num_train_steps, num_warmup_steps,
                                        False)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {"logits": logits, "labels": features["labels"]}
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        else:
            raise ValueError("\n\n=======================\n\tMODE error!!!\n=======================\n\n")

        return output_spec

    return model_fn


def input_fn_builder(filenames,
                     sent1_length, sent2_length, sent3_length,
                     is_training=True):

    def input_fn(params):
        batch_size = params["batch_size"]
        dim = params["dim"]

        name_to_features = {
            "sent1": tf.FixedLenFeature([sent1_length, dim], tf.float32),
            "sent2": tf.FixedLenFeature([sent2_length, dim], tf.float32),
            "sent3": tf.FixedLenFeature([sent3_length, dim], tf.float32),

            "mask1": tf.FixedLenFeature([sent1_length], tf.float32),
            "mask2": tf.FixedLenFeature([sent2_length], tf.float32),
            "mask3": tf.FixedLenFeature([sent3_length], tf.float32),

            "mark0": tf.FixedLenFeature([dim], tf.float32),
            "mark1": tf.FixedLenFeature([dim], tf.float32),
            "mark2": tf.FixedLenFeature([dim], tf.float32),
            "mark3": tf.FixedLenFeature([dim], tf.float32),

            "q_type": tf.FixedLenFeature([1], tf.int64),
            "labels": tf.FixedLenFeature([1], tf.int64)
        }

        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(filenames))
            d = d.repeat()
            d = d.shuffle(buffer_size=len(filenames))

            d = d.apply(data.parallel_interleave(tf.data.TFRecordDataset,
                                                 sloppy=True,
                                                 cycle_length=len(filenames)))
            d = d.shuffle(buffer_size=100)
            drop_remainder = True
        else:
            d = tf.data.TFRecordDataset(filenames)
            drop_remainder = False

        d = d.apply(data.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                       batch_size=batch_size,
                                       num_parallel_batches=4,
                                       drop_remainder=drop_remainder))

        return d

    return input_fn

