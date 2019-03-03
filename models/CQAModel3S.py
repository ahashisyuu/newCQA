import os
import argparse
import numpy as np
import pickle as pkl
import tensorflow as tf

from tqdm import tqdm
from EvalHook import EvalHook
from BertCQAModel import BertCQAModel
from layers.optimization import create_optimizer


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="./models/bert_model")
parser.add_argument("--save_checkpoints_steps", type=int, default=1000)

parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--num_train_steps", type=int, default=50000)
parser.add_argument("--batch_size", type=int, default=2)

parser.add_argument("--train_filenames", type=str, default="bert_train")
parser.add_argument("--dev_filenames", type=str, default="bert_cqa_eval")

parser.add_argument("--max_sent1_length", type=int, default=39)
parser.add_argument("--max_sent2_length", type=int, default=110)
parser.add_argument("--max_sent3_length", type=int, default=152)

parser.add_argument("--keys_num", type=int, default=6)
parser.add_argument("--update_num", type=int, default=3)


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


def glove_embedding(filename):
    with tf.variable_scope("embedding"):
        embedding_matrix = pkl.load(filename)
        matrix_tensor = tf.get_variable("embed_matrix",
                                        initializer=tf.constant(embedding_matrix, dtype=tf.float32),
                                        trainable=True)

        def _embedding(sent):
            return tf.nn.embedding_lookup(matrix_tensor, sent)

    return _embedding


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
        # q_type = features["q_type"]
        # labels = features["labels"]

        features["mark0"] = features["sent1"]  # (B, L1)
        features["mark1"] = None
        features["mark2"] = None
        features["mark3"] = None

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # EMBEDDING
        embed_func = glove_embedding(config.embed_file)
        features["sent1"] = embed_func(features["sent1"])
        features["sent2"] = embed_func(features["sent2"])
        features["sent3"] = embed_func(features["sent3"])

        total_loss, logits, per_loss = _create_model(is_training, features, num_labels, config)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_warmup_steps = int(num_train_steps * 0.1)
            train_op = create_optimizer(total_loss, learning_rate,
                                        num_train_steps, num_warmup_steps,
                                        False)
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {"logits": logits, "labels": features["labels"], "per_loss": per_loss}
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        else:
            raise ValueError("\n\n=======================\n\tMODE error!!!\n=======================\n\n")

        return output_spec

    return model_fn


def get_examples(filename):
    with open(filename, "rb") as fr:
        pandas_data = pkl.load(filename)
        # padding and making mask
        return {"sent1": sent1, "sent2": sent2, "sent3": sent3,
                "mask1": mask1, "mask2": mask2, "mask3": mask3,
                "q_type": q_type, "labels": label_ids}


def input_fn_builder(filenames,
                     sent1_length, sent2_length, sent3_length,
                     is_training=True):

    all_data = get_examples(filenames)

    def input_fn(params):
        batch_size = params["batch_size"]

        d = tf.data.Dataset.from_tensor_slices({
            "sent1":
                tf.constant(value=all_data["sent1"], shape=[sent1_length], dtype=tf.int32),
            "sent2":
                tf.constant(value=all_data["sent2"], shape=[sent2_length], dtype=tf.int32),
            "sent3":
                tf.constant(value=all_data["sent3"], shape=[sent3_length], dtype=tf.int32),

            "mask1":
                tf.constant(value=all_data["mask1"], shape=[sent1_length], dtype=tf.float32),
            "mask2":
                tf.constant(value=all_data["mask2"], shape=[sent2_length], dtype=tf.float32),
            "mask3":
                tf.constant(value=all_data["mask3"], shape=[sent3_length], dtype=tf.float32),

            "q_type": tf.constant(value=all_data["q_type"], shape=[], dtype=tf.int32),
            "labels": tf.constant(value=all_data["labels"], shape=[], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
            drop_remainder = True
        else:
            drop_remainder = False
        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)

        return d

    return input_fn


def main(args):
    tf.logging.set_verbosity(tf.logging.ERROR)

    session_config = tf.ConfigProto(log_device_placement=False)
    session_config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(model_dir=args.model_dir,
                                        save_checkpoints_steps=args.save_checkpoints_steps,
                                        session_config=session_config)

    model_fn = model_fn_builder(num_labels=2, learning_rate=args.lr, num_train_steps=args.num_train_steps, config=args)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       config=run_config,
                                       params={"batch_size": args.batch_size,
                                               "dim": 768})

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", args.batch_size)

    train_input_fn = input_fn_builder(filenames=os.path.join("./data", args.train_filenames) + ".pkl",
                                      sent1_length=args.max_sent1_length,
                                      sent2_length=args.max_sent2_length,
                                      sent3_length=args.max_sent3_length,
                                      is_training=True)

    eval_hook = EvalHook(estimator=estimator,
                         filenames=args.dev_filenames,
                         sent1_length=args.max_sent1_length,
                         sent2_length=args.max_sent2_length,
                         sent3_length=args.max_sent3_length,
                         eval_steps=args.save_checkpoints_steps,
                         basic_dir="./data")

    estimator.train(input_fn=train_input_fn, max_steps=args.num_train_steps,
                    hooks=[eval_hook])


if __name__ == "__main__":
    main(parser.parse_args())


