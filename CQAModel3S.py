import os
import argparse
import numpy as np
import pickle as pkl
import tensorflow as tf

from tqdm import tqdm
from BertCQAModel import BertCQAModel
from layers.optimization import create_optimizer
from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.training.training_util import _get_or_create_global_step_read as get_global_step
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.session_run_hook import SessionRunArgs, SessionRunHook
from utils import PRF, eval_reranker, print_metrics


os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="./models/triangular_model")
parser.add_argument("--save_checkpoints_steps", type=int, default=1000)

parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--num_train_steps", type=int, default=50000)
parser.add_argument("--batch_size", type=int, default=2)

parser.add_argument("--train_filenames", type=str, default="datasetFORtri")
parser.add_argument("--dev_filenames", type=str, default="datasetFORtri")
parser.add_argument("--embed_file", type=str, default="./assistFORtri/embedding_matrix_lemma.pkl")

parser.add_argument("--max_sent1_length", type=int, default=30)
parser.add_argument("--max_sent2_length", type=int, default=110)
parser.add_argument("--max_sent3_length", type=int, default=150)

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
    with tf.variable_scope("embedding"), open(filename, 'rb') as fr:
        embedding_matrix = pkl.load(fr)
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

        # features["mark0"] = features["sent1"]  # (B, L1)
        _features = dict()
        _features["mark1"] = None
        _features["mark2"] = None
        _features["mark3"] = None

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        # EMBEDDING
        embed_func = glove_embedding(config.embed_file)
        _features["sent1"] = embed_func(features["sent1"])
        _features["sent2"] = embed_func(features["sent2"])
        _features["sent3"] = embed_func(features["sent3"])
        _features["mask1"] = features["mask1"]
        _features["mask2"] = features["mask2"]
        _features["mask3"] = features["mask3"]
        _features["q_type"] = features["q_type"]
        _features["labels"] = features["labels"]

        _features["mark0"] = _features["sent1"][:, 0, :]  # (B, dim)

        total_loss, logits, per_loss = _create_model(is_training, _features, num_labels, config)

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


def get_examples(filename,
                 sent1_length, sent2_length, sent3_length,
                 is_training):
    with open(filename, "rb") as fr:
        pandas_data = pkl.load(fr)  # the dict of DataFrame

        # keys_list = pandas_data.keys()
        train_list = ["15dev.xml", "15test.xml", "15train.xml",
                      "16dev.xml", "16test.xml", "16train1.xml", "16train2.xml"]

        sent1 = []
        sent2 = []
        sent3 = []
        q_type = []
        label_ids = []
        if is_training:
            for name in train_list:
                samples = pandas_data[name]

                sent1 += samples["q_sub_lemma_index"].values.tolist()
                sent2 += samples["q_body_lemma_index"].values.tolist()
                sent3 += samples["cTEXT_lemma_index"].values.tolist()

                q_type += samples["cate_index"].values.tolist()
                label_ids += samples["Rrel_index"].values.tolist()

        else:
            samples = pandas_data["17test.xml"]

            sent1 += samples["q_sub_lemma_index"].values.tolist()
            sent2 += samples["q_body_lemma_index"].values.tolist()
            sent3 += samples["cTEXT_lemma_index"].values.tolist()

            q_type += samples["cate_index"].values.tolist()
            label_ids += samples["Rrel_index"].values.tolist()

        # padding and making mask
        ls1 = len(sent1)
        ls2 = len(sent2)
        ls3 = len(sent3)
        lq = len(q_type)
        ll = len(label_ids)
        assert ls1 == ls2 and ls2 == ls3 and ls3 == lq and lq == ll

        sent1 = pad_sequences(sent1, maxlen=sent1_length,
                              padding="post", truncating="post")  # (N, L1)
        sent2 = pad_sequences(sent2, maxlen=sent2_length,
                              padding="post", truncating="post")
        sent3 = pad_sequences(sent3, maxlen=sent3_length,
                              padding="post", truncating="post")

        for s1, s2, s3 in zip(sent1, sent2, sent3):
            for a in s1.tolist():
                assert type(a) == int
            for a in s2.tolist():
                assert type(a) == int
            for a in s3.tolist():
                assert type(a) == int

        mask1 = (sent1 != 0).astype(dtype=np.float32)
        mask2 = (sent2 != 0).astype(dtype=np.float32)
        mask3 = (sent3 != 0).astype(dtype=np.float32)

        # convert to numpy
        q_type = np.asarray(q_type)
        label_ids = np.asarray(label_ids)

        return {"sent1": sent1, "sent2": sent2, "sent3": sent3,
                "mask1": mask1, "mask2": mask2, "mask3": mask3,
                "q_type": q_type, "labels": label_ids, "samples_num": ls1}


def input_fn_builder(filenames,
                     sent1_length, sent2_length, sent3_length,
                     is_training=True):

    all_data = get_examples(filenames,
                            sent1_length, sent2_length, sent3_length,
                            is_training)

    samples_num = all_data["samples_num"]

    def input_fn(params):
        batch_size = params["batch_size"]

        d = tf.data.Dataset.from_tensor_slices({
            "sent1":
                tf.constant(value=all_data["sent1"], shape=[samples_num, sent1_length], dtype=tf.int32),
            "sent2":
                tf.constant(value=all_data["sent2"], shape=[samples_num, sent2_length], dtype=tf.int32),
            "sent3":
                tf.constant(value=all_data["sent3"], shape=[samples_num, sent3_length], dtype=tf.int32),

            "mask1":
                tf.constant(value=all_data["mask1"], shape=[samples_num, sent1_length], dtype=tf.float32),
            "mask2":
                tf.constant(value=all_data["mask2"], shape=[samples_num, sent2_length], dtype=tf.float32),
            "mask3":
                tf.constant(value=all_data["mask3"], shape=[samples_num, sent3_length], dtype=tf.float32),

            "q_type": tf.constant(value=all_data["q_type"], shape=[samples_num, ], dtype=tf.int32),
            "labels": tf.constant(value=all_data["labels"], shape=[samples_num, ], dtype=tf.int32),
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


def read_cid(cid_datafile="./data/eval_cid.pkl"):

    with open(cid_datafile, 'rb') as fr:
        all_cid = pkl.load(fr)

    return all_cid


class EvalHook(SessionRunHook):
    def __init__(self,
                 estimator,
                 filenames,
                 sent1_length,
                 sent2_length,
                 sent3_length,
                 eval_steps=None,
                 basic_dir="./data"):

        logging.info("Create EvalHook.")
        self.estimator = estimator

        self.filenames = os.path.join(basic_dir, filenames) + ".pkl"
        self.sent1_length = sent1_length
        self.sent2_length = sent2_length
        self.sent3_length = sent3_length
        self.dev_cid = read_cid(os.path.join(basic_dir, "eval_cid.pkl"))

        self._timer = SecondOrStepTimer(every_steps=eval_steps)
        self._steps_per_run = 1
        self._global_step_tensor = None
        self.global_step = None

    def _set_steps_per_run(self, steps_per_run):
        self._steps_per_run = steps_per_run

    def begin(self):
        self._global_step_tensor = get_global_step()  # pylint: disable=protected-access
        if self._global_step_tensor is None:
            raise RuntimeError(
                "Global step should be created to use EvalHook.")

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results
        print("global_step: ", stale_global_step)
        if self._timer.should_trigger_for_step(
                stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self.global_step = global_step
                self._timer.update_last_triggered_step(global_step)
                self.evaluation()

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        if last_step != self._timer.last_triggered_step():
            self.global_step = last_step
            self.evaluation()

    def evaluation(self):
        print("")
        eval_input_fn = input_fn_builder(filenames=self.filenames,
                                         sent1_length=self.sent1_length,
                                         sent2_length=self.sent2_length,
                                         sent3_length=self.sent3_length,
                                         is_training=False)

        every_prediction = self.estimator.predict(eval_input_fn, yield_single_examples=True)

        labels = []
        predictions = []
        losses = []
        for result in every_prediction:
            labels.append(result["labels"])
            predictions.append(result["logits"])
            losses.append(result["per_loss"])

        labels = np.array(labels)
        predictions = np.array(predictions)
        losses = np.array(losses)

        total_loss = losses.mean()

        print(type(labels), labels.shape)
        print(type(predictions), predictions.shape)

        metrics = PRF(labels, predictions.argmax(axis=-1))

        MAP, AvgRec, MRR = eval_reranker(self.dev_cid, labels, predictions[:, 0])

        metrics['loss'] = total_loss
        metrics['step'] = self.global_step
        metrics['MAP'] = MAP
        metrics['AvgRec'] = AvgRec
        metrics['MRR'] = MRR

        print_metrics(metrics, 'dev')


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
                                               "dim": 300})

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


