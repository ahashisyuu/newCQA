import os
import pickle as pkl
import numpy as np

from tensorflow.python.training.training_util import _get_or_create_global_step_read as get_global_step
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer
from tensorflow.python.training.session_run_hook import SessionRunArgs, SessionRunHook

from estimator_utils import input_fn_builder
from utils import PRF, eval_reranker, print_metrics


def read_cid(filenames, cid_datafile="./data/cid.pkl"):
    all_cid = []

    with open(cid_datafile, 'rb') as fr:
        dataset = pkl.loads(fr)

    for name in filenames:
        all_cid += dataset[name]

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

        self.filenames = [os.path.join(basic_dir, a) + ".tfrecord" for a in filenames]
        self.sent1_length = sent1_length
        self.sent2_length = sent2_length
        self.sent3_length = sent3_length
        self.dev_cid = read_cid(filenames, os.path.join(basic_dir, "cid.pkl"))

        self._timer = SecondOrStepTimer(every_steps=eval_steps)
        self._steps_per_run = 1
        self._global_step_tensor = None

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
        if self._timer.should_trigger_for_step(
                stale_global_step + self._steps_per_run):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)
                self.evaluation()

    def end(self, session):
        last_step = session.run(self._global_step_tensor)
        if last_step != self._timer.last_triggered_step():
            self.evaluation()

    def evaluation(self):
        eval_input_fn = input_fn_builder(filenames=self.filenames,
                                         sent1_length=self.sent1_length,
                                         sent2_length=self.sent2_length,
                                         sent3_length=self.sent3_length,
                                         is_training=False)

        every_prediction = self.estimator.predict(eval_input_fn, yield_single_examples=True)

        labels = []
        predictions = []
        for result in every_prediction:
            labels.append(result["labels"])
            predictions.append(result["logits"])
        labels = np.array(labels)
        predictions = np.array(predictions)

        metrics = PRF(labels, predictions.argmax(axis=-1))

        MAP, AvgRec, MRR = eval_reranker(self.dev_cid, self.dev_label, predictions[:, 0])

        metrics['MAP'] = MAP
        metrics['AvgRec'] = AvgRec
        metrics['MRR'] = MRR

        print_metrics(metrics, 'dev')
