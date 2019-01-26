import os
import argparse
import tensorflow as tf

from EvalHook import EvalHook
from estimator_utils import model_fn_builder, input_fn_builder

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="")

def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)

    session_config = tf.ConfigProto(log_device_placement=True)
    session_config.gpu_options.allow_growth = True

    run_config = tf.estimator.RunConfig(model_dir=args.model_dir,
                                        save_checkpoints_steps=args.save_checkpoints_steps,
                                        session_config=session_config)

    model_fn = model_fn_builder(num_labels=2, learning_rate=args.lr, num_train_steps=args.num_train_steps)

    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       config=run_config,
                                       params={"batch_size": args.batch_size})

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", args.batch_size)

    train_input_fn = input_fn_builder(filenames=args.train_filenames,
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

    estimator.train(input_fn=train_input_fn, max_steps=args.max_train_steps,
                    hooks=[eval_hook])





