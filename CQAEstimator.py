import os
import argparse
import tensorflow as tf

from EvalHook import EvalHook
from estimator_utils import model_fn_builder, input_fn_builder_v2 as input_fn_builder

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
parser.add_argument("--update_num", type=int, default=1)


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


