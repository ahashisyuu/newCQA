import tensorflow as tf
import collections


def main(args):
    writer = tf.python_io.TFRecordWriter(args.filename)


if __name__ == "__main__":
    main(args)
