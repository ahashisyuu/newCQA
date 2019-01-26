import tensorflow as tf
import collections
from tensorflow.contrib.data import map_and_batch, parallel_interleave


def main():
    filename = "test2.tfrecord"
    writer = tf.python_io.TFRecordWriter(filename)

    def create_int_feature(values):
        feature = tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(values)))
        return feature

    features = collections.OrderedDict()
    # features['unique_ids'] = create_int_feature(['m'])
    features['input_ids'] = create_int_feature([9, 5, 9, 0, 0, 0])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())

    features = collections.OrderedDict()
    # features['unique_ids'] = create_int_feature(['m'])
    features['input_ids'] = create_int_feature([15, 5, 9, 0, 0, 0])
    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
    writer.close()


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    print(example)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def read():
    filename = "test.tfrecord"
    filename2 = "test2.tfrecord"
    # dataset = tf.data.TFRecordDataset(filename)
    name_to_features = {"input_ids": tf.FixedLenFeature([6], tf.int64)}
    d1 = tf.data.Dataset.from_tensor_slices(tf.constant([filename, filename2]))
    d1 = d1.apply(
        tf.contrib.data.parallel_interleave(
            tf.data.TFRecordDataset,
            sloppy=False,
            cycle_length=2))
    # d1 = tf.data.TFRecordDataset(filename)
    # print(d1)
    d2 = d1.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=1,
            drop_remainder=True))
    # print(d2)
    iterator = d2.make_initializable_iterator()
    # print(iterator)
    results = iterator.get_next()
    print(results)
    sess = tf.Session()

    sess.run(iterator.initializer)

    # print(results)
    input_ids = results['input_ids']
    # output = input_ids
    print(sess.run(results['input_ids']))
    # print(sess.run(output))
    # print(sess.run(output))


if __name__ == "__main__":
    # main()
    read()
