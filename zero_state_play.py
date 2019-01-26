import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors


values = _zero_state_tensors([5 * 100] * 3, 20, dtype=tf.float32)

print(type(values), values)

values = tf.constant([[[2, 3, 4, 5], [1, 3, 5, 7], [4, 2, 1, 7]],
                      [[1, 5, 7, 2], [9, 4, 3, 5], [4, 2, 3, 1]]])

sess = tf.Session()

print(values.shape)  # (2, 3, 4)
print(sess.run(tf.reshape(values,  [-1, 4])))
print(sess.run(tf.reshape(values, [-1, 3*4])))
print(sess.run(tf.reshape(tf.reshape(values, [-1, 4]), [-1, 3*4])))



