import tensorflow as tf
import keras.backend as K
from .CQAModel import CQAModel
from layers.BiGRU import NativeGRU as BiGRU
import keras.losses


class Baseline3_2(CQAModel):





    def operation(self,input1,input2):
        mult=tf.multiply(input1,input2)
        sub=tf.subtract(input1,input2)
        out_=tf.concat([mult,sub],axis=-1)
        return out_

    def build_model(self):
        print('base3')
        with tf.variable_scope('baseline', initializer=tf.glorot_uniform_initializer()):
            units = 300
            Q, C = self.QS, self.CT
            Q_len, C_len = self.Q_len, self.C_len

            with tf.variable_scope('encode'):
                rnn1 = BiGRU(num_layers=1, num_units=units,
                             batch_size=tf.shape(self.QS)[0], input_size=self.CT.get_shape()[-1],
                             keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='q')
                rnn2 = BiGRU(num_layers=1, num_units=units,
                             batch_size=tf.shape(self.QS)[0], input_size=self.CT.get_shape()[-1],
                             keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='c')

                Q_sequence = rnn1(Q, seq_len=Q_len, return_type=1)
                C_sequence = rnn2(C, seq_len=C_len, return_type=1)
                print(Q_sequence.shape)


            with tf.variable_scope('DecomposableAttention',initializer=tf.glorot_uniform_initializer()):
                Q1=Q_sequence
                Q2=C_sequence
                l1=Q1.shape[1]
                l2=Q2.shape[1]
                demsion=Q1.shape[2]
                Kernel1=tf.Variable(tf.zeros([demsion,demsion]))
                Kernel2=tf.Variable(tf.zeros([demsion,1]))
                Q1_=tf.expand_dims(Q1,2) #(B,L1,1,D)
                Q2_=tf.expand_dims(Q2,1) #(B,1,L2,D)
                Q12s=Q1_-Q2_
                print(Q12s)
                matrix=K.dot(tf.abs(Q12s),Kernel1)
                matrix=tf.tanh(matrix)
                matrix=K.dot(matrix,Kernel2)
                similarity_matrix = tf.squeeze(matrix, axis=-1)
                print(similarity_matrix.shape)
                w_att_1=tf.nn.softmax(similarity_matrix,axis=1)
                w_att_2=tf.transpose(tf.nn.softmax(similarity_matrix,axis=2),perm=[0,2,1])

                Q1_aligned=K.batch_dot(w_att_1,Q_sequence,axes=[1,1])
                Q2_aligned=K.batch_dot(w_att_2,C_sequence,axes=[1,1])

                Q_concat=tf.concat([Q_sequence,Q2_aligned,self.operation(Q_sequence,Q2_aligned)],axis=-1)
                C_concat=tf.concat([C_sequence,Q1_aligned,self.operation(C_sequence,Q1_aligned)],axis=-1)
                print('Q_concat',Q_concat.get_shape())
            with tf.variable_scope('encode2'):
                rnn1 = BiGRU(num_layers=1, num_units=units,
                             batch_size=tf.shape(Q_concat)[0], input_size=Q_concat.get_shape()[-1],
                             keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='q')
                rnn2 = BiGRU(num_layers=1, num_units=units,
                             batch_size=tf.shape(C_concat)[0], input_size=C_concat.get_shape()[-1],
                             keep_prob=self.dropout_keep_prob, is_train=self._is_train, scope='c')
                Q_encode2 = rnn1(Q_concat, seq_len=Q_len, return_type=1)
                C_encode2 = rnn2(C_concat, seq_len=C_len, return_type=1)

                print('encode2 ',Q_encode2)
                Q_vec_mean = tf.reduce_mean(Q_encode2, axis=1)
                C_vec_mean = tf.reduce_mean(C_encode2, axis=1)
                Q_vec_max = tf.reduce_max(Q_encode2, axis=1)
                C_vec_max= tf.reduce_max(C_encode2,axis=1)

                info = tf.concat([Q_vec_mean, C_vec_mean,Q_vec_max,C_vec_max], axis=-1)
                median = tf.layers.dense(info, 300, activation=tf.tanh)
                output = tf.layers.dense(median, 2, activation=tf.identity)

            return output






