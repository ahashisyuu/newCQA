import os
import random

import numpy as np
import tensorflow as tf
import keras.losses
from utils import PRF, print_metrics, eval_reranker
from tqdm import tqdm
from data import BatchDatasets

EPSILON = 1e-7


class CQAModel:

    def __init__(self, embedding_matrix, args, margin=0.25, char_num=128):
        # session info
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        seed = random.randint(1, 600)
        print(seed)
        tf.set_random_seed(seed)

        # hyper parameters and neccesary info
        print(embedding_matrix.shape)
        self._is_train = tf.get_variable('is_train', shape=[], dtype=tf.bool, trainable=False)
        self.word_mat = tf.get_variable('word_mat',
                                        initializer=tf.constant(embedding_matrix, dtype=tf.float32),
                                        trainable=args.word_trainable)
        self._global_step = tf.get_variable('global_step', shape=[], dtype=tf.int32,
                                            initializer=tf.constant_initializer(0), trainable=False)
        self.dropout_keep_prob = tf.get_variable('dropout', shape=[], dtype=tf.float32,
                                                 initializer=tf.constant_initializer(1), trainable=False)
        self._lr = tf.get_variable('lr', shape=[], dtype=tf.float32,
                                   initializer=tf.constant_initializer(0.001), trainable=False)
        self._cweight = tf.get_variable('cweight', dtype=tf.float32,
                                        initializer=tf.constant([1. for _ in range(2)]),
                                        trainable=False)
        self.args = args
        self.char_num = char_num
        self.N = None

        # batch input
        self.inputs = self.QText, self.Q_len, self.PosCText, self.PosC_len, self.NegCText, self.NegC_len, self.Qcate = self.create_input()

        # preparing mask and length info
        self.Q_mask = tf.cast(tf.cast(self.QText, tf.bool), tf.float32)
        self.PosC_mask = tf.cast(tf.cast(self.PosCText, tf.bool), tf.float32)
        self.NegC_mask = tf.cast(tf.cast(self.NegCText, tf.bool), tf.float32)
        self.Q_maxlen = tf.reduce_max(self.Q_len)
        self.PosC_maxlen = tf.reduce_max(self.PosC_len)
        self.NegC_maxlen = tf.reduce_max(self.NegC_len)

        self.N = tf.shape(self.Q_mask)[0]

        # embedding word vector and char vector
        self.QS, self.pos_CT, self.neg_CT, self.cate_f = self.embedding()

        # building model
        self.pos_score, self.neg_score = self.build_model()

        # computing loss
        with tf.variable_scope('loss'):
            original_loss = tf.reduce_sum(margin - self.pos_score + self.neg_score)
            self.loss = tf.cond(tf.less(0.0, original_loss), lambda: original_loss, lambda: tf.constant(0.0))
            if self.args.l2_weight != 0:
                for v in tf.trainable_variables():
                    self.loss += self.args.l2_weight * tf.nn.l2_loss(v)

        # counting parameters
        self.count()

        # getting ready for training
        self.opt = tf.train.AdagradOptimizer(learning_rate=self._lr)
        # self.opt = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=1e-6)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)
        self.train_op = self.opt.apply_gradients(zip(capped_grads, variables),
                                                 global_step=self._global_step)

    def create_input(self):
        qTEXT = tf.placeholder(tf.int32, [None, None])
        q_len = tf.placeholder(tf.int32, [None])
        pos_cTEXT = tf.placeholder(tf.int32, [None, None])
        pos_c_len = tf.placeholder(tf.int32, [None])
        neg_cTEXT = tf.placeholder(tf.int32, [None, None])
        neg_c_len = tf.placeholder(tf.int32, [None])
        inputs = [qTEXT, q_len, pos_cTEXT, pos_c_len, neg_cTEXT, neg_c_len]

        cate = tf.placeholder(tf.int32, [None])

        return inputs + [cate]

    def embedding(self):
        # word embedding
        with tf.variable_scope('emb'):
            QS = tf.nn.embedding_lookup(self.word_mat, self.QText)
            pos_CT = tf.nn.embedding_lookup(self.word_mat, self.PosCText)
            neg_CT = tf.nn.embedding_lookup(self.word_mat, self.NegCText)

            embedded = [QS, pos_CT, neg_CT]

            category_mat = tf.get_variable('cate_mat', shape=[33, 25], dtype=tf.float32,
                                           initializer=tf.glorot_uniform_initializer(), trainable=False)
            cate_f = tf.nn.embedding_lookup(category_mat, self.Qcate)

            return embedded + [cate_f]

    @property
    def cweight(self):
        return self.sess.run(self._cweight)

    @cweight.setter
    def cweight(self, value):
        self.sess.run(tf.assign(self._cweight, tf.constant(value, dtype=tf.float32)))

    @property
    def lr(self):
        return self.sess.run(self._lr)

    @lr.setter
    def lr(self, value):
        assert isinstance(value, float)
        self.sess.run(tf.assign(self._lr, tf.constant(value, dtype=tf.float32)))

    @property
    def dropout(self):
        return 1 - self.sess.run(self.dropout_keep_prob)

    @dropout.setter
    def dropout(self, value):
        assert isinstance(value, float)
        self.sess.run(tf.assign(self.dropout_keep_prob, tf.constant(1-value, dtype=tf.float32)))

    @property
    def global_step(self):
        return self.sess.run(self._global_step)

    @global_step.setter
    def global_step(self, value):
        assert isinstance(value, int)
        self.sess.run(tf.assign(self._global_step, tf.constant(value, dtype=tf.int32)))

    @property
    def is_train(self):
        return self.sess.run(self._is_train)

    @is_train.setter
    def is_train(self, value):
        assert isinstance(value, bool)
        self.sess.run(tf.assign(self._is_train, tf.constant(value, dtype=tf.bool)))

    def build_model(self):
        raise NotImplementedError

    def evaluate(self, eva_data, steps_num, eva_type, eva_ID=None):
        scores = []
        rel = []
        with tqdm(total=steps_num, ncols=70) as tbar:
            for batch_eva_data in eva_data:
                batch_qText, batch_q_len, batch_cText, batch_c_len, batch_cate, batch_rel = batch_eva_data
                feed_dict = {self.QText: batch_qText,
                             self.Q_len: batch_q_len,
                             self.PosCText: batch_cText,
                             self.PosC_len: batch_c_len,
                             self.NegCText: batch_cText,
                             self.NegC_len: batch_c_len,
                             self.Qcate: batch_cate}
                batch_scores = self.sess.run(self.pos_score, feed_dict=feed_dict)
                scores.append(batch_scores)
                rel.append(batch_rel)
                tbar.update(batch_rel.shape[0])

        rel = np.concatenate(rel, axis=0)
        scores = np.concatenate(scores, axis=0)
        MAP, AvgRec, MRR = eval_reranker(eva_ID, rel, scores)
        MAP_summ = tf.Summary(value=[tf.Summary.Value(
            tag="{}/MAP".format(eva_type), simple_value=MAP), ])
        AvgRec_summ = tf.Summary(value=[tf.Summary.Value(
            tag="{}/AvgRec".format(eva_type), simple_value=AvgRec), ])
        MRR_summ = tf.Summary(value=[tf.Summary.Value(
            tag="{}/MRR".format(eva_type), simple_value=MRR), ])
        return {'MAP': MAP, 'AvgRec': AvgRec, 'MRR': MRR}, \
               [MAP_summ, AvgRec_summ, MRR_summ]

    def one_train(self, batch_dataset, saver, writer, config, fold_num=None):
        MAP_save = 100
        patience = 0
        self.lr = config.lr
        self.dropout = config.dropout

        print('---------------------------------------------')
        print('process train data')
        train_data = [batch for batch in batch_dataset.batch_train_triplets(train_files=config.train_list, batch_size=config.batch_size)]
        train_steps = batch_dataset.train_steps_num
        print('---------------------------------------------')

        print('class weight: ', batch_dataset.cweight)
        # self.cweight = [.9, 5., 1.1]

        print('\n---------------------------------------------')
        print('process dev data')
        dev_data = [batch for batch in batch_dataset.batch_dev_data()]
        dev_steps = batch_dataset.dev_steps_num
        dev_id = batch_dataset.c_id_dev
        print('----------------------------------------------\n')
        # print(list(zip(dev_data[0][0], dev_data[0][-1])))

        for epoch in range(1, config.epochs + 1):
            print('---------------------------------------')
            print('EPOCH %d' % epoch)

            print('the number of samples: %d\n' % train_steps)

            print('training model')
            self.is_train = True

            with tqdm(total=train_steps, ncols=70) as tbar:
                for batch_train_data in train_data:
                    feed_dict = {inv: array for inv, array in zip(self.inputs, batch_train_data)}
                    loss, train_op = self.sess.run([self.loss, self.train_op], feed_dict=feed_dict)

                    if self.global_step % config.period == 0:
                        loss_sum = tf.Summary(value=[tf.Summary.Value(tag="model/loss", simple_value=loss)])
                        writer.add_summary(loss_sum, self.global_step)

                        print('\n---------------------------------------')
                        print('\nevaluating model\n')
                        self.is_train = False

                        val_metrics, summ = self.evaluate(dev_data, dev_steps, 'dev', dev_id)
                        val_metrics['epoch'] = epoch

                        if val_metrics['MAP'] < MAP_save:
                            MAP_save = val_metrics['MAP']
                            patience = 0
                        else:
                            patience += 1

                        if patience >= config.patience:
                            self.lr = self.lr / 2.0
                            MAP_save = val_metrics['MAP']
                            patience = 0

                        for s in summ:
                            writer.add_summary(s, self.global_step)
                        writer.flush()

                        path = os.path.join(config.model_dir, self.__class__.__name__)
                        if not os.path.exists(path):
                            os.mkdir(path)
                        if fold_num is not None:
                            path = os.path.join(path, 'fold_%d' % fold_num)
                            if not os.path.exists(path):
                                os.mkdir(path)
                        filename = os.path.join(path, "epoch{0}_MAP{1:.4f}_Avg{2:.4f}_MRR{3:.4f}.model"
                                                .format(epoch, val_metrics['MAP'],
                                                        val_metrics['AvgRec'], val_metrics['MRR']))

                        saver.save(self.sess, filename)

                    tbar.update(batch_dataset.batch_size)

    def train(self, batch_dataset: BatchDatasets, config):
        with self.sess:
            saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

            path = os.path.join(config.model_dir, self.__class__.__name__)
            if not os.path.exists(config.model_dir):
                os.mkdir(config.model_dir)

            if config.load_best_model and os.path.exists(path):
                print('------------  load model  ------------')
                saver.restore(self.sess, tf.train.latest_checkpoint(path))
            if not os.path.exists(path):
                os.mkdir(path)
            writer = tf.summary.FileWriter(path)
            self.one_train(batch_dataset, saver, writer, config)

    def one_test(self, batch_dataset, config):
        test_data = [batch for batch in batch_dataset.batch_test_data(2 * config.batch_size)]
        steps = batch_dataset.test_steps_num
        cID = batch_dataset.c_id_test
        self.is_train = False
        self.cweight = [1., 1., 1.]
        test_metrics, _ = self.evaluate(test_data, steps, 'test', cID)
        print_metrics(test_metrics, 'test')

    def test(self, batch_dataset: BatchDatasets, config):
        with self.sess:
            self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if config.k_fold > 1:
                sub_dir = os.listdir(config.model_dir)
                for name in sub_dir:
                    path = os.path.join(config.model_dir, name)
                    print(tf.train.latest_checkpoint(config.model_dir))
                    saver.restore(self.sess, tf.train.latest_checkpoint(path))
                    self.one_test(batch_dataset, config)
            else:
                print(tf.train.latest_checkpoint(config.model_dir))
                saver.restore(self.sess, tf.train.latest_checkpoint(config.model_dir))
                self.one_test(batch_dataset, config)

    def count(self):
        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('\n\n------------------------------------------------')
        print('total_parameters: ', total_parameters)
        print('------------------------------------------------\n\n')









