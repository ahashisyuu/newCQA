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

    def __init__(self, embedding_matrix, args, char_num=128, seed=1):
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
        self.inputs = self.QText, self.Q_len, self.CText, self.C_len, self.Qcate, self.Rel = self.create_input()

        # preparing mask and length info
        self.Q_mask = tf.cast(tf.cast(self.QText, tf.bool), tf.float32)
        self.C_mask = tf.cast(tf.cast(self.CText, tf.bool), tf.float32)
        self.Q_maxlen = tf.reduce_max(self.Q_len)
        self.C_maxlen = tf.reduce_max(self.C_len)

        self.N = tf.shape(self.Q_mask)[0]

        # embedding word vector and char vector
        self.QS, self.CT, self.cate_f = self.embedding()

        # building model
        self.output = self.build_model()

        # computing loss
        with tf.variable_scope('predict'):
            self.predict_prob = tf.nn.softmax(self.output)
            labels = tf.one_hot(self.Rel, 2, dtype=tf.float32)
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.output)
            self.loss = tf.reduce_mean(losses)
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
        cTEXT = tf.placeholder(tf.int32, [None, None])
        c_len = tf.placeholder(tf.int32, [None])
        inputs = [qTEXT, q_len, cTEXT, c_len]

        cate = tf.placeholder(tf.int32, [None])
        rel = tf.placeholder(tf.int32, [None])

        return inputs + [cate, rel]

    def embedding(self):
        # word embedding
        with tf.variable_scope('emb'):
            QS = tf.nn.embedding_lookup(self.word_mat, self.QText)
            CT = tf.nn.embedding_lookup(self.word_mat, self.CText)

            embedded = [QS, CT]

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
        label = []
        predict = []
        loss = []
        with tqdm(total=steps_num, ncols=70) as tbar:
            for batch_eva_data in eva_data:
                batch_label = batch_eva_data[-1]
                feed_dict = {inv: array for inv, array in zip(self.inputs, batch_eva_data)}
                batch_loss, batch_predict = self.sess.run([self.loss, self.predict_prob], feed_dict=feed_dict)
                label.append(batch_label)
                loss.append(batch_loss * batch_label.shape[0])
                predict.append(batch_predict)

                tbar.update(batch_label.shape[0])

        label = np.concatenate(label, axis=0)
        predict = np.concatenate(predict, axis=0)
        loss = sum(loss) / steps_num
        # predict = (predict > 0.5).astype('int32').reshape((-1,))
        metrics = PRF(label, predict.argmax(axis=-1))
        metrics['loss'] = loss

        MAP, AvgRec, MRR = eval_reranker(eva_ID, label, predict[:, 0])
        metrics['MAP'] = MAP
        metrics['AvgRec'] = AvgRec
        metrics['MRR'] = MRR

        MAP_summ = tf.Summary(value=[tf.Summary.Value(
            tag="{}/MAP".format(eva_type), simple_value=MAP), ])
        AvgRec_summ = tf.Summary(value=[tf.Summary.Value(
            tag="{}/AvgRec".format(eva_type), simple_value=AvgRec), ])
        MRR = tf.Summary(value=[tf.Summary.Value(
            tag="{}/MRR".format(eva_type), simple_value=MRR), ])


        loss_summ = tf.Summary(value=[tf.Summary.Value(
            tag="{}/loss".format(eva_type), simple_value=metrics['loss']), ])
        macro_F_summ = tf.Summary(value=[tf.Summary.Value(
            tag="{}/f1".format(eva_type), simple_value=metrics['macro_prf'][2]), ])
        acc = tf.Summary(value=[tf.Summary.Value(
            tag="{}/acc".format(eva_type), simple_value=metrics['acc']), ])
        return metrics, [loss_summ, macro_F_summ, acc, MAP_summ, AvgRec_summ, MRR]

    def one_train(self, batch_dataset, saver, writer, config, fold_num=None):
        loss_save = 100
        patience = 0
        self.lr = config.lr
        self.dropout = config.dropout

        print('---------------------------------------------')
        print('process train data')
        train_data = [batch for batch in batch_dataset.batch_train_data(fold_num=fold_num)]
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

                        if val_metrics['loss'] < loss_save:
                            loss_save = val_metrics['loss']
                            patience = 0
                        else:
                            patience += 1

                        if patience >= config.patience:
                            self.lr = self.lr / 2.0
                            loss_save = val_metrics['loss']
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
                        filename = os.path.join(path, "epoch{0}_acc{1:.4f}_fscore{2:.4f}.model"
                                                .format(epoch, val_metrics['acc'], val_metrics['each_prf'][2][0]))

                        # print_metrics(metrics, 'train', path, categories_num=self.args.categories_num)
                        print_metrics(val_metrics, 'val', path)
                        saver.save(self.sess, filename)

                    tbar.update(batch_dataset.batch_size)

    def train(self, batch_dataset: BatchDatasets, config):
        with self.sess:
            saver = tf.train.Saver()
            self.sess.run(tf.global_variables_initializer())

            path = os.path.join(config.model_dir, self.__class__.__name__)
            if not os.path.exists(config.model_dir):
                os.mkdir(config.model_dir)

            if config.k_fold > 1:
                for i in range(config.k_fold):
                    if config.load_best_model and os.path.exists(path):
                        print('------------  load model  ------------')
                        print(tf.train.latest_checkpoint(path + '/fold_{}'.format(i)))
                        saver.restore(self.sess, tf.train.latest_checkpoint(path + '/fold_{}'.format(i)))
                    path = os.path.join(path, 'fold_%d' % i)
                    if not os.path.exists(path):
                        os.mkdir(path)
                    writer = tf.summary.FileWriter(path)
                    self.one_train(batch_dataset, saver, writer, config, i)
                    tf.reset_default_graph()
            else:
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









