import math
import os
import random

import numpy as np
import tensorflow as tf
import keras.losses
from utils import PRF, print_metrics, eval_reranker
from tqdm import tqdm
from data import BatchDatasets
from layers.BiGRU import Dropout
from layers import optimization

EPSILON = 1e-7


class TextCNN:
    def __init__(self, input_dim, filter_sizes, filter_num):
        self.filter_sizes = filter_sizes
        self.filters = []
        self.bs = []
        for filter_size in filter_sizes:
            with tf.variable_scope("conv{}".format(filter_size)):
                filter_shape = tf.convert_to_tensor([filter_size, input_dim, 1, filter_num])
                fil = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='filter')
                b = tf.Variable(tf.constant(0.1, shape=[filter_num]))

                self.filters.append(fil)
                self.bs.append(b)

    def __call__(self, inputs, mask=None):
        pooled_outputs = []
        inputs = inputs * tf.expand_dims(mask, axis=-1) if mask is not None else inputs
        input_expand = tf.expand_dims(inputs, -1)  # (b,m,d,1)
        for filter_size, fil, b in zip(self.filter_sizes, self.filters, self.bs):
            with tf.variable_scope("conv{}".format(filter_size)):
                conv = tf.nn.conv2d(input_expand, fil, [1]*4, 'VALID', name='conv')
                h = tf.nn.relu(tf.nn.bias_add(conv, b))
                pooled = tf.reduce_max(h, 1, True)
                pooled_outputs.append(tf.squeeze(pooled, axis=[1, 2]))
        return tf.concat(pooled_outputs, 1)


class MultiLayerHighway:
    def __init__(self, units, layers_num,
                 activation=tf.nn.elu, keep_prob=1.0, is_train=True, need_bias=True, scope=None):
        self.units = units
        self.layers_num = layers_num
        self.activation = activation
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.need_bias = need_bias
        self.first_use = True
        self.scope = scope or 'MulHighway'

    def __call__(self, inputs):
        with tf.variable_scope(self.scope):
            output = inputs
            reuse = True if self.first_use is False else None
            for i in range(self.layers_num):
                output_line = Dropout(output, self.keep_prob, self.is_train)
                output_gate = Dropout(output, self.keep_prob, self.is_train)
                output_tran = Dropout(output, self.keep_prob, self.is_train)

                line = tf.layers.dense(output_line, self.units, tf.identity,
                                       self.need_bias, name='line_%d' % i, reuse=reuse)
                gate = tf.layers.dense(output_gate, self.units, tf.sigmoid,
                                       self.need_bias, name='gate_%d' % i, reuse=reuse)
                tran = tf.layers.dense(output_tran, self.units, self.activation,
                                       self.need_bias, name='tran_%d' % i, reuse=reuse)
                output = gate * tran + (1 - gate) * line
            self.first_use = False
            return output


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
        self.is_training = True
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
        self.batch_size = None
        self.Q_maxlen = args.q_max_len
        self.C_maxlen = args.c_max_len
        self.char_max_len = args.char_max_len
        self.char_num = char_num

        # batch input
        self.inputs = self.QText, self.Q_len, self.CText, self.C_len, \
            self.Q_char, self.C_char, self.Q_bert, self.C_bert, \
            self.Qcate, self.Rel = self.create_input()
        self.inputs = [a for a in self.inputs if a is not None]

        # preparing mask and length info
        self.Q_mask = tf.cast(tf.cast(self.QText, tf.bool), tf.float32)
        self.C_mask = tf.cast(tf.cast(self.CText, tf.bool), tf.float32)
        # self.q_features, self.c_features = self.equal_features(self.QText, self.CText)
        # self.Q_maxlen = tf.reduce_max(self.Q_len)
        # self.C_maxlen = tf.reduce_max(self.C_len)

        self.N = tf.shape(self.Q_mask)[0]

        # embedding word vector and char vector
        self.QS, self.CT, self.cate_f = self.embedding()

        # building model
        self.output = self.build_model()

        # computing loss
        with tf.variable_scope('predict'):
            self.predict_prob = tf.nn.softmax(self.output)
            labels = tf.one_hot(self.Rel, self.args.categories_num, dtype=tf.float32)
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=self.output)
            self.loss = tf.reduce_mean(losses)
            if self.args.l2_weight != 0:
                for v in tf.trainable_variables():
                    self.loss += self.args.l2_weight * tf.nn.l2_loss(v)

        # counting parameters
        self.count()

        self.train_op = None
        self.ready_train(self._lr, 1000)

    def create_input(self):
        qTEXT = tf.placeholder(tf.int32, [self.batch_size, None])
        q_len = tf.placeholder(tf.int32, [self.batch_size])
        cTEXT = tf.placeholder(tf.int32, [self.batch_size, None])
        c_len = tf.placeholder(tf.int32, [self.batch_size])
        q_char = tf.placeholder(tf.int32, [self.batch_size, None, self.char_max_len])
        c_char = tf.placeholder(tf.int32, [self.batch_size, None, self.char_max_len])
        inputs = [qTEXT, q_len, cTEXT, c_len, q_char, c_char]

        if self.args.use_bert:
            q_bert = tf.placeholder(tf.float32, [self.batch_size, None, self.args.bert_dim])
            c_bert = tf.placeholder(tf.float32, [self.batch_size, None, self.args.bert_dim])
            inputs += [q_bert, c_bert]
        else:
            inputs += [None] * 2

        cate = tf.placeholder(tf.int32, [self.batch_size])
        rel = tf.placeholder(tf.int32, [self.batch_size])

        return inputs + [cate, rel]

    def embedding(self):
        # word embedding
        with tf.variable_scope('emb'):
            if self.args.use_word_level:
                QS = tf.nn.embedding_lookup(self.word_mat, self.QText)
                CT = tf.nn.embedding_lookup(self.word_mat, self.CText)

                if self.args.use_bert:
                    if self.args.merge_type is 'concat':
                        QS = tf.concat([QS, self.Q_bert], -1)
                        CT = tf.concat([CT, self.C_bert], -1)
                    elif self.args.merge_type is 'plus':
                        QS = QS + self.Q_bert
                        CT = CT + self.C_bert
            else:
                QS = self.Q_bert
                CT = self.C_bert

            if self.args.use_char_level:
                self.char_embedding = tf.get_variable('char_mat', [self.char_num + 1, self.args.char_dim],
                                                      trainable=self.args.char_trainable)
                q_char = tf.reshape(self.Q_char, [-1, tf.shape(self.Q_char)[2]])
                c_char = tf.reshape(self.C_char, [-1, tf.shape(self.C_char)[2]])
                q_cmask = tf.cast(tf.cast(q_char, tf.bool), tf.float32)
                c_cmask = tf.cast(tf.cast(c_char, tf.bool), tf.float32)

                q_char_emb = tf.nn.embedding_lookup(self.char_embedding, q_char)
                c_char_emb = tf.nn.embedding_lookup(self.char_embedding, c_char)

                text_cnn = TextCNN(self.args.char_dim, [1, 2, 3, 4, 5, 6], 50)
                q_char_emb = text_cnn(q_char_emb, q_cmask)
                c_char_emb = text_cnn(c_char_emb, c_cmask)

                q_char_emb = tf.reshape(q_char_emb, [self.N, -1, 300])
                c_char_emb = tf.reshape(c_char_emb, [self.N, -1, 300])

                if self.args.merge_type is 'concat':
                    QS = tf.concat([QS, q_char_emb], -1)
                    CT = tf.concat([CT, c_char_emb], -1)
                elif self.args.merge_type is 'plus':
                    QS = QS + q_char_emb
                    CT = CT + c_char_emb
                else:
                    raise ValueError('merge_type error')

            if self.args.use_highway:
                highway = MultiLayerHighway(300, 1, tf.nn.elu, self.dropout_keep_prob, self._is_train)
                QS = highway(QS)
                CT = highway(CT)

            category_mat = tf.get_variable('cate_mat', shape=[33, 25], dtype=tf.float32,
                                           initializer=tf.glorot_uniform_initializer(),
                                           trainable=self.args.cate_trainable)
            cate_f = tf.nn.embedding_lookup(category_mat, self.Qcate)

            return [QS, CT, cate_f]

    def equal_features(self, text_a, text_b):
        expanded_a = tf.expand_dims(text_a, axis=2)  # (B, L1, 1)
        expanded_b = tf.expand_dims(text_b, axis=1)  # (B, 1, L2)

        shape = [-1, tf.shape(text_a)[1], tf.shape(text_b)[2]]
        text_a = tf.tile(expanded_a, shape)
        text_b = tf.tile(expanded_b, shape)

        truth_matrix = tf.equal(text_a, text_b)
        features_a = tf.cast(tf.reduce_any(truth_matrix, 2), tf.float32)  # (B, L1)
        features_b = tf.cast(tf.reduce_any(truth_matrix, 1), tf.float32)  # (B, L2)

        return features_a, features_b

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
            # processed_samples = 0
            for batch_eva_data in eva_data:
                batch_label = batch_eva_data[-1]
                feed_dict = {inv: array for inv, array in zip(self.inputs, batch_eva_data)}
                batch_loss, batch_predict = self.sess.run([self.loss, self.predict_prob], feed_dict=feed_dict)
                label.append(batch_label)
                loss.append(batch_loss * batch_label.shape[0])
                predict.append(batch_predict)

                tbar.update(batch_label.shape[0])

                # processed_samples += self.batch_size
                # if processed_samples >= steps_num:
                #     break

        label = np.concatenate(label, axis=0)[:steps_num]
        predict = np.concatenate(predict, axis=0)[:steps_num]
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
        MRR_summ = tf.Summary(value=[tf.Summary.Value(
            tag="{}/MRR".format(eva_type), simple_value=MRR), ])

        loss_summ = tf.Summary(value=[tf.Summary.Value(
            tag="{}/loss".format(eva_type), simple_value=metrics['loss']), ])
        macro_F_summ = tf.Summary(value=[tf.Summary.Value(
            tag="{}/f1".format(eva_type), simple_value=metrics['each_prf'][2][0]), ])
        acc = tf.Summary(value=[tf.Summary.Value(
            tag="{}/acc".format(eva_type), simple_value=metrics['acc']), ])
        return metrics, [loss_summ, macro_F_summ, acc, MAP_summ, AvgRec_summ, MRR_summ]

    def ready_train(self, lr, num_train_steps):
        # getting ready for training
        # self.opt = tf.train.AdagradOptimizer(learning_rate=self._lr)
        self.opt = tf.train.AdamOptimizer(learning_rate=self._lr, epsilon=1e-6)
        grads = self.opt.compute_gradients(self.loss)
        gradients, variables = zip(*grads)
        capped_grads, _ = tf.clip_by_global_norm(gradients, clip_norm=5.0)
        self.train_op = self.opt.apply_gradients(zip(capped_grads, variables),
                                                 global_step=self._global_step)

        # self.train_op = optimization.create_optimizer(self.loss, lr, num_train_steps, None, False)

    def one_train(self, batch_dataset, path, config, fold_num=None):
        loss_save = 100
        patience_loss = 0
        map_save = 0
        patience_map = 0
        self.lr = config.lr
        self.dropout = config.dropout

        # train_data = batch_dataset.batch_train_data(fold_num=fold_num)
        # train_steps = batch_dataset.train_steps_num

        # num_trian_steps = math.ceil(train_steps / self.args.batch_size * self.args.epochs)
        # self.ready_train(config.lr, num_trian_steps)

        saver = tf.train.Saver(max_to_keep=10)
        self.sess.run(tf.global_variables_initializer())

        if config.load_best_model and os.path.exists(path):
            print('------------  load model  ------------')
            saver.restore(self.sess, tf.train.latest_checkpoint(path))
        if not os.path.exists(path):
            os.mkdir(path)
        writer = tf.summary.FileWriter(path)

        for epoch in range(self.args.epochs):
            print('---------------------------------------')
            print('EPOCH %d' % epoch)

            print('---------------------------------------------')
            print('process train data')
            train_data = [a for a in batch_dataset.batch_train_data(fold_num=fold_num)]
            train_steps = batch_dataset.train_steps_num
            print('---------------------------------------------')

            print('class weight: ', batch_dataset.cweight)
            # self.cweight = [.9, 5., 1.1]

            print('\n---------------------------------------------')
            print('process dev data')
            dev_data = [a for a in batch_dataset.batch_dev_data()]
            dev_steps = batch_dataset.dev_steps_num
            dev_id = batch_dataset.c_id_dev
            print('----------------------------------------------\n')

            print('the number of samples: %d\n' % train_steps)

            print('training model')
            self.is_train = True
            self.is_training = True
            self.dropout = config.dropout

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
                        self.is_training = False
                        self.dropout = 1.0

                        # print('\n---------------------------------------------')
                        # print('process dev data')
                        # dev_data = batch_dataset.batch_dev_data()
                        # dev_steps = batch_dataset.dev_steps_num
                        # dev_id = batch_dataset.c_id_dev
                        # print('----------------------------------------------\n')

                        val_metrics, summ = self.evaluate(dev_data, dev_steps, 'dev', dev_id)
                        val_metrics['step'] = self.global_step

                        if val_metrics['loss'] < loss_save:
                            loss_save = val_metrics['loss']
                            patience_loss = 0
                        else:
                            patience_loss += 1

                        if patience_loss >= config.patience:
                            self.lr = self.lr / 2.0
                            loss_save = val_metrics['loss']
                            patience_loss = 0

                        if math.fabs(val_metrics['MAP'] - map_save) < 0.0001:
                            patience_map += 1
                        else:
                            patience_map = 0
                            map_save = val_metrics['MAP']
                        if patience_map >= config.patience:
                            return val_metrics['MAP']

                        for s in summ:
                            writer.add_summary(s, self.global_step)
                        writer.flush()

                        path = os.path.join(config.model_dir, self.__class__.__name__)
                        if not os.path.exists(path):
                            os.mkdir(path)
                        # else:
                        #     for i in range(10):
                        #         if not os.path.exists(path + '_{}'.format(i+1)):
                        #             path = path + '_{}'.format(i+1)
                        #             os.mkdir(path)
                        #             break

                        if fold_num is not None:
                            path = os.path.join(path, 'fold_%d' % fold_num)
                            if not os.path.exists(path):
                                os.mkdir(path)
                        filename = os.path.join(path, "step{0}_MAP{1:.4f}_Avg{2:.4f}_MRR{3:.4f}.model"
                                                .format(self.global_step, val_metrics['MAP'], val_metrics['AvgRec'], val_metrics['MRR']))

                        # print_metrics(metrics, 'train', path, categories_num=self.args.categories_num)
                        # print_metrics(val_metrics, 'val', path)
                        saver.save(self.sess, filename)
                        # if self.global_step >= self.args.max_steps:
                        #     break

                    tbar.update(batch_dataset.batch_size)

        return 0

    def train(self, batch_dataset: BatchDatasets, config):
        with self.sess:
            path = os.path.join(config.model_dir, self.__class__.__name__)
            if not os.path.exists(config.model_dir):
                os.mkdir(config.model_dir)

            return self.one_train(batch_dataset, path, config)

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
            # print(shape, variable)
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print('\n\n------------------------------------------------')
        print('total_parameters: ', total_parameters)
        print('------------------------------------------------\n\n')









