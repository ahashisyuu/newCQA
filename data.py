import os
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.sequence import pad_sequences


class BatchDatasets:
    def __init__(self,  q_max_len=None, c_max_len=None, char_max_len=16, word_type='lemma',
                 need_shuffle=True, use_char_level=True, batch_size=64, k_fold=0,
                 train_samples: list=None, dev_samples: list=None, test_samples=None, triplets_file=None):
        self.train_samples = self.processing_sample(train_samples)  # merge
        self.dev_samples = self.processing_sample(dev_samples)
        self.test_samples = self.processing_sample(test_samples)
        self.q_max_len = q_max_len
        self.c_max_len = c_max_len
        self.char_max_len = char_max_len
        self.word_type = word_type
        self.need_shuffle = need_shuffle
        self.use_char_level = use_char_level
        self.batch_size = batch_size
        self.train_samples_num = len(self.train_samples)
        self.dev_samples_num = len(self.dev_samples)
        self.k_fold = k_fold
        self.train_steps_num = 0
        self.dev_steps_num = 0
        self.test_steps_num = 0
        self.cweight = []

        self.qTEXT_train = None
        self.q_len_train = None
        self.cTEXT_train = None
        self.c_len_train = None
        self.q_char_train = None
        self.c_char_train = None
        self.qCate_train = None
        self.rel_train = None

        self.q_id_dev = None
        self.c_id_dev = None
        self.qTEXT_dev = None
        self.q_len_dev = None
        self.cTEXT_dev = None
        self.c_len_dev = None
        self.q_char_dev = None
        self.c_char_dev = None
        self.qCate_dev = None
        self.rel_dev = None

        self.q_id_test = None
        self.c_id_test = None
        self.qTEXT_test = None
        self.q_len_test = None
        self.cTEXT_test = None
        self.c_len_test = None
        self.q_char_test = None
        self.c_char_test = None
        self.qCate_test = None
        self.rel_test = None

        if train_samples is not None and dev_samples is not None:
            if k_fold > 1:  # merge train data and dev data
                self.train_samples = pd.concat([self.train_samples, self.dev_samples], axis=0, ignore_index=True)
                skf = StratifiedKFold(n_splits=k_fold, random_state=0)
                self.index_list = [index
                                   for index in skf.split(self.train_samples['qTEXT_token_index'].values,
                                                          self.train_samples['rel_index'].values)]

        # load triplets training samples
        with open(os.path.join(triplets_file), 'rb') as fr:
            self.data_triplets = pkl.load(fr)

    @staticmethod
    def processing_sample(samples_list):
        if samples_list is None or len(samples_list) == 0:
            return None

        new_samples = pd.concat(samples_list, axis=0, ignore_index=True)

        return new_samples

    def get_len(self, e, max_len):
        return min(len(max(e, key=len)), max_len)

    @staticmethod
    def pad_sentence(e, maxlen):
        return pad_sequences(e, padding='post', truncating='post', maxlen=maxlen)

    def padding(self, qtext, q_len, ctext, c_len, q_char, c_char):
        q_max_len = min(max(q_len), self.q_max_len)
        q_len[q_len > q_max_len] = q_max_len
        c_max_len = min(max(c_len), self.c_max_len)
        c_len[c_len > c_max_len] = c_max_len
        cur_max_len = [q_max_len, c_max_len]
        pad_res = [self.pad_sentence(e, maxlen=l) for e, l in zip([qtext, ctext], cur_max_len)]
        pad_char_res = [self.pad_sentence(e, maxlen=l) for e, l in zip([q_char, c_char], cur_max_len)]
        return [pad_res[0], q_len, pad_res[1], c_len] + pad_char_res

    def mini_batch_data(self, qText, q_len, cText, c_len, q_char, c_char, cate, rel, batch_size):
        data_size = rel.shape[0]
        for batch_start in np.arange(0, data_size, batch_size):
            sl = slice(batch_start, batch_start + batch_size)
            batch_qText = qText[sl]
            batch_q_len = q_len[sl]
            batch_cText = cText[sl]
            batch_c_len = c_len[sl]
            batch_q_char = q_char[sl]
            batch_c_char = c_char[sl]
            batch_cate = cate[sl]
            batch_rel = rel[sl]

            yield self.padding(batch_qText, batch_q_len, batch_cText, batch_c_len, batch_q_char, batch_c_char) + [batch_cate, batch_rel]

    def compute_class_weight(self, train_label):
        label = train_label
        number = [(label == i).astype('int32').sum() for i in range(2)]

        max_num = max(number)
        min_num = min(number)
        median = max_num
        for n in number:
            if n != max_num and n != min_num:
                median = n

        return [median/n for n in number]

    def gen_train_triplets(self, train_files, batch_size):
        samples = [self.data_triplets[key + '.xml'] for key in train_files]
        samples = pd.concat(samples, axis=0, ignore_index=True)
        shuffle_index = [a for a in range(len(samples))]
        np.random.shuffle(shuffle_index)
        samples = samples.iloc[shuffle_index]
        self.train_steps_num = len(samples)
        for batch_start in np.arange(0, self.train_steps_num, batch_size):
            batch_data = samples.iloc[batch_start:batch_start+batch_size]
            q_cate = batch_data['q_cate'].values
            qTEXT_index = batch_data['qTEXT_index'].values
            q_len = batch_data['q_len'].values
            c_pos_index = batch_data['c_pos_index'].values
            c_pos_len = batch_data['c_pos_len'].values
            c_neg_index = batch_data['c_neg_index'].values
            c_neg_len = batch_data['c_neg_len'].values

            q_max_len = min(max(q_len), self.q_max_len)
            c_pos_max_len = min(max(c_pos_len), self.c_max_len)
            c_neg_max_len = min(max(c_neg_len), self.c_max_len)

            q_len[q_len > q_max_len] = q_max_len
            c_pos_len[c_pos_len > c_pos_max_len] = c_pos_max_len
            c_neg_len[c_neg_len > c_neg_max_len] = c_neg_max_len

            yield [self.pad_sentence(qTEXT_index, q_max_len), q_len,
                   self.pad_sentence(c_pos_index, c_pos_max_len), c_pos_len,
                   self.pad_sentence(c_neg_index, c_neg_max_len), c_neg_len, q_cate]

    def batch_train_triplets(self, batch_size=None, train_files=None):
        self.q_id_dev = self.dev_samples['q_id'].values
        self.c_id_dev = self.dev_samples['c_id'].values
        self.qTEXT_dev = self.dev_samples['qTEXT_lemma_index'].values.tolist()
        self.q_len_dev = self.dev_samples['qTEXT_len'].values
        self.cTEXT_dev = self.dev_samples['cTEXT_lemma_index'].values.tolist()
        self.c_len_dev = self.dev_samples['cTEXT_len'].values
        self.qCate_dev = self.dev_samples['cate_index'].values
        self.rel_dev = self.dev_samples['Rrel_index'].values

        batch_size = batch_size or self.batch_size
        return self.gen_train_triplets(train_files, batch_size)

    def batch_train_data(self, batch_size=None, fold_num=None):
        if self.k_fold > 1:
            train_index, dev_index = self.index_list[fold_num]
            q_id = self.train_samples['q_id'].values
            c_id = self.train_samples['c_id'].values
            qTEXT = self.train_samples['qTEXT_{}_index'.format(self.word_type)].values
            q_len = self.train_samples['qTEXT_len'].values
            cTEXT = self.train_samples['cTEXT_{}_index'.format(self.word_type)].values
            c_len = self.train_samples['cTEXT_len'].values
            q_char = self.train_samples['qTEXT_{}_char_index'.format(self.word_type)].values
            c_char = self.train_samples['cTEXT_{}_char_index'.format(self.word_type)].values
            qCate = self.train_samples['cate_index'].values
            Rrel = self.train_samples['Rrel_index'].values

            self.qTEXT_train = qTEXT[train_index].tolist()
            self.q_len_train = q_len[train_index]
            self.cTEXT_train = cTEXT[train_index].tolist()
            self.c_len_train = c_len[train_index]
            self.q_char_train = q_char[train_index]
            self.c_char_train = c_char[train_index]
            self.qCate_train = qCate[train_index]
            self.rel_train = Rrel[train_index]

            self.q_id_dev = q_id[dev_index]
            self.c_id_dev = c_id[dev_index]
            self.qTEXT_dev = qTEXT[dev_index].tolist()
            self.q_len_dev = q_len[dev_index]
            self.cTEXT_dev = cTEXT[dev_index].tolist()
            self.c_len_dev = c_len[dev_index]
            self.q_char_dev = q_char[dev_index]
            self.c_char_dev = c_char[dev_index]
            self.qCate_dev = qCate[dev_index]
            self.rel_dev = Rrel[dev_index]
        else:
            if self.need_shuffle:
                shuffle_index = [i for i in range(len(self.train_samples))]
                np.random.shuffle(shuffle_index)
                self.train_samples = self.train_samples.iloc[shuffle_index]
            self.qTEXT_train = self.train_samples['qTEXT_{}_index'.format(self.word_type)].values.tolist()
            self.q_len_train = self.train_samples['qTEXT_len'].values
            self.cTEXT_train = self.train_samples['cTEXT_{}_index'.format(self.word_type)].values.tolist()
            self.c_len_train = self.train_samples['cTEXT_len'].values
            self.q_char_train = self.train_samples['qTEXT_{}_char_index'.format(self.word_type)].values
            self.c_char_train = self.train_samples['cTEXT_{}_char_index'.format(self.word_type)].values
            self.qCate_train = self.train_samples['cate_index'].values
            self.rel_train = self.train_samples['Rrel_index'].values

            self.q_id_dev = self.dev_samples['q_id'].values
            self.c_id_dev = self.dev_samples['c_id'].values
            self.qTEXT_dev = self.dev_samples['qTEXT_{}_index'.format(self.word_type)].values.tolist()
            self.q_len_dev = self.dev_samples['qTEXT_len'].values
            self.cTEXT_dev = self.dev_samples['cTEXT_{}_index'.format(self.word_type)].values.tolist()
            self.c_len_dev = self.dev_samples['cTEXT_len'].values
            self.q_char_dev = self.dev_samples['qTEXT_{}_char_index'.format(self.word_type)].values
            self.c_char_dev = self.dev_samples['cTEXT_{}_char_index'.format(self.word_type)].values
            self.qCate_dev = self.dev_samples['cate_index'].values
            self.rel_dev = self.dev_samples['Rrel_index'].values

        self.train_steps_num = self.rel_train.shape[0]
        self.cweight = self.compute_class_weight(self.rel_train)

        batch_size = self.batch_size if batch_size is None else batch_size
        return self.mini_batch_data(self.qTEXT_train, self.q_len_train, self.cTEXT_train, self.c_len_train,
                                    self.q_char_train, self.c_char_train,
                                    self.qCate_train, self.rel_train, batch_size)

    def batch_dev_data(self, dev_batch_size=None):
        self.dev_steps_num = self.rel_dev.shape[0]

        batch_size = self.batch_size if dev_batch_size is None else dev_batch_size
        return self.mini_batch_data(self.qTEXT_dev, self.q_len_dev, self.cTEXT_dev, self.c_len_dev,
                                    self.q_char_dev, self.c_char_dev,
                                    self.qCate_dev, self.rel_dev, batch_size)

    def batch_test_data(self, test_batch_size=None):
        assert self.test_samples is not None

        self.q_id_test = self.test_samples['q_id'].values
        self.c_id_test = self.test_samples['c_id'].values
        self.qTEXT_test = self.test_samples['qTEXT_{}_index'.format(self.word_type)].values.tolist()
        self.q_len_test = self.test_samples['qTEXT_len'].values
        self.cTEXT_test = self.test_samples['cTEXT_{}_index'.format(self.word_type)].values.tolist()
        self.c_len_test = self.test_samples['cTEXT_len'].values
        self.q_char_test = self.test_samples['qTEXT_{}_char_index'.format(self.word_type)].values
        self.c_char_test = self.test_samples['cTEXT_{}_char_index'.format(self.word_type)].values
        self.qCate_test = self.test_samples['cate_index'].values
        self.rel_test = self.test_samples['Rrel_index'].values

        if test_batch_size is None:
            test_batch_size = self.batch_size

        self.test_steps_num = self.rel_test.shape[0]
        batch_size = self.batch_size if test_batch_size is None else test_batch_size
        return self.mini_batch_data(self.qTEXT_test, self.q_len_test, self.cTEXT_test, self.c_len_test,
                                    self.q_char_test, self.c_char_test,
                                    self.qCate_test, self.rel_test, batch_size)
