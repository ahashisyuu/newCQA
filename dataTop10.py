import os
import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from keras.preprocessing.sequence import pad_sequences


class BatchDatasets:
    def __init__(self,  q_max_len=None, c_max_len=None, char_max_len=16, word_type='lemma',
                 need_shuffle=True, use_char_level=True, batch_size=64, k_fold=0, categories_num=2,
                 train_samples: list=None, dev_samples: list=None, test_samples=None, triplets_file=None, args=None):
        self.train_samples = self.processing_sample(train_samples)  # merge
        self.dev_samples = self.processing_sample(dev_samples)
        self.test_samples = self.processing_sample(test_samples)
        self.args = args
        self.q_max_len = q_max_len
        self.c_max_len = c_max_len
        self.char_max_len = char_max_len
        self.word_type = word_type
        self.need_shuffle = need_shuffle
        self.use_char_level = use_char_level
        self.batch_size = batch_size
        self.train_samples_num = len(self.train_samples)
        self.dev_samples_num = len(self.dev_samples)
        self.categories_num = categories_num
        self.train_steps_num = 0
        self.dev_steps_num = 0
        self.test_steps_num = 0
        self.cweight = []

        self.qTEXT_train = []
        self.q_len_train = []
        self.cTEXT_train = []
        self.c_len_train = []
        self.q_char_train = []
        self.c_char_train = []
        self.qCate_train = []
        self.rel_train = []

        self.q_id_dev = []
        self.c_id_dev = []
        self.qTEXT_dev = []
        self.q_len_dev = []
        self.cTEXT_dev = []
        self.c_len_dev = []
        self.q_char_dev = []
        self.c_char_dev = []
        self.qCate_dev = []
        self.rel_dev = []

        self.q_id_test = []
        self.c_id_test = []
        self.qTEXT_test = []
        self.q_len_test = []
        self.cTEXT_test = []
        self.c_len_test = []
        self.q_char_test = []
        self.c_char_test = []
        self.qCate_test = []
        self.rel_test = []

        self.label_name = 'rel_index' if self.categories_num == 3 else 'Rrel_index'

        if self.need_shuffle:
            shuffle_index = [i for i in range(len(self.train_samples))]
            np.random.shuffle(shuffle_index)
            self.train_samples = self.train_samples[shuffle_index]

        _, self.qTEXT_train, self.q_len_train, \
            self.cTEXT_train, self.c_len_train, \
            self.q_char_train, self.c_char_train, \
            self.qCate_train, self.rel_train = self.trans_data(self.train_samples)

        self.c_id_dev, self.qTEXT_dev, self.q_len_dev, \
            self.cTEXT_dev, self.c_len_dev, \
            self.q_char_dev, self.c_char_dev, \
            self.qCate_dev, self.rel_dev = self.trans_data(self.dev_samples)

        if self.test_samples is not None:
            self.c_id_test, self.qTEXT_test, self.q_len_test, \
                self.cTEXT_test, self.c_len_test, \
                self.q_char_test, self.c_char_test, \
                self.qCate_test, self.rel_test = self.trans_data(self.test_samples)

    def trans_data(self, samples):
        c_id = []
        qTEXT = []
        q_len = []
        cTEXT = []
        c_len = []
        q_char = []
        c_char = []
        qCate = []
        rel = []
        for q_id, groupby in samples:
            c_id.append(groupby['c_id'].values.tolist())

            q_len_name = 'qTEXT_pro_len' if 'pro' in self.word_type else 'qTEXT_len'
            c_len_name = 'cTEXT_pro_len' if 'pro' in self.word_type else 'cTEXT_len'

            qTEXT.append(groupby.iloc[0]['qTEXT_{}_index'.format(self.word_type)])
            q_len.append(groupby.iloc[0][q_len_name])
            cTEXT.append(self.pad_sentence(groupby['cTEXT_{}_index'.format(self.word_type)].values.tolist(), self.c_max_len))
            c_len_temp = groupby[c_len_name].values
            c_len_temp[c_len_temp > self.c_max_len] = self.c_max_len

            if c_len_temp.shape[0] != 10:
                print("{0:10}\t{1:10}".format(q_id, c_len_temp.shape[0]))
            c_len.append(c_len_temp)
            q_char.append(groupby.iloc[0]['qTEXT_{}_char_index'.format(self.word_type)])
            c_char.append(self.pad_sentence(groupby['cTEXT_{}_char_index'.format(self.word_type)].values, self.c_max_len))
            qCate.append(groupby.iloc[0]['cate_index'])
            rel.append(groupby[self.label_name].values)
        qTEXT = np.asarray(qTEXT)
        q_len = np.asarray(q_len)
        c_len = np.asarray(c_len)
        qCate = np.asarray(qCate)
        rel = np.asarray(rel)

        return c_id, qTEXT, q_len, cTEXT, c_len, q_char, c_char, qCate, rel

    @staticmethod
    def processing_sample(samples_list):
        if samples_list is None or len(samples_list) == 0:
            return None

        new_samples = pd.concat(samples_list, axis=0, ignore_index=True)

        group_samples = new_samples.groupby('q_id')

        return np.asarray([a for a in group_samples])

    def get_len(self, e, max_len):
        return min(len(max(e, key=len)), max_len)

    @staticmethod
    def pad_sentence(e, maxlen):
        return pad_sequences(e, padding='post', truncating='post', maxlen=maxlen)

    def paddingTop10(self, qtext, q_len, ctext, c_len, q_char, c_char, cate, rel):
        # q_max_len = min(max(q_len), self.q_max_len)
        q_max_len = self.q_max_len
        q_len[q_len > q_max_len] = q_max_len
        # c_max_len = min(max(c_len), self.c_max_len)
        c_max_len = self.c_max_len
        # c_len[c_len > c_max_len] = c_max_len
        cur_max_len = [q_max_len, 10]

        pad_res = [self.pad_sentence(e, maxlen=l) for e, l in zip([qtext, ctext], cur_max_len)]
        c_len = self.pad_sentence(c_len, 10)
        # print(len(c_char))
        # print([a.shape for a in c_char])
        pad_char_res = [self.pad_sentence(e, maxlen=l) for e, l in zip([q_char, c_char], cur_max_len)]
        pad_rel_res = self.pad_sentence(rel, 10)
        return [pad_res[0], q_len, pad_res[1], c_len] + pad_char_res + [cate, pad_rel_res]

    def mini_batch_data(self, qText, q_len, cText, c_len, q_char, c_char, cate, rel, batch_size):
        data_size = len(rel)
        batch_start = 0
        # for batch_start in np.arange(0, data_size, batch_size):
        for step in range(self.args.max_steps):
            batch_end = min(batch_start + batch_size, data_size)
            sl = slice(batch_start, batch_end)
            batch_qText = qText[sl]
            batch_q_len = q_len[sl]
            batch_cText = cText[sl]
            batch_c_len = c_len[sl]
            batch_q_char = q_char[sl]
            batch_c_char = c_char[sl]
            batch_cate = cate[sl]
            batch_rel = rel[sl]

            batch_start = batch_end % data_size

            yield self.paddingTop10(batch_qText, batch_q_len,
                                    batch_cText, batch_c_len,
                                    batch_q_char, batch_c_char,
                                    batch_cate, batch_rel)

    def batch_train_data(self, batch_size=None):

        self.train_steps_num = len(self.rel_train)

        batch_size = self.batch_size if batch_size is None else batch_size
        return self.mini_batch_data(self.qTEXT_train, self.q_len_train, self.cTEXT_train, self.c_len_train,
                                    self.q_char_train, self.c_char_train,
                                    self.qCate_train, self.rel_train, batch_size)

    def batch_dev_data(self, dev_batch_size=None):
        self.dev_steps_num = len(self.rel_dev)

        batch_size = self.batch_size if dev_batch_size is None else dev_batch_size
        return self.mini_batch_data(self.qTEXT_dev, self.q_len_dev, self.cTEXT_dev, self.c_len_dev,
                                    self.q_char_dev, self.c_char_dev,
                                    self.qCate_dev, self.rel_dev, batch_size)

    def batch_test_data(self, test_batch_size=None):
        assert self.test_samples is not None

        if test_batch_size is None:
            test_batch_size = self.batch_size

        self.test_steps_num = len(self.rel_test)
        batch_size = self.batch_size if test_batch_size is None else test_batch_size
        return self.mini_batch_data(self.qTEXT_test, self.q_len_test, self.cTEXT_test, self.c_len_test,
                                    self.q_char_test, self.c_char_test,
                                    self.qCate_test, self.rel_test, batch_size)
