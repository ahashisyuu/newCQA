import os

import pandas as pd
import pickle as pkl
from bert_serving.client import BertClient

bc_q = BertClient(port=5555, port_out=5556)
bc_c = BertClient(port=5557, port_out=5558)


def main(data_dir, word_type='lemma'):
    # reading data
    key_list = ['15dev.xml', '15test.xml', '15train.xml',
                '16dev.xml', '16test.xml', '16train1.xml',
                '16train2.xml', '17test.xml']

    data_file = os.path.join(data_dir, 'dataset.pkl')
    with open(data_file, 'rb') as fr:
        data = pkl.load(fr)

    bert_data = {}
    for key in key_list:
        bert_data[key] = data[key][['q_id', 'c_id']]

        temp_data = data[key][['qTEXT_' + word_type, 'cTEXT_' + word_type,
                               'qTEXT_len', 'cTEXT_len']]
        temp_data['all_len'] = temp_data['qTEXT_len'] + temp_data['cTEXT_len']

        bert_data[key]['q_bert'] = data[key]['qTEXT_' + word_type].apply()


if __name__ == '__main__':
    main('../data', word_type="lemma_pro")


