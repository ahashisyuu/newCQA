import argparse
import json
import pickle as pkl
import tensorflow as tf

from config import Config
from data import BatchDatasets
from models import model_dict


config = Config()
parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--model', type=str, default='BaselineCell3')
parser.add_argument('--train_list', type=list, default=['15train', '15dev', '15test', '16train1', '16train2', '16dev'])
parser.add_argument('--dev_list', type=list, default=['16test'])
parser.add_argument('--test_list', type=list, default=['16test'])

parser.add_argument('--lr', type=float, default=config.lr)
parser.add_argument('--dropout', type=float, default=config.dropout)
parser.add_argument('--q_max_len', type=int, default=config.q_max_len)
parser.add_argument('--c_max_len', type=int, default=config.c_max_len)
parser.add_argument('--char_max_len', type=int, default=config.char_max_len)
parser.add_argument('--epochs', type=int, default=config.epochs)
parser.add_argument('--max_steps', type=int, default=config.max_steps)
parser.add_argument('--batch_size', type=int, default=config.batch_size)
parser.add_argument('--char_dim', type=int, default=config.char_dim)
parser.add_argument('--bert_dim', type=int, default=config.bert_dim)
parser.add_argument('--l2_weight', type=int, default=config.l2_weight)
parser.add_argument('--margin', type=float, default=config.margin)

parser.add_argument('--patience', type=int, default=config.patience)
parser.add_argument('--k_fold', type=int, default=config.k_fold)
parser.add_argument('--categories_num', type=int, default=config.categories_num)
parser.add_argument('--period', type=int, default=config.period)

parser.add_argument('--wipe_num', type=int, default=config.wipe_num)

parser.add_argument('--word_trainable', type=bool, default=config.word_trainable)
parser.add_argument('--char_trainable', type=bool, default=config.char_trainable)
parser.add_argument('--cate_trainable', type=bool, default=config.cate_trainable)
parser.add_argument('--use_highway', type=bool, default=config.use_highway)
parser.add_argument('--use_bert', type=bool, default=config.use_bert)
parser.add_argument('--merge_type', type=str, default=config.merge_type)
parser.add_argument('--need_shuffle', type=bool, default=config.need_shuffle)
parser.add_argument('--use_word_level', type=bool, default=config.use_word_level)
parser.add_argument('--use_char_level', type=bool, default=config.use_char_level)
parser.add_argument('--load_best_model', type=bool, default=config.load_best_model)

parser.add_argument('--model_dir', type=str, default=config.model_dir)
parser.add_argument('--log_dir', type=str, default=config.log_dir)
parser.add_argument('--word_type', type=str, default=config.word_type)


def run(args):
    map_save = 0
    map_res = 0.1
    while map_save < map_res:
        map_save = map_res
        # loading preprocessed data
        with open('./data/dataset.pkl', 'rb') as fr, \
             open('./assist/embedding_matrix_{}.pkl'.format(args.word_type), 'rb') as fr_embed, \
             open('./assist/char2index.json', 'r') as fr_char:
            data = pkl.load(fr)
            embedding_matrix = pkl.load(fr_embed)
            char2index = json.load(fr_char)

        train_samples = [data[k + '.xml'] for k in args.train_list]
        dev_samples = [data[k + '.xml'] for k in args.dev_list]
        # test_samples = [data[k + '.xml'] for k in args.test_list]
        test_samples = None

        train_bert_features = None
        dev_bert_features = None
        test_bert_features = None
        if args.use_bert:
            train_bert_filename_list = ['./data/bert_data_{0}_{1}.xml.pkl'.format(args.word_type, k)
                                        for k in args.train_list]
            dev_bert_filename_list = ['./data/bert_data_{0}_{1}.xml.pkl'.format(args.word_type, k)
                                      for k in args.dev_list]
            test_bert_filename_list = ['./data/bert_data_{0}_{1}.xml.pkl'.format(args.word_type, k)
                                       for k in args.test_list]

            def open_bert(filename):
                with open(filename, 'rb') as _fr:
                    features = pkl.load(_fr)
                return features

            train_bert_features = [open_bert(f) for f in train_bert_filename_list]
            dev_bert_features = [open_bert(f) for f in dev_bert_filename_list]
            # test_bert_features = [open_bert(f) for f in test_bert_filename_list]

        all_data = BatchDatasets(args.q_max_len, args.c_max_len, args.char_max_len, args.word_type,
                                 need_shuffle=args.need_shuffle, use_char_level=args.use_char_level,
                                 batch_size=args.batch_size, k_fold=args.k_fold, categories_num=args.categories_num,
                                 train_samples=train_samples, dev_samples=dev_samples, test_samples=test_samples,
                                 # train_bert_features=train_bert_features,
                                 # dev_bert_features=dev_bert_features,
                                 # test_bert_features=test_bert_features,
                                 triplets_file='./data/data_triplets.pkl', args=args)

        model = model_dict[args.model](embedding_matrix=embedding_matrix, args=args, char_num=len(char2index))

        if args.mode == 'train':
            map_res = model.train(all_data, args)
            tf.reset_default_graph()
            print('=============================================')
            print('\tmap save: ', map_save)
            print('\tmap res: ', map_res)
            print('=============================================')
        elif args.mode == 'test':
            model.test(all_data, args)

        del model


if __name__ == '__main__':
    run(parser.parse_args())

