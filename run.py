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
parser.add_argument('--model', type=str, default='Baseline5_finetune_lemma')
parser.add_argument('--train_list', type=list, default=['16train1', '16train2', '15train', '15dev', '15test', '16dev'])
parser.add_argument('--dev_list', type=list, default=['16test'])
parser.add_argument('--test_list', type=list, default=['16test'])

parser.add_argument('--lr', type=float, default=config.lr)
parser.add_argument('--dropout', type=float, default=config.dropout)
parser.add_argument('--q_max_len', type=int, default=config.q_max_len)
parser.add_argument('--c_max_len', type=int, default=config.c_max_len)
parser.add_argument('--char_max_len', type=int, default=config.char_max_len)
parser.add_argument('--epochs', type=int, default=config.epochs)
parser.add_argument('--batch_size', type=int, default=config.batch_size)
parser.add_argument('--char_dim', type=int, default=config.char_dim)
parser.add_argument('--l2_weight', type=int, default=config.l2_weight)

parser.add_argument('--patience', type=int, default=config.patience)
parser.add_argument('--k_fold', type=int, default=config.k_fold)
parser.add_argument('--period', type=int, default=config.period)

parser.add_argument('--wipe_num', type=int, default=config.wipe_num)

parser.add_argument('--word_trainable', type=bool, default=config.word_trainable)
parser.add_argument('--need_shuffle', type=bool, default=config.need_shuffle)
parser.add_argument('--use_char_level', type=bool, default=config.use_char_level)
parser.add_argument('--load_best_model', type=bool, default=config.load_best_model)

parser.add_argument('--model_dir', type=str, default=config.model_dir)
parser.add_argument('--log_dir', type=str, default=config.log_dir)
parser.add_argument('--glove_file', type=str, default=config.glove_filename)


def run(args):
    map_save = 0
    map_res = 0.1
    while map_save < map_res:
        map_save = map_res
        # loading preprocessed data
        with open('./data/dataset.pkl', 'rb') as fr, \
             open('./assist/embedding_matrix_lemma.pkl', 'rb') as fr_embed, \
             open('./assist/char2index.json', 'r') as fr_char:
            data = pkl.load(fr)
            embedding_matrix = pkl.load(fr_embed)
            char2index = json.load(fr_char)

        train_samples = [data[k + '.xml'] for k in args.train_list]
        dev_samples = [data[k + '.xml'] for k in args.dev_list]
        test_samples = [data[k + '.xml'] for k in args.test_list]

        all_data = BatchDatasets(args.q_max_len, args.c_max_len, args.char_max_len,
                                 need_shuffle=args.need_shuffle, use_char_level=args.use_char_level,
                                 batch_size=args.batch_size, k_fold=args.k_fold,
                                 train_samples=train_samples, dev_samples=dev_samples, test_samples=test_samples,
                                 triplets_file='./data/data_triplets.pkl')

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

