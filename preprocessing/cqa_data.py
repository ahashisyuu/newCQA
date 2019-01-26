import os
import pickle as pkl
import numpy as np
import tensorflow as tf
import pandas as pd
from .tokenization import FullTokenizer, BasicTokenizer, WordpieceTokenizer, \
    load_vocab, convert_to_unicode, whitespace_tokenize, printable_text


def _truncate_seq_pair(tokens_a, tokens_b, q_max_len, c_max_len):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    def truncate(tokens, max_len):
        while True:
            total_length = len(tokens)
            if total_length <= max_len:
                break
            tokens.pop()

    # print('a:', len(tokens_a), '\tb:', len(tokens_b))
    truncate(tokens_a, q_max_len)
    truncate(tokens_b, c_max_len)


class InputExample(object):
    """
       A single training/test example for simple sequence classification.
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def read_examples(data: pd.DataFrame, word_type='lemma'):
    examples = []
    for index, row in data.iterrows():
        guid = row['c_id']
        text_a = row['qTEXT_' + word_type]
        text_b = row['cTEXT_' + word_type]
        label = row['Rrel_index']
        examples.append(InputExample(guid, text_a, text_b, label))
    return examples


class NBTokenizer(BasicTokenizer):
    def _clean_text(self, text):
        text = super(NBTokenizer, self)._clean_text(text)
        return text.split()

    def tokenize(self, text):
        text = [convert_to_unicode(a) for a in text]
        text2 = []
        for token in text:
            text2 += self._clean_text(token)
        split_tokens = []
        for token in text2:
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.append(token)
        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens


class Tokenizer(FullTokenizer):
    def __init__(self, vocab_file, do_lower_case=True):
        super().__init__(vocab_file, do_lower_case)
        # self.vocab = load_vocab(vocab_file)
        self.basic_tokenizer = NBTokenizer(do_lower_case=do_lower_case)
        # self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab['[UNK]'])
        return ids


def convert_to_features(examples, q_max_len, c_max_len, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = tokenizer.tokenize(example.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, q_max_len, c_max_len)

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            token = '[SEP]' if token == '//' else token
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        max_len = q_max_len + c_max_len + 3
        while len(input_ids) < max_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_len
        assert len(input_mask) == max_len
        assert len(segment_ids) == max_len

        label_id = example.label
        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("guid: %s" % example.guid)
            tf.logging.info("tokens: %s" % " ".join(
                [printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))
    return features


def main():
    bert_dir = '/home/ahashi_syuu/Documents/BERT/uncased_L-12_H-768_A-12/uncased_L-12_H-768_A-12/'
    vocab_file = bert_dir + 'vocab.txt'
    key_list = ['15dev.xml', '15test.xml', '15train.xml',
                '16dev.xml', '16test.xml', '16train1.xml',
                '16train2.xml', '17test.xml']

    print('reading dataset')
    data_dir = '/home/ahashi_syuu/PycharmProjects/newCQA/data'
    data_file = os.path.join(data_dir, 'dataset.pkl')
    with open(data_file, 'rb') as fr:
        data_set = pkl.load(fr)

    word_type = 'lemma'
    q_max_len = 110
    c_max_len = 150

    train_list = key_list[0:4] + key_list[5:7]
    dev_list = key_list[4:5]
    test_list = []

    print('processing: reading')
    train_examples = []
    for name in train_list:
        train_examples += read_examples(data_set[name], word_type)

    dev_examples = []
    dev_cid = []
    for name in dev_list:
        dev_cid += data_set[name]['c_id'].values.tolist()
        dev_examples += read_examples(data_set[name], word_type)

    tokenizer = Tokenizer(vocab_file)
    print('processing: train to features')
    train_features = convert_to_features(train_examples, q_max_len, c_max_len, tokenizer)
    print('processing: dev to features')
    dev_features = convert_to_features(dev_examples, q_max_len, c_max_len, tokenizer)

    with open('cqa_data.pkl', 'wb') as fw:
        pkl.dump([train_features, dev_cid, dev_features], fw)


if __name__ == '__main__':
    main()

