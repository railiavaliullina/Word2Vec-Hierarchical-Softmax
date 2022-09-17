import torch
import numpy as np
from collections import Counter
import random
from copy import deepcopy
import pickle

from datasets.HuffmanTree import VocabWord
from evaluation.accuracy import Evaluation
from models.Word2Vec import get_model
from configs.dataset_config import cfg as dataset_cfg
from configs.train_config import cfg as train_cfg

import pyximport
pyximport.install(setup_args={"include-dirs": np.get_include()})
from pure_cython import train_sentence


class Trainer(torch.utils.data.Dataset):
    def __init__(self, cfg):
        """
        Class for reading and preprocessing data, training.
        :param cfg: dataset config
        """
        self.cfg = cfg

        self.read_data()
        self.get_vocab()
        self.get_huffman_tree(load=self.cfg.load_saved_data)
        self.run_subsampling(load=self.cfg.load_saved_data)
        self.training(load=self.cfg.load_saved_data)

    def read_data(self):
        """
        Reads source data.
        """
        print('Reading data for train...')
        with open(self.cfg.ds_path, 'r') as f:
            self.text = f.read().split()
        self.words_num = len(self.text)

        counts = dict(Counter(self.text))
        norm_f = np.sum(np.asarray(list(counts.values())))
        self.counts_dict_ = {w: counts[w] / float(norm_f) for w in counts}

    def get_vocab(self):
        counter = dict(Counter(self.text))
        self.vocabulary = {w: freq for w, freq in counter.items() if freq >= 5}

        self.word_to_index, self.index_to_word = {}, {}
        for w, word in enumerate(self.vocabulary):
            self.word_to_index[word] = w
            self.index_to_word[w] = word

        norm_f = np.sum(np.asarray(list(self.vocabulary.values())))
        self.counts_dict = {w: self.vocabulary[w] / float(norm_f) for w in self.vocabulary}

    def run_subsampling(self, load=True):
        if load:
            print(f'Loading subsampled words...')
            with open("../data/subsampled_text.pickle", 'rb') as f:
                self.subsampled_text = pickle.load(f)
        else:
            print(f'Subsampling...')
            self.subsampled_text = []

            for w, word in enumerate(self.text):
                keep_prob = random.random()
                word_weight = (np.sqrt(self.counts_dict_[word] * 1e3) + 1) * 1e-3 / float(self.counts_dict_[word])
                if word_weight > keep_prob:
                    self.subsampled_text.append(word)

                if w % 1e6 == 0:
                    print(f'subsampling, word: {w}/{self.words_num}')

            with open("../data/subsampled_text.pickle", 'wb') as f:
                pickle.dump(self.subsampled_text, f)

        self.subsampled_text_size = len(self.subsampled_text)
        self.subsampled_words_dict = dict(Counter(self.subsampled_text))
        self.subsampled_words_counts = list(self.subsampled_words_dict.values())

    def training(self, load=True):

        self.vocab_size = len(self.vocabulary)

        if load:
            print('Loading prepared sentence...')
            with open("../data/sentence.pickle", 'rb') as f:
                sentence = pickle.load(f)
        else:
            print('Preparing sentence...')
            sentence = deepcopy(self.subsampled_text)
            # OOV words -> None
            for w, word in enumerate(sentence):
                index = self.word_to_index.get(word, None)
                if index is None:
                    sentence[w] = None
                else:
                    sentence[w] = self.tree[index]
                    sentence[w].code = np.asarray(self.tree[index].code).astype(np.uint8)
                    sentence[w].nodes = np.asarray(self.tree[index].nodes).astype(np.uint32)
                    assert index == self.tree[index].index
                    sentence[w].index = np.asarray(index).astype(np.uint32)
                    self.tree[index].word = self.tree[index].cn = None

            with open("../data/sentence.pickle", 'wb') as f:
                pickle.dump(sentence, f)

        self.model = get_model(train_cfg.hidden_layer_size, self.vocab_size)

        if train_cfg.evaluate_before_training:
            with open("../data/model_new_1.pickle", 'rb') as f:
                model = pickle.load(f)

            eval = Evaluation(model, self.vocabulary, self.index_to_word, self.word_to_index)
            print('Evaluating with saved model...')
            eval.accuracy(dataset_cfg.ds_path_test)

        alpha = 0.025
        print('Training...')
        for epoch in range(1):
            self.model = train_sentence(self.model, sentence, alpha)

        with open("../data/model.pickle", 'wb') as f:
            pickle.dump(self.model, f)

        print('Evaluating...')
        eval = Evaluation(self.model, self.vocabulary, self.index_to_word, self.word_to_index)
        eval.accuracy(dataset_cfg.ds_path_test)

    def get_huffman_tree(self, max_code_length=1000, load=True):
        if load:
            print(f'Loading Huffman tree...')
            with open("../data/tree.pickle", 'rb') as f:
                self.tree = pickle.load(f)
        else:
            print(f'Building Huffman tree...')
            sorted_vocab = dict(sorted(self.vocabulary.items(), key=lambda item: item[1], reverse=True))

            vocab = []
            vocab_size = len(sorted_vocab)
            count, binary, parent_node = np.empty(vocab_size * 2 + 1, dtype=np.int64), \
                                         np.zeros(vocab_size * 2 + 1, dtype=np.int8), \
                                         np.zeros(vocab_size * 2 + 1, dtype=np.int32)
            code, point = np.empty(vocab_size * 2 + 1, dtype=np.int8), np.empty(vocab_size * 2 + 1, dtype=np.int32)

            for index, (word, cn) in enumerate(sorted_vocab.items()):
                vocab_word = VocabWord(word, cn, index)
                vocab.append(vocab_word)

            for a in range(vocab_size * 2):
                if a < vocab_size:
                    count[a] = vocab[a].cn
                else:
                    count[a] = 1e15

            pos1 = vocab_size - 1
            pos2 = vocab_size

            for a in range(vocab_size - 1):
                if pos1 >= 0:
                    if count[pos1] < count[pos2]:
                        min1i = pos1
                        pos1 -= 1
                    else:
                        min1i = pos2
                        pos2 += 1
                else:
                    min1i = pos2
                    pos2 += 1

                if pos1 >= 0:
                    if count[pos1] < count[pos2]:
                        min2i = pos1
                        pos1 -= 1
                    else:
                        min2i = pos2
                        pos2 += 1
                else:
                    min2i = pos2
                    pos2 += 1

                count[vocab_size + a] = count[min1i] + count[min2i]
                parent_node[min1i] = vocab_size + a
                parent_node[min2i] = vocab_size + a
                binary[min2i] = 1

            for a in range(vocab_size):
                if a % 100 == 0:
                    print(f'{a}/{vocab_size - 1}')
                vocab[a].code = np.ones(max_code_length, dtype=np.int8) * -1
                vocab[a].nodes = np.ones(max_code_length, dtype=np.int32) * -1
                b = a
                i = 0
                while True:
                    code[i] = binary[b]
                    point[i] = b
                    i += 1
                    b = parent_node[b]
                    if b == vocab_size * 2 - 2:
                        break
                vocab[a].nodes[0] = vocab_size - 2

                for b in range(i):
                    vocab[a].code[i - b - 1] = code[b]
                    vocab[a].nodes[i - b] = point[b] - vocab_size

            for a in range(vocab_size):
                if a % 1000 == 0:
                    print(f'Building tree: {a}/{vocab_size - 1}')
                vocab[a].nodes = [p for p in vocab[a].nodes if p > -1]
                vocab[a].code = [c for c in vocab[a].code if c > -1]

            with open("../data/tree.pickle", 'wb') as f:
                pickle.dump(vocab, f)

            self.tree = vocab
