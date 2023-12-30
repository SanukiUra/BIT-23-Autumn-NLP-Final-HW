import pandas as pd
import os
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import torch
from torch.utils.data import TensorDataset
import time
from gensim.models.keyedvectors import KeyedVectors


class Dictionary(object):
    def __init__(self):
        self.word2tkn = {"[PAD]": 0, "[SOS]": 1, "[EOS]": 2, "[UNK]": 3}
        self.tkn2word = ["[PAD]", "[SOS]", "[EOS]", "[UNK]"]

        self.label2idx = {}
        self.idx2label = []

    def add_word(self, word):
        if word not in self.word2tkn:
            self.tkn2word.append(word)
            self.word2tkn[word] = len(self.tkn2word) - 1
        return self.word2tkn[word]


class Corpus(object):

    def __init__(self, path, max_token_per_sent, embedding_dim=300, need_vectorize = True):
        self.dictionary = Dictionary()
        self.embedding_weight = None

        self.max_token_per_sent = max_token_per_sent
        self.embedding_dim = embedding_dim

        print(f"Start building dataset...")
        self.train = self.tokenize(os.path.join(path, 'ROCStories_train.csv'))
        print("1/3...")
        self.test = self.tokenize(os.path.join(path, 'ROCStories_test.csv'))
        print("2/3...")
        self.val = self.tokenize(os.path.join(path, 'ROCStories_val.csv'))
        print(f"Build Successfully.")

        self.vocab_size = len(self.dictionary.tkn2word)
        if need_vectorize:
            self.vectorize()

    def vectorize(self):
        print("loading word2vec ...")
        t1 = time.time()
        word_vector = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec')
        t2 = time.time()
        print(f"loading success, costing {t2 - t1:.2f} seconds.")

        self.embedding_weight = torch.zeros(self.vocab_size, self.embedding_dim)
        vector_count = 0
        mean = word_vector.vectors.mean(axis=0)
        for i in range(self.vocab_size):
            if self.dictionary.tkn2word[i] in word_vector:
                self.embedding_weight[i] = torch.from_numpy(word_vector[self.dictionary.tkn2word[i]])
                vector_count += 1
            else:
                # 若词向量中没有该词，则初始化成所有词向量的均值；PAD补0
                self.embedding_weight[i] = torch.from_numpy(mean)
        # [PAD]的初始化
        self.embedding_weight[0] = torch.zeros(self.embedding_dim)
        # [SOS]的初始化
        self.embedding_weight[1] = torch.from_numpy(mean)
        # [EOS]的初始化
        self.embedding_weight[2] = torch.from_numpy(mean)
        # [UNK]的初始化
        self.embedding_weight[3] = torch.from_numpy(mean)
        print(
            f"embedding_rate: {vector_count}/{self.vocab_size} = {vector_count / self.vocab_size * 100:.2f}%.")
        # os.system("pause")
        return self.embedding_weight

    def pad(self, origin_token_seq, length):
        if len(origin_token_seq) > length:
            attention_mask = [0] * length
            tokens = origin_token_seq[:length]
            return tokens, attention_mask
        else:
            attention_mask = [0] * len(origin_token_seq)
            mask_pad = [1] * (length - len(origin_token_seq))
            attention_mask.extend(mask_pad)
            tokens = origin_token_seq + [0 for _ in range(length - len(origin_token_seq))]
            return tokens, attention_mask

    def tokenize(self, path):
        src_idss = []
        tgt_idss = []
        src_key_padding_mask = []
        tgt_key_padding_mask = []
        # 打开csv文件
        with open(path, 'r', encoding='utf-8') as f:
            # 读取csv文件
            df = pd.read_csv(f)
            for line in df.values:
                sents = line[2:7]

                src = sents[0]
                tgt = sents[1:]
                # 处理src
                src_ids = [1]
                src_words = word_tokenize(src)
                for word in src_words:
                    self.dictionary.add_word(word)
                    src_ids.append(self.dictionary.word2tkn[word])
                src_ids.append(2)
                src_inputs, attention_mask = self.pad(src_ids, self.max_token_per_sent)
                src_idss.append(src_inputs)
                src_key_padding_mask.append(attention_mask)

                # 处理tgt
                total_tgt = []
                for sent in tgt:
                    tgt_ids = [1]
                    tgt_words = word_tokenize(sent)
                    for word in tgt_words:
                        self.dictionary.add_word(word)
                        tgt_ids.append(self.dictionary.word2tkn[word])
                    tgt_ids.append(2)
                    total_tgt.extend(tgt_ids)
                tgt_inputs, attention_mask = self.pad(total_tgt, self.max_token_per_sent)

                tgt_idss.append(tgt_inputs)
                tgt_key_padding_mask.append(attention_mask)

        src_idss = torch.tensor(np.array(src_idss))
        tgt_idss = torch.tensor(np.array(tgt_idss))

        src_key_padding_mask = np.array(src_key_padding_mask, dtype=float)
        tgt_key_padding_mask = np.array(tgt_key_padding_mask, dtype=float)

        src_masks = torch.FloatTensor(src_key_padding_mask)
        tgt_masks = torch.FloatTensor(tgt_key_padding_mask)

        return TensorDataset(src_idss, tgt_idss, src_masks, tgt_masks)