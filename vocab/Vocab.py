from collections import Counter
from dataloader.Dataloader import load_dataset
import numpy as np
import pickle


# 创建词表
def create_vocab(path):
    wd_counter, lbl_counter = Counter(), Counter()
    insts = load_dataset(path)
    for inst in insts:
        wd_counter.update(inst.words)
        lbl_counter[inst.label] += 1
    return WordVocab(wd_counter, lbl_counter)


# 词表
class WordVocab(object):
    def __init__(self, wd_counter, lbl_counter):
        self._min_count = 5
        self.PAD = 0  # 填充词索引
        self.UNK = 1  # OOV词索引

        # 统计词频，过滤低频词
        self._wd2freq = {wd: count for wd, count in wd_counter.items() if count > self._min_count}

        # 词索引表
        self._wd2idx = {wd: idx+2 for idx, wd in enumerate(self._wd2freq.keys())}
        self._wd2idx['<pad>'] = self.PAD
        self._wd2idx['<unk>'] = self.UNK

        # 索引词表
        self._idx2wd = {idx: wd for wd, idx in self._wd2idx.items()}

        self._extwd2idx = {}
        self._extidx2wd = {}

        # 标签->索引
        self._lbl2idx = {lbl: idx for idx, lbl in enumerate(lbl_counter.keys())}
        # 索引->标签
        self._idx2lbl = {idx: lbl for lbl, idx in self._lbl2idx.items()}

    # 获取预训练的词向量权重
    def get_embedding_weights(self, path):
        # 词向量表
        vec_tabs = {}
        vec_size = 0
        with open(path, 'r', encoding='utf-8') as fin:
            for line in fin:
                tokens = line.strip().split(' ')
                wd, vecs = tokens[0], tokens[1:]
                if vec_size == 0:
                    vec_size = len(vecs)
                vec_tabs[wd] = np.asarray(vecs, dtype=np.float32)

        self._extwd2idx = {wd: idx+2 for idx, wd in enumerate(vec_tabs.keys())}
        self._extwd2idx['<pad>'] = self.PAD
        self._extwd2idx['<unk>'] = self.UNK
        self._extidx2wd = {idx: wd for wd, idx in self._extwd2idx.items()}

        # oov ratio
        cout = 0
        for words in self._wd2idx.keys():
            if words not in self._extwd2idx.keys():
                cout += 1
        rate = cout / (len(self._wd2idx)-2)
        print('oov ratio:', rate)

        vocab_size = len(self._extwd2idx)
        embedding_weights = np.zeros((vocab_size, vec_size), dtype=np.float32)
        for i, wd in self._extidx2wd.items():
            if i != self.UNK and i != self.PAD:
                embedding_weights[i] = vec_tabs[wd]

        # embedding_weights[self.UNK] = np.random.uniform(-0.25, 0.25, vec_size)
        embedding_weights[self.UNK] = np.mean(embedding_weights, 0) / np.std(embedding_weights)
        return embedding_weights

    def save(self, path):
        with open(path, 'wb') as fw:
            pickle.dump(self, fw)

    def word2index(self, wds):
        if isinstance(wds, list):
            return [self._wd2idx.get(wd, self.UNK) for wd in wds]
        else:
            return self._wd2idx.get(wds, self.UNK)

    def index2word(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2wd.get(i) for i in idxs]
        else:
            return self._idx2wd.get(idxs)

    def extwd2idx(self, wds):
        if isinstance(wds, list):
            return [self._extwd2idx.get(wd, self.UNK) for wd in wds]
        else:
            return self._extwd2idx.get(wds, self.UNK)

    def extidx2wd(self, ids):
        if isinstance(ids, list):
            return [self._extidx2wd.get(i, '<unk>') for i in ids]
        else:
            return self._extidx2wd.get(ids, '<unk>')

    def label2index(self, lbls):
        if isinstance(lbls, list):
            return [self._lbl2idx.get(lbl, -1) for lbl in lbls]
        else:
            return self._lbl2idx.get(lbls, -1)

    def index2label(self, idxs):
        if isinstance(idxs, list):
            return [self._idx2lbl.get(i) for i in idxs]
        else:
            return self._idx2lbl.get(idxs)

    # 词表大小
    @property  # 方法(只读)属性化
    def vocab_size(self):
        return len(self._wd2idx)

    @property
    def ext_vocab_size(self):
        return len(self._extwd2idx)

    # 类别数
    @property
    def label_size(self):
        return len(self._lbl2idx)
