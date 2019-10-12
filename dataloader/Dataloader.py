import torch
import numpy as np


class Instance(object):
    def __init__(self, words, label):
        self.words = words     # 词序列
        self.label = label     # 标签值


def load_dataset(path):
    insts = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            lbl, raw_data = line.strip().split('|||')
            words = raw_data.split()
            insts.append(Instance(words, lbl))
    np.random.shuffle(insts)
    return insts


# 封装Batch
class Batch(object):
    def __init__(self, src, target):
        self.src = src          # 数据
        self.target = target    # 标签


def batch_iter(dataset, batch_size, wd_vocab, shuffle=True, device=torch.device('cpu')):
    if shuffle:
        np.random.shuffle(dataset)

    # batch数量
    nb_batch = int(np.ceil(len(dataset) / batch_size))

    for i in range(nb_batch):
        batch_data = dataset[i*batch_size: (i+1)*batch_size]
        if shuffle:
            np.random.shuffle(batch_data)

        yield batch_gen(batch_data, wd_vocab, device)


# batch数据变量化，提取数据和标签
def batch_gen(batch_data, wd_vocab, device=torch.device('cpu')):
    batch_size = len(batch_data)
    # 求最长的序列长度
    max_seq_len = max([len(inst.words) for inst in batch_data])

    wd_idx = torch.zeros((batch_size, max_seq_len), dtype=torch.long).to(device)
    lbl_idx = torch.zeros(batch_size, dtype=torch.long).to(device)
    for i, inst in enumerate(batch_data):
        seq_len = len(inst.words)
        wd_idx[i, :seq_len] = torch.tensor(wd_vocab.extwd2idx(inst.words))
        lbl_idx[i] = torch.tensor(wd_vocab.label2index(inst.label))

    return Batch(wd_idx, lbl_idx)

# class BatchIter(object):
#     def __init__(self, dataset):
#         self._dataset = dataset
#         self.x = None
#         self.y = None
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         pass



