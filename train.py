from dataloader.Dataloader import load_dataset
from conf.Config import data_path_conf, arg_conf
import torch
from vocab.Vocab import create_vocab
from classifier import Classifier
from modules.encoder import Encoder
import numpy as np

if __name__ == '__main__':
    # 固定随机数，使模型可重复性（结果确定）
    np.random.seed(3347)
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    # torch.backends.cudnn.deterministic = True

    # 数据集（训练集-学习、开发集-调参、测试集-评估）
    opts = data_path_conf('./conf/data_path.json')
    train_data = load_dataset(opts['data']['train_path'])
    dev_data = load_dataset(opts['data']['dev_path'])
    test_data = load_dataset(opts['data']['test_path'])
    # 参数（模型参数+数据参数）
    args = arg_conf()
    print('GPU available:', torch.cuda.is_available())
    print('CuDNN available', torch.backends.cudnn.enabled)
    print('GPU number: ', torch.cuda.device_count())

    if torch.cuda.is_available() and args.cuda >= 0:
        args.device = torch.device('cuda', args.cuda)
    else:
        args.device = torch.device('cpu')

    # 词表
    wdvocab = create_vocab(opts['data']['train_path'])
    embedding_weights = wdvocab.get_embedding_weights(opts['data']['embedding_weights'])
    # wdvocab.save(opts['vocab']['save_vocab'])

    # 模型
    args.label_size = wdvocab.label_size
    args.pad = wdvocab.PAD
    # Transformer Encoder文本分类模型
    trans_encoder = Encoder(args, embedding_weights).to(args.device)
    classifier = Classifier(trans_encoder, args, wdvocab)
    classifier.summary()

    # 训练
    classifier.train(train_data, dev_data)

    # 评估
    classifier.evaluate(test_data)
