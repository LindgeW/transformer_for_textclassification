import json
import argparse


def data_path_conf(path):
    with open(path, 'r', encoding='utf-8') as fin:
        opts = json.load(fin)
    return opts


def arg_conf():
    parser = argparse.ArgumentParser("Text Classification")
    # 通用参数
    parser.add_argument("--cuda", type=int, default=-1, help="which device, default cpu")
    parser.add_argument("--patience", type=int, default=5, help='early-stopping patient iters')
    # 数据参数
    parser.add_argument("--epoch", type=int, default=20, help="Iter number of all data")
    parser.add_argument("--batch_size", type=int, default=32, help="batch data size")

    # 优化器参数参数
    parser.add_argument("--lr", type=float, default=3e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-7, help="weight decay when update")

    # 模型参数
    parser.add_argument("--d_model", type=int, default=300, help='sub-layer feature size')
    parser.add_argument("--d_k", type=int, default=50, help='Query or Key feature size')
    parser.add_argument("--d_v", type=int, default=50, help='Value feature size')
    parser.add_argument("--d_ff", type=int, default=1024, help='pwffn inner-layer feature size')
    parser.add_argument("--nb_heads", type=int, default=6, help='sub-layer feature size')
    parser.add_argument("--encoder_layer", type=int, default=1, help='the number of encoder layer')
    parser.add_argument("--use_sin_pos", type=bool, default=True, help='use sine & cosine position embedding')
    parser.add_argument("--max_pos_embeddings", type=int, default=200, help='max position embeddings')

    parser.add_argument("--dropout", type=float, default=0.1, help="drop out value")

    args = parser.parse_args()

    # 打印出对象的属性和方法
    print(vars(args))
    return args
