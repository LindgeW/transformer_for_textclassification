import torch.nn as nn
import torch
import numpy as np


# 绝对位置嵌入
# PE(pos, 2i) = sin(pos/10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
class PositionEmbedding(nn.Module):
    def __init__(self, d_model):
        super(PositionEmbedding, self).__init__()
        # 注册不应被视为模型参数的缓冲区，而是持久状态的一部分，缓冲区可以使用给定的名称作为属性访问
        # self.register_buffer('pos_emb', 1. / torch.pow(10000, torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer('pos_emb', 1. / torch.pow(10000, 2.0 * torch.arange(0.0, d_model//2) / d_model))

    def forward(self, pos_seqs):
        '''
        :param pos_seqs: [seq_len, ] or [batch_size, seq_len]
        :return: [seq_len, embed_dim] or [batch_size, seq_len, embed_dim]
        '''
        pos_embed = []
        if pos_seqs.dim() == 1:
            outer_product = torch.ger(pos_seqs.float(), self.pos_emb)
            return torch.cat((torch.sin(outer_product), torch.cos(outer_product)), dim=-1)
        else:
            for ps in pos_seqs:
                # 向量外积
                # [seq_len, 1] * [1, embed_dim] -> [seq_len, embed_dim]
                outer_product = torch.ger(ps.float(), self.pos_emb)
                pe = torch.cat((torch.sin(outer_product), torch.cos(outer_product)), dim=-1)
                pos_embed.append(pe)

            return torch.stack(tuple(pos_embed), dim=0)


# 绝对位置嵌入
# PE(pos, 2i) = sin(pos/10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos/10000^(2i/d_model))
def PositionEmbed(max_len, d_model, pad_idx=None):
    pe = np.asarray([[pos / np.power(10000, 2*(i//2) / d_model) for i in range(d_model)]
                     for pos in range(max_len)], dtype=np.float32)
    pe[:, 0::2] = np.sin(pe[:, 0::2])  # start : end : step
    pe[:, 1::2] = np.cos(pe[:, 1::2])

    if pad_idx is not None:
        pe[pad_idx] = 0

    return pe


if __name__ == '__main__':
    # x = torch.tensor([[0, 1, 2, -1], [0, 0, 2, 3]])
    x = torch.arange(4)
    print(x)
    pe = PositionEmbedding(10)
    print(pe(x))

    y = PositionEmbed(4, 10)
    print(y)

    embed = nn.Embedding.from_pretrained(torch.from_numpy(PositionEmbed(4, 10, pad_idx=0)))
    print(embed(torch.tensor([0, 1, 2, 3])))