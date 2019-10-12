import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        # self._dropout = dropout
        self._dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, pad_mask=None):
        '''
        :param q: [bz, len_q, Q]
        :param k: [bz, len_k, K]
        :param v: [bz, len_v, V]
        :param pad_mask: [bz, len_q, len_k]  填充部分的mask
        more: Q==K, len_k==len_v
        :return: [bz, len_q, V]
        '''
        # [bz, len_q, Q] * [bz, K, len_k] -> [bz, len_q, len_k]
        att_weights = torch.bmm(q, k.transpose(1, 2))
        att_weights /= math.sqrt(k.size(-1))

        if pad_mask is not None:
            att_weights.masked_fill_(pad_mask, float('-inf'))

        # [bz, len_q, len_k]
        soft_att_weights = F.softmax(att_weights, dim=2)

        if self.training:
            soft_att_weights = self._dropout(soft_att_weights)
        # soft_att_weights = F.dropout(soft_att_weights, p=self._dropout, training=self.training)

        # [bz, len_q, len_k] * [bz, len_v, V] -> [bz, len_q, V]
        att_out = torch.bmm(soft_att_weights, v)

        return att_out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, nb_heads, dropout):
        super(MultiHeadAttention, self).__init__()

        self._d_model = d_model
        self._d_k = d_k
        self._d_v = d_v
        self._nb_heads = nb_heads

        self._linear_qs = nn.ModuleList([
            nn.Linear(in_features=d_model, out_features=d_k, bias=False)
            for _ in range(nb_heads)
        ])

        self._linear_ks = nn.ModuleList([
            nn.Linear(in_features=d_model, out_features=d_k, bias=False)
            for _ in range(nb_heads)
        ])

        self._linear_vs = nn.ModuleList([
            nn.Linear(in_features=d_model, out_features=d_v, bias=False)
            for _ in range(nb_heads)
        ])

        self._linear_out = nn.Linear(in_features=nb_heads * d_v,
                                     out_features=d_model)

        self._self_attention = SelfAttention(dropout)

        self._layer_norm = nn.LayerNorm(d_model)

        self._dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, pad_mask=None):
        '''
        :param q: [bz, len_q, d_model]
        :param k: [bz, len_k, d_model]
        :param v: [bz, len_v, d_model]
        :param pad_mask: [bz, len_k]
        more: Q == K, len_k==len_v
        :return: [bz, len_q, d_model]
        '''
        residual = q

        heads = []
        # [bz, len_k] -> [bz, len_q, len_k]
        pad_mask = pad_mask.unsqueeze(1).expand((-1, q.size(1), -1))
        for linear_q, linear_k, linear_v in zip(self._linear_qs, self._linear_ks, self._linear_vs):
            # [bz, len_q, d_k]
            q_fc = linear_q(self._layer_norm(q))
            # [bz, len_k, d_k]
            k_fc = linear_k(self._layer_norm(k))
            # [bz, len_v, d_v]
            v_fc = linear_v(self._layer_norm(v))
            # [bz, len_q, d_v]
            head = self._self_attention(q_fc, k_fc, v_fc, pad_mask)
            heads.append(head)

        # [bz, len_q, nb_heads*d_v] -> [bz, len_q, d_model]
        multi_head = self._linear_out(torch.cat(tuple(heads), dim=-1))

        if self.training:
            multi_head = self._dropout(multi_head)

        # multi_head = F.dropout(multi_head, p=self._dropout, training=self.training)

        # return self._layer_norm(residual + multi_head)
        return residual + multi_head


# class MultiHeadAttention(nn.Module):
#     def __init__(self, d_model, d_k, d_v, nb_heads, dropout):
#         super(MultiHeadAttention, self).__init__()
#
#         self._d_model = d_model
#         self._d_k = d_k
#         self._d_v = d_v
#         self._nb_heads = nb_heads
#
#         self._linear_qs = nn.Linear(in_features=d_model, out_features=d_k * nb_heads)
#
#         self._linear_ks = nn.Linear(in_features=d_model, out_features=d_k * nb_heads)
#
#         self._linear_vs = nn.Linear(in_features=d_model, out_features=d_v * nb_heads)
#
#         self._linear_out = nn.Linear(in_features=nb_heads * d_v, out_features=d_model)
#
#         self._self_attention = SelfAttention(dropout)
#
#         self._layer_norm = nn.LayerNorm(d_model)
#
#         self._dropout = nn.Dropout(dropout)
#
#     def forward(self, q, k, v, pad_mask=None):
#         '''
#         :param q: [bz, len_q, d_model]
#         :param k: [bz, len_k, d_model]
#         :param v: [bz, len_v, d_model]
#         :param pad_mask: [bz, len_k]  填充部分的mask
#         more: Q == K, len_k==len_v
#         :return: [bz, len_q, d_model]
#         '''
#         residual = q
#
#         bz, len_q, _ = q.size()
#         bz, len_k, _ = k.size()
#         bz, len_v, _ = v.size()
#         # [bz, len_q, d_k * nb_heads]
#         q_fc = self._linear_qs(self._layer_norm(q))
#         # [bz, len_k, d_k * nb_heads]
#         k_fc = self._linear_ks(self._layer_norm(k))
#         # [bz, len_v, d_v * nb_heads]
#         v_fc = self._linear_vs(self._layer_norm(v))
#
#         # [bz, len_q, d_k, nb_heads] -> [bz, nb_heads, len_q, d_k] -> [bz*nb_heads, len_q, d_k]
#         q_in = q_fc.reshape(bz, len_q, self._d_k, -1).permute(0, 3, 1, 2).reshape(-1, len_q, self._d_k)
#         # [bz, len_k, d_k, nb_heads] -> [bz, nb_heads, len_k, d_k] -> [bz*nb_heads, len_k, d_k]
#         k_in = k_fc.reshape(bz, len_k, self._d_k, -1).permute(0, 3, 1, 2).reshape(-1, len_k, self._d_k)
#         # [bz, len_v, d_v, nb_heads] -> [bz, nb_heads, len_v, d_v] -> [bz*nb_heads, len_v, d_v]
#         v_in = v_fc.reshape(bz, len_v, self._d_v, -1).permute(0, 3, 1, 2).reshape(-1, len_v, self._d_v)
#
#         # [bz, len_k] -> [bz, 1, len_k] -> [bz*nb_heads, len_q, len_k]
#         pad_mask = pad_mask.unsqueeze(1).repeat(self._nb_heads, len_q, 1)
#         # [bz*nb_heads, len_q, d_v] -> [bz, nb_heads, len_q, d_v] -> [bz, len_q, nb_heads * d_v]
#         att_out = self._self_attention(q_in, k_in, v_in, pad_mask).reshape(bz, self._nb_heads, len_q, self._d_v).permute(0, 2, 1, 3).reshape(bz, len_q, -1)
#
#         # [bz, len_q, nb_heads*d_v] -> [bz, len_q, d_model]
#         multi_head = self._linear_out(att_out)
#
#         if self.training:
#             multi_head = self._dropout(multi_head)
#
#         # multi_head = F.dropout(multi_head, p=self._dropout, training=self.training)
#
#         # return self._layer_norm(residual + multi_head)
#         return residual + multi_head
