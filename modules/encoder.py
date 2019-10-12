import torch
import torch.nn as nn
import torch.nn.functional as F
from .embedding import PositionEmbedding, PositionEmbed
from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward
from .rnn_encoder import RNNEncoder


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, d_ff, nb_heads, dropout):
        super(EncoderLayer, self).__init__()
        # multi_head self-attention
        self._multi_head_att = MultiHeadAttention(d_model=d_model,
                                                  d_k=d_k,
                                                  d_v=d_v,
                                                  nb_heads=nb_heads,
                                                  dropout=dropout)
        # feedforward
        self._pwffn = PositionwiseFeedForward(d_in=d_model,
                                              d_ff=d_ff)

    def forward(self, enc_in, pad_mask=None, non_pad_mask=None):
        '''
        :param enc_in: [bz, len_k, d_model]
        :param pad_mask: [bz, len_k] 填充部分mask
        :param non_pad_mask: [bz, len_q, 1]
        :return: [bz, len_q, d_model]
        '''
        # [bz, len_q, d_model]
        multi_head = self._multi_head_att(enc_in, enc_in, enc_in, pad_mask)
        if non_pad_mask is not None:
            multi_head *= non_pad_mask

        # [bz, len_q, d_model]
        out = self._pwffn(multi_head)
        if non_pad_mask is not None:
            out *= non_pad_mask

        return out


class Encoder(nn.Module):
    def __init__(self, args, embedding_weights=None):
        super(Encoder, self).__init__()

        self._pad = args.pad  # 序列填充值

        # word embedding
        self._wd_embed = nn.Embedding.from_pretrained(torch.from_numpy(embedding_weights))

        # position embedding
        # pe = PositionEmbedding(args.d_model)
        # # self._pos_embed = nn.Embedding.from_pretrained()
        # self._pos_embed = nn.Embedding(num_embeddings=100,
        #                                embedding_dim=args.d_model,
        #                                padding_idx=0)  # pad部分向量为0
        # self._pos_embed.weight.data[1:].copy_(pe(torch.arange(1, 100)))
        # self._pos_embed.weight.requires_grad = False

        self._pos_embed = nn.Embedding.from_pretrained(PositionEmbed(100, args.d_model, pad_idx=0))

        self._encoder_stack = nn.ModuleList([
            EncoderLayer(args.d_model, args.d_k, args.d_v, args.d_ff, args.nb_heads, args.dropout)
            for _ in range(args.encoder_layer)
        ])

        self._bilstm = RNNEncoder(
            input_size=args.d_model,
            hidden_size=args.d_model//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            rnn_type='lstm'
        )

        self._linear = nn.Linear(in_features=args.d_model,
                                 out_features=args.label_size)

        self._drop_embed = nn.Dropout(args.dropout)

    def forward(self, inputs):
        '''
        :param inputs: [bz, seq_len]
        :return:
        '''
        # 填充部分的mask(uint8类型)
        # [bz, seq_len]
        pad_mask = inputs.eq(self._pad)  # 等于
        # [bz, seq_len, 1]
        # non_pad_mask = inputs.ne(self._pad).float().unsqueeze(-1)  # 非

        wd_embed = self._wd_embed(inputs)
        seq_range = torch.arange(inputs.size(1), dtype=torch.long, device=inputs.device).unsqueeze(dim=0)
        pos_embed = self._pos_embed(seq_range)
        # [bz, seq_len, d_model]
        input_embed = wd_embed + pos_embed
        # input_embed = torch.cat((wd_embed, pos_embed), dim=-1)

        if self.training:
            input_embed = self._drop_embed(input_embed)

        encoder_in = input_embed
        for encoder in self._encoder_stack:
            # [bz, len_q, d_model]
            encoder_in = encoder(encoder_in, pad_mask)

        # [bz, len_q, d_model]
        # enc_out, _ = self._bilstm(encoder_in, non_pad_mask)

        # [bz, d_model, len_q]
        enc_out = encoder_in.transpose(1, 2)
        # [bz, d_model]
        out = F.max_pool1d(enc_out, kernel_size=enc_out.size(-1)).squeeze(-1)
        # [bz, label_size]
        return self._linear(out)

