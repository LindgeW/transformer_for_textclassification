import torch
import torch.nn as nn


# 通用的RNN
class RNNEncoder(nn.Module):
    def __init__(self, input_size,  # 输入的特征维度
                    hidden_size,  # 隐层特征维度
                    num_layers=1,  # RNN层数
                    batch_first=True,  # (batch_size, seq_len, feature_size)
                    bidirectional=False,  # 是否是双向RNN
                    dropout=0.2,  # RNN层与层之间是否dropout,
                    rnn_type='lstm'
                 ):
        super(RNNEncoder, self).__init__()

        self._batch_first = batch_first
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._bidirectional = bidirectional
        self._drop_out = dropout
        self._num_directions = 2 if bidirectional else 1
        self._rnn_type = rnn_type.upper()
        self._RNNs = ['RNN', 'GRU', 'LSTM']
        assert self._rnn_type in self._RNNs

        if self._rnn_type == 'RNN':
            self._rnn_cell = nn.RNNCell
        elif self._rnn_type == 'GRU':
            self._rnn_cell = nn.GRUCell
        elif self._rnn_type == 'LSTM':
            self._rnn_cell = nn.LSTMCell

        self.fw_cells = nn.ModuleList()
        self.bw_cells = nn.ModuleList()
        for layer_i in range(num_layers):
            layer_input_size = input_size if layer_i == 0 else hidden_size * self._num_directions
            self.fw_cells.append(self._rnn_cell(layer_input_size, hidden_size))
            if self._bidirectional:
                self.bw_cells.append(self._rnn_cell(layer_input_size, hidden_size))

    def _forward(self, cell, inputs, init_hidden, mask):
        '''
        :param inputs: [seq_len, batch_size, input_size]
        :param init_hidden: [batch_size, hidden_size], if it is LSTM, init_hidden is tuple type
        :param mask: [seq_len, batch_size, hidden_size]
        :return: [seq_len, batch_size, hidden_size]
        '''
        seq_len = inputs.shape[0]  # inputs.size(0)
        fw_next = init_hidden
        outputs = []
        for xi in range(seq_len):
            if self._rnn_type == 'LSTM':
                # LSTMCell
                # init_hidden: (h0, c0)
                h_next, c_next = cell(inputs[xi], fw_next)
                h_next = h_next * mask[xi] + init_hidden[0] * (1-mask[xi])
                c_next = c_next * mask[xi] + init_hidden[1] * (1-mask[xi])
                fw_next = (h_next, c_next)
                outputs.append(h_next)
            else:
                # RNNCell / GRUCell
                # init_hidden: h0
                fw_next = cell(inputs[xi], fw_next)
                fw_next = fw_next * mask[xi] + init_hidden * (1-mask[xi])
                outputs.append(fw_next)

        return torch.stack(tuple(outputs), dim=0), fw_next

    def _backward(self, cell, inputs, init_hidden, mask):
        '''
       :param inputs: [seq_len, batch_size, input_size]
       :param init_hidden: [batch_size, hidden_size], if it is LSTM, init_hidden is tuple type
       :param mask: [seq_len, batch_size, hidden_size]
       :return: [seq_len, batch_size, hidden_size]
       '''
        seq_len = inputs.shape[0]  # inputs.size(0)
        bw_next = init_hidden
        outputs = []
        for xi in reversed(range(seq_len)):
            if self._rnn_type == 'LSTM':
                # LSTMCell
                # init_hidden: (h0, c0)
                h_next, c_next = cell(inputs[xi], bw_next)
                h_next = h_next * mask[xi] + init_hidden[0] * (1 - mask[xi])
                c_next = c_next * mask[xi] + init_hidden[1] * (1 - mask[xi])
                bw_next = (h_next, c_next)
                outputs.append(h_next)
            else:
                # RNNCell / GRUCell
                # init_hidden: h0
                bw_next = cell(inputs[xi], bw_next)
                bw_next = bw_next * mask[xi] + init_hidden * (1 - mask[xi])
                outputs.append(bw_next)
        outputs.reverse()
        return torch.stack(tuple(outputs), dim=0), bw_next

    def _init_hidden(self, batch_size, device=torch.device('cpu')):
        h0 = torch.zeros(batch_size, self._hidden_size, device=device)
        if self._rnn_type == 'LSTM':
            return h0, h0
        else:
            return h0

    # 自定义dropout: 所有节点的输出以概率p被置为0
    def _dropout(self, inputs, p=0.5, training=False):
        if training:
            if inputs.dim() == 2:  # [batch_size, input_size]
                drop_mask = torch.zeros(inputs.shape, device=inputs.device).fill_(1 - p)
                # 所有元素以概率(1-p)被置成1
                drop_mask = torch.bernoulli(drop_mask)  # 两点分布，只返回0或1
                # 因为评估的时候不需要dropout，为了保证期望值一样(保证网络的每一层在训练阶段和测试阶段数据分布相同)，因此需要除以(1-p)
                inputs *= drop_mask
                inputs /= (1 - p)
            elif inputs.dim() == 3:  # [seq_len, batch_size, input_size]
                drop_mask = torch.zeros((inputs.size(1), inputs.size(2)), device=inputs.device).fill_(1 - p)
                # 所有元素以概率(1-p)被置成1
                drop_mask = torch.bernoulli(drop_mask)  # 两点分布，只返回0或1
                # [batch_size, input_size, seq_len] -> [seq_len, batch_size, input_size]
                drop_mask = drop_mask.unsqueeze(-1).expand((-1, -1, inputs.size(0))).permute((2, 0, 1))
                inputs *= drop_mask
                inputs /= (1 - p)
        return inputs

    def forward(self, inputs, mask, init_hidden=None):
        '''
        :param inputs: [batch_size, seq_len, input_size]
        :param mask: [batch_size, seq_len]
        :param init_hidden: [batch_size, hidden_size]
        :return:
            out: [batch_size, seq_len, hidden_size * num_directions]
            hidden: [num_layer, batch_size, hidden_size * num_directions]
        '''
        if self._batch_first:
            # [seq_len, batch_size, input_size]
            inputs = inputs.transpose(0, 1)
            # [seq_len, batch_size]
            mask = mask.transpose(0, 1)

        # [seq_len, batch_size] -> [seq_len, batch_size, 1]
        # -> [seq_len, batch_size, hidden_size]
        mask = mask.unsqueeze(dim=2).expand((-1, -1, self._hidden_size))

        batch_size = inputs.shape[1]
        if init_hidden is None:
            init_hidden = self._init_hidden(batch_size, inputs.device)

        hn, cn = [], []
        for i in range(self._num_layers):
            if i != 0:
                inputs = self._dropout(inputs, p=self._drop_out, training=self.training)
            # fw_out: [seq_len, batch_size, hidden_size]
            # fw_hidden: (hn, cn) [batch_size, hidden_size]
            fw_out, fw_hidden = self._forward(self.fw_cells[i], inputs, init_hidden, mask)
            bw_out, bw_hidden = None, None
            if self._bidirectional:
                bw_out, bw_hidden = self._backward(self.bw_cells[i], inputs, init_hidden, mask)

            if self._rnn_type == 'LSTM':
                hn.append(torch.cat((fw_hidden[0], bw_hidden[0]), dim=1) if self._bidirectional else fw_hidden[0])
                cn.append(torch.cat((fw_hidden[1], bw_hidden[1]), dim=1) if self._bidirectional else bw_hidden[1])
            else:
                # RNN / GRU
                hn.append(torch.cat(fw_hidden, bw_hidden) if self._bidirectional else fw_hidden)

            # out: [seq_len, batch_size, hidden_size * num_directions]
            inputs = torch.cat((fw_out, bw_out), dim=2) if self._bidirectional else fw_out

        # [batch_size, seq_len, hidden_size * num_directions]
        output = inputs.transpose(0, 1) if self._batch_first else inputs

        hn = torch.stack(tuple(hn), dim=0)
        # hn, cn: [num_layer, batch_size, hidden_size * num_directions]
        if self._rnn_type == 'LSTM':
            cn = torch.stack(tuple(cn), dim=0)
            hidden = (hn, cn)
        else:
            hidden = hn

        return output, hidden
