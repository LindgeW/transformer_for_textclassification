# import torch.nn.functional as F
import torch.nn as nn
from .activations import GeLU


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.ffn = nn.Sequential(
            nn.Conv1d(in_channels=d_in,
                      out_channels=d_ff,
                      kernel_size=1),  # 权重共享
            # nn.ReLU(),
            GeLU(),
            nn.Conv1d(in_channels=d_ff,
                      out_channels=d_in,
                      kernel_size=1)
        )

        self._layer_norm = nn.LayerNorm(d_in)

        self._dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        '''
        :param inputs: [bz, len_q, d_model]
        :return: [bz, len_q, d_model]
        '''
        residual = inputs

        # [bz, len_q, d_model] -> [bz, d_model, len_q]
        fc_in = self._layer_norm(inputs).transpose(1, 2)

        # [bz, d_model, len_q]
        fc_out = self.ffn(fc_in)

        # [bz, len_q, d_model]
        out = fc_out.transpose(1, 2)

        if self.training:
            out = self._dropout(out)

        # return self._layer_norm(residual + out)
        return residual + out