import torch
from torch import nn


class MolModel(nn.Module):
    def __init__(self, n_inputs, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.LSTMCell(n_inputs, hidden_size)
        self.linear = nn.Linear(hidden_size, n_inputs)

    def forward(self, x, hidden_states=None):
        x = x.transpose(0, 1)  # (seq, batch, size)

        outputs = []

        for x_t in x:
            h, c = self.rnn(x_t, hidden_states)

            hidden_states = h, c

            out = self.linear(h.view(-1, self.hidden_size))

            outputs.append(out)

        outputs = torch.stack(outputs)

        return outputs.transpose(0, 1), hidden_states
