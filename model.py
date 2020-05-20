from torch import nn
from widis_lstm_tools.nn import LSTMLayer


class MolModel(nn.Module):
    def __init__(self, n_inputs, hidden_size):
        super().__init__()

        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(n_inputs, hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, n_inputs)

    def forward(self, x, hidden=None):

        _, (h, c) = self.rnn(x, hidden)

        return self.linear(h.view(-1, self.hidden_size)), (h, c)
