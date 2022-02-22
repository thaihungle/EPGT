"""LSTM Controller."""
import torch
from torch import nn
from torch.nn import Parameter
import numpy as np


class FFWController(nn.Module):
    """An NTM controller based on LSTM."""
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(FFWController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers



    def create_new_state(self, batch_size):
        h = torch.zeros(batch_size, self.num_outputs)
        if torch.cuda.is_available():
            h = h.cuda()
        return h

    def reset_parameters(self):
        pass

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state):
        return x, prev_state

class LSTMController(nn.Module):
    """An NTM controller based on LSTM."""
    def __init__(self, num_inputs, num_outputs, num_layers):
        super(LSTMController, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=num_inputs,
                            hidden_size=num_outputs,
                            num_layers=num_layers)
        # The hidden state is a learned parameter
        if  torch.cuda.is_available():
            self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs).cuda() * 0, requires_grad=False)
            self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs).cuda() * 0, requires_grad=False)

        else:
            self.lstm_h_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0, requires_grad=False)
            self.lstm_c_bias = Parameter(torch.randn(self.num_layers, 1, self.num_outputs) * 0, requires_grad=False)

        self.reset_parameters()

    def create_new_state(self, batch_size):
        # Dimension: (num_layers * num_directions, batch, hidden_size)
        lstm_h = self.lstm_h_bias.clone().repeat(1, batch_size, 1)
        lstm_c = self.lstm_c_bias.clone().repeat(1, batch_size, 1)
        # h = torch.zeros(self.num_layers, batch_size, self.num_outputs)
        # c = torch.zeros(self.num_layers, batch_size, self.num_outputs)
        # if torch.cuda.is_available():
        #     h = h.cuda()
        #     c = c.cuda()
        # return h,c


        return lstm_h, lstm_c

    def reset_parameters(self):
        for p in self.lstm.parameters():
            if p.dim() == 1:
                nn.init.constant_(p, 0)
            else:
                stdev = 5 / (np.sqrt(self.num_inputs +  self.num_outputs))
                nn.init.uniform_(p, -stdev, stdev)

    def size(self):
        return self.num_inputs, self.num_outputs

    def forward(self, x, prev_state):
        x = x.unsqueeze(0)
        outp, state = self.lstm(x, prev_state)
        return outp.squeeze(0), state