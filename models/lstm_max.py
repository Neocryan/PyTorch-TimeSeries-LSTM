import torch
import torch.nn as nn


class LstmMax(nn.Module):
    def __init__(self, input_size, hidden_size,dp,bi):
        super(LstmMax, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size,
                            batch_first=True,
                            bidirectional=bi,
                            dropout=dp)
        self.lm = nn.Sequential(nn.Linear(2*hidden_size, input_size),
                                    nn.ReLU(),
                                    nn.Dropout(dp),
                                    nn.Linear(input_size,input_size))

    def forward(self, x):
        x = self.lstm(x)[0]
        x = torch.max(x,dim=1)[0]
        return self.lm(x)
