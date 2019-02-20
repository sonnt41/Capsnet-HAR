import torch
import torch.nn as nn
from torch.autograd import Variable


# Define the model
class DeepConvLSTM(nn.Module):
    def __init__(self, args, n_feature, n_class):
        super(DeepConvLSTM, self).__init__()
        w = args.window_size
        p = self.tpoint = args.tpoint

        self.n_class = n_class
        self.n_feature = n_feature
        self.hidden_size = args.unit
        dropout = args.dropout
        if w % args.tpoint == 0:
            self.rnn_step = w / p
        else:
            self.rnn_step = w / p + 1
        padding_size = self.rnn_step * p - w
        self.padding = nn.ZeroPad2d((0, 0, padding_size, 0))

        # print(' | Input dim: %d' % (self.input_dim))
        # print(' | RNN step: %d' % (self.rnn_step))
        # print(' | Tpoint step: %d' % (p))
        # print(' | Feature: %d' % (n_feature))
        # print(' | Padding: %d' % (padding_size))
        # print(' | RNN layer: %d' % (args.layer))
        self.n_class = n_class

        self.n_sensor_channel = 9
        self.n_filter = 64
        self.filter_size = 11
        self.sliding_window_length = 10
        self.sliding_window_step = 5

        self.padding = nn.ConstantPad2d((self.filter_size / 2, self.filter_size / 2), 0.0)

        self.conv1 = nn.Conv1d(self.n_feature, self.n_filter, kernel_size=self.filter_size, padding=0)
        self.conv2 = nn.Conv1d(self.n_filter, self.n_filter, kernel_size=self.filter_size, padding=0)
        self.conv3 = nn.Conv1d(self.n_filter, self.n_filter, kernel_size=self.filter_size, padding=0)
        self.conv4 = nn.Conv1d(self.n_filter, self.n_filter, kernel_size=self.filter_size, padding=0)

        self.lstm1 = nn.LSTM(self.n_filter, self.hidden_size, num_layers=2,
                             dropout=dropout, batch_first=True, bidirectional=False)

        self.dense1 = nn.Linear(w * self.hidden_size, 512)
        self.dense2 = nn.Linear(512, self.n_class)

    def forward(self, x, hidden=None):
        """

        :param x: batch_size x (tpoint_per_step * recurrent_step) x n_feature
        :param hidden:
        :return:
        """

        x = Variable(torch.transpose(x, 1, 2))

        #print(x.shape)

        # x = x.unsqueeze(-1)



        x = self.padding(x)
        # batch x channel x length
        x = self.conv1(x)

        #print(x.shape)
        x = self.padding(x)
        x = self.conv2(x)
        #print(x.shape)
        x = self.padding(x)
        x = self.conv3(x)
        #print(x.shape)
        x = self.padding(x)
        # batch x channel x length
        x = self.conv4(x)

        # length x input_dim x batch (note: channel = input_dim)
        x = torch.transpose(x, 0,2)
        # length x batch x input_dim
        x = torch.transpose(x, 1,2)

        x, hidden = self.lstm1(x)

        # batch x length x hidden_size
        x = torch.transpose(x, 0,1)

        # batch x (length*hidden_size)
        x = x.view(x.shape[0], -1)

        x = self.dense1(x)
        x = self.dense2(x)

        #print(x.shape)

        return x
