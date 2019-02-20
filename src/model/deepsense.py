import torch
import torch.nn as nn
from torch.autograd import Variable
import math

SEPCTURAL_SAMPLES = 10
FEATURE_DIM = SEPCTURAL_SAMPLES * 6 * 2
CONV_LEN = 3
CONV_LEN_INTE = 3  # 4
CONV_LEN_LAST = 3  # 5
CONV_NUM = 64
CONV_MERGE_LEN = 8
CONV_MERGE_LEN2 = 6
CONV_MERGE_LEN3 = 4
CONV_NUM2 = 64
INTER_DIM = 120
OUT_DIM = 6  # len(idDict)
WIDE = 20


###### Import training data


class SingleSensorTransformer(nn.Module):
    def __init__(self, args, n_feature=3):
        super(SingleSensorTransformer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=CONV_NUM,
                               kernel_size=(2 * 3 * CONV_LEN, 1), stride=(2 * 3, 1), padding=0)
        self.batch_norm1 = nn.BatchNorm2d(CONV_NUM)

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=args.dropout)

        self.conv2 = nn.Conv2d(in_channels=CONV_NUM, out_channels=CONV_NUM,
                               kernel_size=(CONV_LEN_INTE, 1), stride=(1, 1), padding=0)
        self.batch_norm2 = nn.BatchNorm2d(CONV_NUM)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=args.dropout)

        self.conv3 = nn.Conv2d(in_channels=CONV_NUM, out_channels=CONV_NUM,
                               kernel_size=(CONV_LEN_LAST, 1), stride=(1, 1), padding=0)
        self.batch_norm3 = nn.BatchNorm2d(CONV_NUM)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        """

        :param x: b(batch, channel, length)
        :return:
        """
        # Assume that x (batch, wide, feature_dim, channel=1)

        # (batch, wide, feature_dim, channel = 1)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        return x


class MultipSensorTransformer(nn.Module):
    def __init__(self, args):
        super(MultipSensorTransformer, self).__init__()
        n_feature = CONV_NUM * 3
        self.conv1 = nn.Conv2d(in_channels=CONV_NUM * 3, out_channels=CONV_NUM2,
                               kernel_size=(2 * 3 * CONV_LEN, 1), stride=(CONV_MERGE_LEN, 1), padding=0)
        self.batch_norm1 = nn.BatchNorm2d(CONV_NUM2)

        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=args.dropout)

        self.conv2 = nn.Conv2d(in_channels=CONV_NUM2, out_channels=CONV_NUM2,
                               kernel_size=(CONV_LEN_INTE, 1), stride=(CONV_MERGE_LEN2, 1), padding=0)
        self.batch_norm2 = nn.BatchNorm2d(CONV_NUM2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=args.dropout)

        self.conv3 = nn.Conv2d(in_channels=CONV_NUM2, out_channels=CONV_NUM2,
                               kernel_size=(CONV_LEN_LAST, 1), stride=(CONV_MERGE_LEN3, 1), padding=0)
        self.batch_norm3 = nn.BatchNorm2d(CONV_NUM2)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        # Assume that x (batch, wide, feature_dim, channel=1)
        x = self.dropout1(x)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1()

        x = self.dropout2(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2()

        x = self.dropout3(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3()

        return x


# Define the model
class DeepSense(nn.Module):
    def __init__(self, args, n_feature, n_class):
        super(DeepSense, self).__init__()
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

        self.acce_shoe_net = SingleSensorTransformer(args, n_feature=3)
        self.acce_watch_net = SingleSensorTransformer(args, n_feature=3)
        self.gyro_net = SingleSensorTransformer(args, n_feature=3)

        self.sensor_net = MultipSensorTransformer(args)

        self.rnn = nn.GRU(self.hidden_size, self.hidden_size, num_layers=2,
                          dropout=dropout, batch_first=True, bidirectional=False)

        self.dense =nn.Linear(self.hidden_size, self.n_class)

    def forward(self, x, hidden=None):
        """

        :param x: batch_size x (tpoint_per_step * recurrent_step) x n_feature
        :param hidden:
        :return:
        """
        print('-')

        print(x.shape)
        # Split into three parts
        # (batch, length, feature_dim) -> (batch, channel=1, length, feature_dim)
        x = torch.unsqueeze(x, 1)
        x_acc_shoe, x_acc_watch, x_gyro = torch.split(x, split_size=3, dim=3)
        # x_acc_shoe = Variable(torch.transpose(x_acc_shoe, 1, 2))
        # x_acc_watch = Variable(torch.transpose(x_acc_watch, 1, 2))
        # x_gyro = Variable(torch.transpose(x_gyro, 1, 2))

        x_acc_shoe = Variable(x_acc_shoe)
        x_acc_watch = Variable(x_acc_watch)
        x_gyro = Variable(x_gyro)

        print(x_acc_shoe.shape)
        print(x_acc_watch.shape)
        print(x_gyro.shape)

        x_acc_shoe = self.acce_shoe_net(x_acc_shoe)
        x_acc_watch = self.acce_watch_net(x_acc_watch)
        x_gyro = self.gyro_net(x_gyro)

        print('-')
        print(x_gyro.shape)

        x = torch.cat([x_acc_shoe, x_acc_watch, x_gyro])
        x = self.sensor_net(x)

        x, hidden = self.rnn(x)

        x = self.dense(x)

        return x
