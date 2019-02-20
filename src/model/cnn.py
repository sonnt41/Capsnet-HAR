import torch
import torch.nn as nn


# Define the model
class Transformer(nn.Module):
    def __init__(self, args, n_feature, n_class):
        super(Transformer, self).__init__()
        w = args.window_size
        p = self.tpoint = args.tpoint

        self.n_class = n_class
        self.n_feature = n_feature
        self.hidden_size = args.unit
        self.input_dim = p * n_feature
        dropout = args.dropout
        if w % args.tpoint == 0:
            self.rnn_step = w / p
        else:
            self.rnn_step = w / p + 1
        padding_size = self.rnn_step * p - w
        self.padding = nn.ZeroPad2d((0, 0, padding_size, 0))

        print(' | Input dim: %d' % (self.input_dim))
        print(' | RNN step: %d' % (self.rnn_step))
        print(' | Tpoint step: %d' % (p))
        print(' | Feature: %d' % (n_feature))
        print(' | Padding: %d' % (padding_size))
        print(' | RNN layer: %d' % (args.layer))
        self.n_class = n_class

        self.norm = nn.BatchNorm1d(self.rnn_step)


    def forward(self, inputs, hidden=None):
        """

        :param inputs: batch_size x (tpoint_per_step * recurrent_step) x n_feature
        :param hidden:
        :return:
        """

        #print(inputs.shape)

        batch_size = inputs.shape[0]

        # Pad zero to the start of sequence
        inputs = self.padding(inputs)
        #print(inputs.shape)



        # Reshape batch_size x (tpoint_per_step * rnn_step) x n_feature
        #         -> batch_size x rnn_step x (tpoint * n_feature)
        inputs = inputs.view(batch_size, self.rnn_step, self.tpoint * self.n_feature)
        #print(inputs.shape)
        inputs = self.input(inputs)
        inputs = self.norm(inputs)
        for idx, (rnn, linear) in enumerate(zip(self.rnns, self.projections)):
            #print(idx, rnn)
            inputs, hidden = rnn(inputs)
            inputs = linear(inputs)
            inputs = self.norm(inputs.contiguous())

        inputs = self.out(inputs)
        return torch.sum(inputs, 1)
