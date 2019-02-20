"""
Some key layers used for constructing a Capsule Network. These layers can used to construct CapsNet on other dataset,
not just on MNIST.
Author: Xifeng Guo, E-mail: `guoxifeng1990@163.com`, Github: `https://github.com/XifengGuo/CapsNet-Pytorch`
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
from torchvision import transforms, datasets

from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix


def squash(inputs, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param inputs: vectors to be squashed
    :param axis: the axis to squash
    :return: a Tensor with same size as inputs
    """
    norm = torch.norm(inputs, p=2, dim=axis, keepdim=True)
    scale = norm ** 2 / (1 + norm ** 2) / (norm + 1e-8)
    return scale * inputs


class DenseCapsule(nn.Module):
    """
    The dense capsule layer. It is similar to Dense (FC) layer. Dense layer has `in_num` inputs, each is a scalar, the
    output of the neuron from the former layer, and it has `out_num` output neurons. DenseCapsule just expands the
    output of the neuron from scalar to vector. So its input size = [None, in_num_caps, in_dim_caps] and output size = \
    [None, out_num_caps, out_dim_caps]. For Dense Layer, in_dim_caps = out_dim_caps = 1.
    :param in_num_caps: number of cpasules inputted to this layer
    :param in_dim_caps: dimension of input capsules
    :param out_num_caps: number of capsules outputted from this layer
    :param out_dim_caps: dimension of output capsules
    :param routings: number of iterations for the routing algorithm
    """

    def __init__(self, in_num_caps, in_dim_caps, out_num_caps, out_dim_caps, routings=3, gpu=0):
        super(DenseCapsule, self).__init__()
        self.in_num_caps = in_num_caps
        self.in_dim_caps = in_dim_caps
        self.out_num_caps = out_num_caps
        self.out_dim_caps = out_dim_caps
        self.routings = routings
        self.gpu = gpu
        # print('Dense capsule:')
        # print('in_num_caps/in_dim_caps: %d/%d'%(in_num_caps, in_dim_caps))
        # print('out_num_caps/out_dim_caps: %d/%d'%(out_num_caps, out_dim_caps))
        self.weight = nn.Parameter(0.01 * torch.randn(out_num_caps, in_num_caps, out_dim_caps, in_dim_caps))
        #
        # print('Weight dims: %s'%(str(self.weight.size())))

    def forward(self, x):
        # x.size=[batch, in_num_caps, in_dim_caps]
        # expanded to    [batch, 1,            in_num_caps, in_dim_caps,  1]
        # weight.size   =[       out_num_caps, in_num_caps, out_dim_caps, in_dim_caps]
        # torch.matmul: [out_dim_caps, in_dim_caps] x [in_dim_caps, 1] -> [out_dim_caps, 1]
        # => x_hat.size =[batch, out_num_caps, in_num_caps, out_dim_caps]
        x_hat = torch.squeeze(torch.matmul(self.weight, x[:, None, :, :, None]), dim=-1)

        # In forward pass, `x_hat_detached` = `x_hat`;
        # In backward, no gradient can flow from `x_hat_detached` back to `x_hat`.
        x_hat_detached = x_hat.detach()

        # The prior for coupling coefficient, initialized as zeros.
        # b.size = [batch, out_num_caps, in_num_caps]
        b = Variable(torch.zeros(x.size(0), self.out_num_caps, self.in_num_caps)).cuda(self.gpu)

        assert self.routings > 0, 'The \'routings\' should be > 0.'
        for i in range(self.routings):
            # c.size = [batch, out_num_caps, in_num_caps]
            c = F.softmax(b, dim=1)

            # At last iteration, use `x_hat` to compute `outputs` in order to backpropagate gradient
            if i == self.routings - 1:
                # c.size expanded to [batch, out_num_caps, in_num_caps, 1           ]
                # x_hat.size     =   [batch, out_num_caps, in_num_caps, out_dim_caps]
                # => outputs.size=   [batch, out_num_caps, 1,           out_dim_caps]
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat))  # alternative way
            else:  # Otherwise, use `x_hat_detached` to update `b`. No gradients flow on this path.
                outputs = squash(torch.sum(c[:, :, :, None] * x_hat_detached, dim=-2, keepdim=True))
                # outputs = squash(torch.matmul(c[:, :, None, :], x_hat_detached))  # alternative way

                # outputs.size       =[batch, out_num_caps, 1,           out_dim_caps]
                # x_hat_detached.size=[batch, out_num_caps, in_num_caps, out_dim_caps]
                # => b.size          =[batch, out_num_caps, in_num_caps]
                b = b + torch.sum(outputs * x_hat_detached, dim=-1)

        return torch.squeeze(outputs, dim=-2)


class PrimaryCapsule(nn.Module):
    """
    Apply Conv2D with `out_channels` and then reshape to get capsules
    :param in_channels: input channels
    :param out_channels: output channels
    :param dim_caps: dimension of capsule
    :param kernel_size: kernel size
    :return: output tensor, size=[batch, num_caps, dim_caps]
    """

    def __init__(self, in_channels, out_channels, dim_caps, kernel_size, stride=1, padding=0, gpu=0):
        super(PrimaryCapsule, self).__init__()
        self.gpu = gpu
        self.dim_caps = dim_caps
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # print('primarycaps> after conv1d: %s'%(str(x.size())))
        outputs = self.conv1d(x)
        # print('primarycaps> after conv1d: %s'%(str(outputs.size())))
        outputs = outputs.view(x.size(0), -1, self.dim_caps)
        # print('primarycaps> after views: %s' % (str(outputs.size())))
        outputs = squash(outputs)
        # print('primarycaps> after squash
        # : %s' % (str(outputs.size())))
        return outputs


class CapsuleNet(nn.Module):
    """
    A Capsule Network on MNIST.
    :param input_size: data size = [channels, width, height]
    :param classes: number of classes
    :param routings: number of routing iterations
    Shape:
        - Input: (batch, channels, width, height), optional (batch, classes) .
        - Output:((batch, classes), (batch, channels, width, height))
    """

    def __init__(self, args, n_feature, n_class):
        super(CapsuleNet, self).__init__()
        self.input_size = n_feature
        self.classes = n_class
        self.routings = args.routings
        self.kernel_size = args.kernel_size
        self.gpu = args.gpu

        input_size = [n_feature, args.window_size, 1]

        # Layer 1: Just a conventional Conv1
        # D layer
        stride = 1
        self.conv1 = nn.Conv1d(n_feature, 256, kernel_size=self.kernel_size,
                               stride=stride, padding=0)
        length = (args.window_size - self.kernel_size) / 1 + 1
        # print('Length %d'%(length))
        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_caps, dim_caps]
        stride = 2
        self.primarycaps = PrimaryCapsule(256, 256, 8, kernel_size=self.kernel_size,
                                          stride=stride, padding=0, gpu=args.gpu)
        #
        length = (length - self.kernel_size) / 2 + 1
        # print('Length %d' % (length))

        # Layer 3: Capsule layer. Routing algorithm works here.
        self.digitcaps = DenseCapsule(in_num_caps=32 * length, in_dim_caps=8,
                                      out_num_caps=self.classes, out_dim_caps=16,
                                      routings=self.routings, gpu=args.gpu)
        # Decoder network.
        self.decoder = nn.Sequential(
            nn.Linear(16 * self.classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, input_size[0] * input_size[1] * input_size[2]),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x, y=None):
        """

        :param x: batch_size x channel x length
        :param y:
        :return:
        """

        x = self.relu(self.conv1(x))
        x = self.primarycaps(x)
        x = self.digitcaps(x)

        length = x.norm(dim=-1)
        if y is None:  # during testing, no label given. create one-hot coding using `length`
            _, index = length.max(dim=1)
            index = index.view(-1, 1).cuda(self.gpu).data
            y = Variable(torch.zeros(length.size()).cuda(self.gpu).scatter_(1, index, 1.).cuda(self.gpu))
        reconstruction = self.decoder((x * y[:, :, None]).view(x.size(0), -1))
        return length, reconstruction.view(-1, self.input_size)


def caps_loss(y_true, y_pred, x, x_recon, lam_recon):
    """
    Capsule loss = Margin loss + lam_recon * reconstruction loss.
    :param y_true: true labels, one-hot coding, size=[batch, classes]
    :param y_pred: predicted labels by CapsNet, size=[batch, classes]
    :param x: input data, size=[batch, channels, width, height]
    :param x_recon: reconstructed data, size is same as `x`
    :param lam_recon: coefficient for reconstruction loss
    :return: Variable contains a scalar loss value.
    """
    L = y_true * torch.clamp(0.9 - y_pred, min=0.) ** 2
    L += 0.5 * (1 - y_true) * torch.clamp(y_pred - 0.1, min=0.) ** 2
    L_margin = L.sum(dim=1).mean()
    L_recon = nn.MSELoss()(x_recon, x)

    return L_margin + lam_recon * L_recon


# def test_capsnet(model, test_loader, args, n_class):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for x, y in test_loader:
#         x, y = torch.transpose(x, dim0=1, dim1=2).cuda(args.gpu), y.cuda(args.gpu)
#         y = torch.zeros(y.size(0), 10).cuda(args.gpu).scatter_(1, y.view(-1, 1), 1.)
#         x, y = Variable(x, volatile=True), Variable(y)
#         y_pred, x_recon = model(x)
#         test_loss += caps_loss(y, y_pred, x, x_recon, args.lam_recon).data[0] * x.size(0)  # sum up batch loss
#         y_pred = y_pred.data.max(1)[1]
#         y_true = y.data.max(1)[1]
#         correct += y_pred.eq(y_true).cpu().sum()
#
#     test_loss /= len(test_loader.dataset)
#     return test_loss, float(correct) / len(test_loader.dataset)
#
#
# def train_capsnet(model, optimizer, train_loader, test_loader, args, n_class):
#     """
#     Training a CapsuleNet
#     :param model: the CapsuleNet model
#     :param train_loader: torch.utils.data.DataLoader for training data
#     :param test_loader: torch.utils.data.DataLoader for test data
#     :param args: arguments
#     :return: The trained model
#     """
#     print('Begin Training' + '-' * 70)
#     from time import time
#     import csv
#     # logfile = open(args.save_dir + '/log.csv', 'w')
#     # logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
#     # logwriter.writeheader()
#
#     t0 = time()
#     best_val_acc = 0.
#     for epoch in range(args.epoch):
#         model.train()  # set to training mode
#         ti = time()
#         training_loss = 0.0
#         for i, (x, y) in enumerate(train_loader):  # batch training
#             # x should have the size of batch_size x channel x height x width
#             x, y = torch.transpose(x, dim0=1, dim1=2).cuda(args.gpu), y.cuda(args.gpu)
#             y = torch.zeros(y.size(0), 10).cuda(args.gpu).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
#             x, y = Variable(x), Variable(y)
#
#             optimizer.zero_grad()  # set gradients of optimizer to zero
#             y_pred, x_recon = model(x, y)  # forward
#             loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
#             loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
#             training_loss += loss.data[0] * x.size(0)  # record the batch loss
#             optimizer.step()  # update the trainable parameters with computed gradients
#
#         # compute validation loss and acc
#         val_loss, val_acc = test_capsnet(model, test_loader, args, n_class)
#         # logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
#         #                         val_loss=val_loss, val_acc=val_acc))
#         print("==> Epoch %02d: loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
#               % (epoch, training_loss / len(train_loader.dataset),
#                  val_loss, val_acc, time() - ti))
#         if val_acc > best_val_acc:  # update best validation acc and save model
#             best_val_acc = val_acc
#             # torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
#             print("best val_acc increased to %.4f" % best_val_acc)
#     # logfile.close()
#     # torch.save(model.state_dict(), args.save_dir + '/trained_model.pkl')
#     # print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)
#     print("Total time = %ds" % (time() - t0))
#     print('End Training' + '-' * 70)
#     return model


def test_capsnet(model, test_loader, args, n_class):
    model.eval()
    test_loss = 0
    correct = 0
    y_trues, y_preds = [], []
    for x, y in test_loader:
        x, y = torch.transpose(x, dim0=1, dim1=2).cuda(args.gpu), y.cuda(args.gpu)
        y = torch.zeros(y.size(0), n_class).cuda(args.gpu).scatter_(1, y.view(-1, 1), 1.)
        x, y = Variable(x, volatile=True), Variable(y)
        y_pred, x_recon = model(x)
        test_loss += caps_loss(y, y_pred, x, x_recon, args.lam_recon).data[0] * x.size(0)  # sum up batch loss
        y_pred = y_pred.data.max(1)[1]
        y_true = y.data.max(1)[1]
        y_trues.append(y_true)
        y_preds.append(y_pred)
        correct += y_pred.eq(y_true).cpu().sum()
    y_trues = torch.cat(y_trues, dim=0).cpu().numpy()
    y_preds = torch.cat(y_preds, dim=0).cpu().numpy()
    test_loss /= len(test_loader.dataset)
    return test_loss, float(correct) / len(test_loader.dataset), y_trues, y_preds

def train_capsnet(model, optimizer, train_loader, test_loader, args, n_class):
    """
    Training a CapsuleNet
    :param model: the CapsuleNet model
    :param train_loader: torch.utils.data.DataLoader for training data
    :param test_loader: torch.utils.data.DataLoader for test data
    :param args: arguments
    :return: The trained model
    """
    print('Begin Training' + '-' * 70)
    from time import time
    import csv
    # logfile = open(args.save_dir + '/log.csv', 'w')
    # logwriter = csv.DictWriter(logfile, fieldnames=['epoch', 'loss', 'val_loss', 'val_acc'])
    # logwriter.writeheader()

    t0 = time()
    best_val_acc = 0.
    best_y_true = None
    best_y_pred = None
    for epoch in range(args.epoch):
        model.train()  # set to training mode
        ti = time()
        training_loss = 0.0
        correct = 0
        for i, (x, y) in enumerate(train_loader):  # batch training
            # x should have the size of batch_size x channel x height x width
            x, y = torch.transpose(x, dim0=1, dim1=2).cuda(args.gpu), y.cuda(args.gpu)
            y = torch.zeros(y.size(0), n_class).cuda(args.gpu).scatter_(1, y.view(-1, 1), 1.)  # change to one-hot coding
            x, y = Variable(x), Variable(y)

            optimizer.zero_grad()  # set gradients of optimizer to zero
            # print('train_capsnet - x: ', x.shape)
            # print('train_capsnet - y: ', y.shape)
            y_pred, x_recon = model(x, y)  # forward
            loss = caps_loss(y, y_pred, x, x_recon, args.lam_recon)  # compute loss
            loss.backward()  # backward, compute all gradients of loss w.r.t all Variables
            training_loss += loss.data[0] * x.size(0)  # record the batch loss
            y_pred = y_pred.data.max(1)[1]
            y_true = y.data.max(1)[1]
            correct += y_pred.eq(y_true).cpu().sum()
            optimizer.step()  # update the trainable parameters with computed gradients
        training_acc = float(correct) / len(train_loader.dataset)
        # compute validation loss and acc
        val_loss, val_acc, y_true, y_pred = test_capsnet(model, test_loader, args, n_class)
        # logwriter.writerow(dict(epoch=epoch, loss=training_loss / len(train_loader.dataset),
        #                         val_loss=val_loss, val_acc=val_acc))
        print("==> Epoch %02d: acc=%.5f, loss=%.5f, val_loss=%.5f, val_acc=%.4f, time=%ds"
              % (epoch, training_acc, training_loss / len(train_loader.dataset),
                 val_loss, val_acc, time() - ti))
        if val_acc > best_val_acc:  # update best validation acc and save model
            best_val_acc = val_acc
            best_y_true = y_true
            best_y_pred = y_pred
            # torch.save(model.state_dict(), args.save_dir + '/epoch%d.pkl' % epoch)
            print("best val_acc increased to %.4f" % best_val_acc)

    p = precision_score(best_y_true, best_y_pred, average=None)
    r = recall_score(best_y_true, best_y_pred, average=None)
    f = f1_score(best_y_true, best_y_pred, average=None)
    cm = confusion_matrix(best_y_true, best_y_pred)

    print('-----------')
    print('Acc= %0.4f'%(best_val_acc))
    print('\t'.join(['%0.4f'%(x) for x in p]))
    print('\t'.join(['%0.4f'%(x) for x in r]))
    print('\t'.join(['%0.4f'%(x) for x in f]))

    p = np.mean(p)
    r = np.mean(r)
    f = np.mean(f)

    print('Average P/R/F: %0.4f\t%0.4f\t%0.4f'%(p,r,f))
    print(cm)

    print('')
    print("Total time = %ds" % (time() - t0))
    print('End Training' + '-' * 70)
    return model
