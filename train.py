from __future__ import print_function

import copy
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import math, random
import argparse

import src.preprocess.folding as folding
from os.path import join
from src.eval import accuracy
from src.model.lstm import StackedLSTM
from src.model.gru import StackedGRU
from src.model.deepConvLSTM import DeepConvLSTM
from src.model.deepsense import DeepSense
from src.model.capsulenet import CapsuleNet, train_capsnet, test_capsnet
from src.preprocess.data import TimeSeriesDataset

from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score


def train(model, dataloader, criterion, optimizer, args):
    # Feature: batch_size x RNN_steps x embedding_dim
    # Label: batch_size x 1
    n_batch = len(dataloader)

    for idx, (feature, label) in enumerate(dataloader):
        # batch_size x 1 -> batch_size
        if args.gpu > -1:
            model = model.cuda()
            criterion = criterion.cuda()
            feature = feature.cuda()
            # print(feature.mean(), feature.std())
            label = Variable(label.squeeze()).cuda()
        else:
            label = Variable(label.squeeze())
        # print(feature.shape)
        # Output should be batch_size x n_class
        outputs = model(feature)
        optimizer.zero_grad()
        l2_reg = Variable(torch.cuda.FloatTensor(1), requires_grad=True)
        for W in model.parameters():
            l2_reg = l2_reg + W.norm(2)

        loss = criterion(outputs, label) + l2_reg * 0.0005

        # print('Loss/Reg: %f/%f'%(loss.data[0], l2_reg.data[0]))
        loss.backward()
        optimizer.step()
        if idx % 50 == 0:
            avg_loss = loss.data[0] / args.batch


def test(model, dataloader, criterion, args, cm=False, with_return_y=False):
    losses = 0.0
    correct, total = 0.0, 0.0
    y_true, y_pred = [], []
    for idx, (feature, label) in enumerate(dataloader):
        # batch_size x 1 -> batch_size
        if args.gpu > -1:
            model = model.cuda()
            criterion = criterion.cuda()
            feature = feature.cuda()
            label = Variable(label.squeeze()).cuda()
        else:
            label = Variable(label.squeeze())
        y_true.append(label.data)

        # Output should be batch_size x n_class
        scores = model(feature)

        max, pred = scores.max(1)

        y_pred.append(pred)
        # print 'Output shape: ', outputs.shape
        # print 'Label shape : ', label.shape

        loss = criterion(scores, label)
        losses += loss.data[0]

        _correct, _total = accuracy(scores, label)
        correct += _correct
        total += _total
    print('> loss=%.4f acc=%.4f' % (losses / total, correct / total), end='')
    if with_return_y:
        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).data.cpu().numpy()
        return scores, losses, correct, total, y_true, y_pred
    if cm:
        y_true = torch.cat(y_true, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).data.cpu().numpy()

        # print(type(y_true))
        # print(type(y_pred))
        cm = confusion_matrix(y_true, y_pred)
        print(' ')
        print(cm)

    return scores, losses, correct, total


def infer(model, dataloader):
    softmax = nn.Softmax(1)
    prediction = []
    labels = []
    for idx, (feature, label) in enumerate(dataloader):
        # batch_size x 1 -> batch_size
        if args.gpu > -1:
            model = model.cuda()
            feature = feature.cuda()
        labels.append(label)

        # Output should be batch_size x n_class
        scores = model(feature)
        prediction.append(scores)

    prediction = torch.cat(prediction, 0)
    probs = softmax(prediction)
    return probs, Variable(torch.cat(labels, 0).cuda())


def train_folding(data_files, args):
    print ("TRAIN FOLDING")
    criterion = nn.CrossEntropyLoss()
    model_class = globals()[args.model]

    for test_file in data_files:
        train_files = [x for x in data_files if x != test_file]

        train_dataset = folding.TimeSeriesDataset(args=args)
        for file in train_files:
            train_dataset.load(file)
        train_dataset.over_sample()
        test_dataset = folding.TimeSeriesDataset(args=args)
        test_dataset.load(test_file)
        n_class = len(train_dataset.tags)
        model = model_class(args, train_dataset.n_feature, n_class)

        model.cuda(args.gpu)

        if args.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.moment, nesterov=True)

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch,
                                      shuffle=True, num_workers=args.worker)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch,
                                     shuffle=True, num_workers=args.worker)
        if args.model == 'CapsuleNet':
            train_capsnet(model, optimizer, train_dataloader, test_dataloader, args, n_class)
        else:
            best = 0.0
            best_at = 0
            best_y_true = None
            best_y_pred = None
            for epoch in range(args.epoch):
                print('\n Epoch %d/%d ' % (epoch, args.epoch), end='')
                loss = train(model, train_dataloader, criterion, optimizer, args=args)
                loss = test(model, train_dataloader, criterion, args)
                scores, loss, correct, total, y_true, y_pred = test(model, test_dataloader, criterion, args, cm=False, with_return_y=True)
                if correct / total > best:
                    best = correct / total
                    best_at = epoch
                    best_y_true = y_true
                    best_y_pred = y_pred
            print('Best at epoch %d: %f' % (best_at, best))
            p = precision_score(best_y_true, best_y_pred, average=None)
            r = recall_score(best_y_true, best_y_pred, average=None)
            f = f1_score(best_y_true, best_y_pred, average=None)
            cm = confusion_matrix(best_y_true, best_y_pred)

            print('-----------')
            print('Acc= %0.4f'%(best))
            print('\t'.join(['%0.4f'%(x) for x in p]))
            print('\t'.join(['%0.4f'%(x) for x in r]))
            print('\t'.join(['%0.4f'%(x) for x in f]))

            p = np.mean(p)
            r = np.mean(r)
            f = np.mean(f)

            print('Average P/R/F: %0.4f\t%0.4f\t%0.4f'%(p,r,f))
            print(cm)


def train_ensemble(data_files, args):
    criterion = nn.CrossEntropyLoss()
    model_class = globals()[args.model]
    for idx, test_file in enumerate(data_files):
        train_files = [x for x in data_files if x != test_file]
        valid_file = train_files[0]
        train_files = train_files[1:]
        test_dataset = TimeSeriesDataset([test_file], args)

        valid_dataset = TimeSeriesDataset([valid_file], args, tags=test_dataset.tags, scaler=test_dataset.scaler)

        valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch,
                                      shuffle=False, num_workers=args.worker)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch,
                                     shuffle=False, num_workers=args.worker)

        # Train 11 classifiers
        classifiers = []
        for file in train_files:
            train_dataset = TimeSeriesDataset([file], args, tags=test_dataset.tags, scaler=test_dataset.scaler)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch,
                                          shuffle=True, num_workers=args.worker)
            model = model_class(args, train_dataset.n_feature, train_dataset.n_class)
            if args.optim == 'adam':
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
            else:
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.moment, nesterov=True)
            train_dataloader = DataLoader(train_dataset, batch_size=args.batch,
                                          shuffle=True, num_workers=args.worker)
            best_loss = 100000
            for epoch in range(args.epoch):
                # print('Epoch %3d/%d ' % (epoch, args.epoch), end='')
                loss = train(model, train_dataloader, criterion, optimizer, args=args)
                predict, loss, _, _ = test(model, train_dataloader, criterion, args)
                predict, loss, correct, total = test(model, valid_dataloader, criterion, args)
                if loss < best_loss and correct / total > 0.5:
                    best_loss = loss
                    best_model = copy.deepcopy(model)
                    # predict, loss = test(best_model, valid_dataloader, criterion, args)
                    print(' <')
                else:
                    print(' ')
            if best_loss != 100000:
                classifiers.append(best_model)

        # Majority voting on 11 classifiers
        print('Number of classifiers: %d' % (len(classifiers)))
        probs, labels = None, None

        for classifier in classifiers:
            # print('\n')
            _prob, _labels = infer(classifier, valid_dataloader)
            labels = _labels
            if probs is not None:
                probs += _prob
            else:
                probs = _prob

        correct, total = accuracy(probs, labels)
        print(correct / (total + 1e-10))
        # print('-----> Acc: %f'%(acc))


def main(args):
    # raw_data_files = ['data/001/in/data.csv',
    #                   'data/004/in/data.csv',
    #                   'data/005/in/data.csv',
    #                   'data/007/in/data.csv',
    #                   'data/008/in/data.csv',
    #                   'data/010/in/data.csv',
    #                   'data/011/in/data.csv',
    #                   'data/013/in/data.csv']

    # train_data_files = ['breath/data_Feature_SILENT6.csv']
    train_data_files = ['breath/data_Feature_SILENT1.csv',
                      'breath/data_Feature_SILENT2.csv',
                      'breath/data_Feature_SILENT3.csv',
                      'breath/data_Feature_SILENT4.csv',
                      'breath/data_Feature_SILENT5.csv']
    # test_data_files = ['breath/data_Feature_SILENT1.csv']
    test_data_files = ['breath/data_Feature_SILENT6.csv',
                      'breath/data_Feature_SILENT7.csv']

    bin_data_files = ['data/bin/in/0.pickle',
                      'data/bin/in/1.pickle',
                      'data/bin/in/2.pickle',
                      'data/bin/in/3.pickle',
                      'data/bin/in/4.pickle',
                      'data/bin/in/5.pickle',
                      'data/bin/in/6.pickle',
                      'data/bin/in/7.pickle',
                      'data/bin/in/8.pickle',
                      'data/bin/in/9.pickle']

    torch.cuda.manual_seed_all(4444)

    if args.gpu > -1:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    criterion = nn.CrossEntropyLoss()

    if args.ensemble:
        train_ensemble(train_data_files, args)
    elif args.folding > -1:
        print('Train 10-fold')
        train_folding(bin_data_files, args)
    else:
        model_class = globals()[args.model]


        train_dataset = TimeSeriesDataset(train_data_files, args)

        # sys.exit(0)
        train_dataset.over_sample()
        test_dataset = TimeSeriesDataset(test_data_files, args, tags=train_dataset.tags, scaler=train_dataset.scaler)

        print('> Model summary:')
        model = model_class(args, train_dataset.n_feature, train_dataset.n_class)

        if args.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        else:
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.moment, nesterov=True)

        train_dataset.print_summary()
        test_dataset.print_summary()

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch,
                                      shuffle=True, num_workers=args.worker)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch,
                                     shuffle=True, num_workers=args.worker)
        if args.model == 'CapsuleNet':
            train_capsnet(model, optimizer, train_dataloader, test_dataloader, args, train_dataset.n_class)
        else:
            best = 0.0
            best_at = 0
            best_y_true = None
            best_y_pred = None
            for epoch in range(args.epoch):
                print('\n Epoch %d/%d ' % (epoch, args.epoch), end='')
                loss = train(model, train_dataloader, criterion, optimizer, args=args)
                loss = test(model, train_dataloader, criterion, args)
                scores, loss, correct, total, y_true, y_pred = test(model, test_dataloader, criterion, args, cm=True, with_return_y=True)
                if correct / total > best:
                    best = correct / total
                    best_at = epoch
                    best_y_true = y_true
                    best_y_pred = y_pred

            print ('\nBest at epoch %d: %f' % (best_at, best))
            print ("Precision")
            print (precision_score(best_y_true, best_y_pred, average=None))
            print ("Recall")
            print (recall_score(best_y_true, best_y_pred, average=None))
            print ("f1_score")
            print (f1_score(best_y_true, best_y_pred, average=None))
            print ("confusion_matrix")
            print (confusion_matrix(best_y_true, best_y_pred))


def add_capsnet_params(parser):
    parser.add_argument('-r', '--routings', default=3, type=int,
                        help="Number of iterations used in routing algorithm. should > 0")  # num_routing should > 0
    parser.add_argument('-shift_pixels', default=2, type=int,
                        help="Number of pixels to shift at most in each direction.")
    parser.add_argument('-lam_recon', default=0.0005, type=float,
                        help="The coefficient for the loss of decoder")
    return parser


def add_model_params(parser):
    parser.add_argument('-model', type=str, default='lstm', help='Model type [capsnet, gru, lstm, cnn]')
    parser.add_argument('-unit', type=int, default=50, help='Number of hidden units')
    parser.add_argument('-layer', type=int, default=1, help='Number of LSTM layers')
    parser.add_argument('-window_size', type=int, default=100, help='Sampling window size')
    parser.add_argument('-tpoint', type=int, default=10, help='Number of fed timepoints per step')
    parser.add_argument('-skip', type=int, default=5, help='Number of skipping timepoint when sampling')
    parser.add_argument('-kernel_size', type=int, default=9, help='CNN kernel size')
    parser.add_argument('-folding', type=int, default=-1, help='Run n-fold cross validation')
    return parser


def add_training_params(parser):
    parser.add_argument('-optim', default='adam', help='Optimizer: adam, lbfgs, sgd')
    parser.add_argument('-batch', type=int, default=10, help='Batch size')
    parser.add_argument('-lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-moment', type=float, default=0.99, help='Momentum factor of optimizer')
    parser.add_argument('-decay', type=float, default=0.99, help='Learning rate decay')
    parser.add_argument('-epoch', type=int, default=20, help='Maximum number of training epochs')
    parser.add_argument('-gpu', type=int, default=-1, help='GPU ID')
    parser.add_argument('-worker', type=int, default=4, help='Number of preprocessing workers')
    parser.add_argument('-dropout', type=float, default=0.1, help='Dropout probability')
    parser.add_argument('-ensemble', action='store_true', default=False, help='Run ensemble learning')
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_model_params(parser)
    parser = add_training_params(parser)
    parser = add_capsnet_params(parser)
    args = parser.parse_args()
    # parser.print_usage()

    print(args)

    main(args)
