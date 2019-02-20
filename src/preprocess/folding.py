import argparse
import collections
import numpy as np
import random
import torch
from os.path import join
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class Normalization(object):
    def __init__(self, scaler):
        self.scaler = scaler

    def __call__(self, feature, label):
        feature = self.scaler.transform(feature)
        return feature, label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, feature, label):
        # print(len(feature), len(feature[0]))
        # print(feature)
        # print(label)

        # dims: tpoint x feature
        feature = np.array(feature, np.float32)

        # dims: 1
        label = np.array([label], np.int64)

        return torch.from_numpy(feature), torch.from_numpy(label)


class TimeSeriesDataset(Dataset):
    def __init__(self, args, name=None, scaler=None, files=None, tags=None, n_feature=9):
        """

        :param files: list of CSV file
        :param transform:
        """
        self.args = args
        self.window_size = args.window_size
        self.skip = args.skip
        if n_feature:
            self.n_feature = n_feature
        else:
            self.n_feature = 0
        if tags:
            self.tags = tags
        else:
            self.tags = {}
        self.data = []
        self.label = []
        if name:
            self.name = name
        else:
            self.name = 'default-name'
        if files:
            for file in files:
                self.read_raw_data(file)
        self.scaler = scaler
        self.transform = [ToTensor()]

    def read_raw_data(self, path):
        with open(path) as f:
            lines = f.readlines()
        # lines = [x for x in lines if 'unknow' not in x]
        current_label, tmp = None, []
        for line in lines:
            parts = line.strip().split(',')
            if current_label == None:
                tmp.append([float(x) / 180 for x in parts[1:-2]])
                current_label = parts[-1]
            elif current_label == parts[-1]:
                tmp.append([float(x) / 180 for x in parts[1:-2]])
            else:
                if len(tmp) >= self.window_size:
                    for i in range(0, len(tmp) - self.window_size, self.skip):
                        self.data.append(tmp[i:i + self.window_size])
                        self.label.append(current_label)
                current_label = parts[-1]
                tmp = []
        if len(tmp) >= self.window_size:
            for i in range(0, len(tmp) - self.window_size, self.skip):
                self.data.append(tmp[i:i + self.window_size])
                self.label.append(current_label)
        tag_set = set(self.label)
        for idx, tag in enumerate(list(tag_set)):
            self.tags[tag] = idx

        self.n_feature = len(self.data[0][0])
        print('Set n_feature %d' % (self.n_feature))

    def load(self, path):
        import pickle
        with open(path, 'rb') as f:
            self.data += pickle.load(f)
            self.label += pickle.load(f)
            self.tags = pickle.load(f)
            self.n_feature = len(self.data[0][0])

    def dump(self, path):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
            pickle.dump(self.label, f)
            pickle.dump(self.tags, f)

    def add_sample(self, feature, label):
        self.data.append(feature)
        self.label.append(label)

    def folding(self, n_fold=10):
        assert n_fold > 5, 'Number of fold is less then 5'
        datasets = [TimeSeriesDataset(self.args, name=str(i), tags=self.tags, n_feature=self.n_feature) for i in
                    range(n_fold)]
        for idx, (feature, label) in enumerate(zip(self.data, self.label)):
            datasets[idx % n_fold].add_sample(feature, label)

        return datasets

    def over_sample(self):
        print('-> Resample the dataset')
        data_by_label = {}
        length = []
        for tag in self.tags:
            d = [(feature, label) for feature, label in zip(self.data, self.label) if label == tag]
            l = len(d)
            data_by_label[tag] = (d, l)
            length.append(l)
        max_n_sample_by_tag = max(length)
        for tag, (data, l) in data_by_label.items():
            for i in range(0, max_n_sample_by_tag - l):
                x = random.sample(data, 1)
                self.data.append(x[0][0])
                self.label.append(x[0][1])

    def __getitem__(self, idx):
        feature, label = self.data[idx], self.label[idx]
        if self.transform:
            for transform in self.transform:
                sample = transform(feature, self.tags[label])
        else:
            sample = (feature, label)
        return sample

    def __len__(self):
        return len(self.data)

    def print_summary(self):
        print('> Data set summary:')
        print('  | Number of samples: %d ' % (self.__len__()))
        print('  | Window size: %d ' % (self.window_size))
        # print('  | RNN steps: %d ' % (self.step))
        print('  | Features: %d' % (self.n_feature))
        # for tag, index in self.tags.items():
        #     print('   %d  %s'%(index, tag))
        if hasattr(self, 'files'):
            print('  | Source files: %s' % ('; '.join(self.files)))

        print("Label distribution")

        from collections import Counter

        label_counter = Counter(self.label)
        print(label_counter)


def add_model_params(parser):
    parser.add_argument('-output', type=str, default='data/bin/', help='Where to save the data')
    parser.add_argument('-window_size', type=int, default=100, help='Sampling window size')
    parser.add_argument('-skip', type=int, default=70, help='Number of skipping timepoint when sampling')
    return parser


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser = add_model_params(parser)
    args = parser.parse_args()

    data_files = ['data/001/in/data.csv',
                  'data/004/in/data.csv',
                  'data/005/in/data.csv',
                  'data/007/in/data.csv',
                  'data/008/in/data.csv',
                  'data/010/in/data.csv',
                  'data/011/in/data.csv',
                  'data/013/in/data.csv']

    print(args)

    all_dataset = TimeSeriesDataset(args, files=data_files)
    folds = all_dataset.folding(10)
    for idx, dataset in enumerate(folds):
        dataset.dump(join(args.output, dataset.name + '.pickle'))
        dataset.print_summary()

    print('-------------')

    loaded_dataset = [TimeSeriesDataset(args=args, name=str(i)) for i in range(10)]
    for dataset in loaded_dataset:
        dataset.load(join(args.output, dataset.name + '.pickle'))

    for dataset in loaded_dataset:
        dataset.print_summary()
