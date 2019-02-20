import collections
import numpy as np
import random
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    def __init__(self, files, args, scaler=None, tags=None):
        """

        :param files: list of CSV file
        :param transform:
        """
        self.files = files
        self.data = []
        self.label = []
        self.window_size = args.window_size
        # self.step = args.step
        self.skip = args.skip

        self.n_feature = 0
        if tags and scaler:
            # print('Using existing tags set and scaler: %s %s') % (str(tags), str(scaler))
            self.scaler = scaler
            self.tags = tags
            self.build_tag_set = False
            for file in files:
                print "reading test data " + file
                self._read_test_data_(file)
        else:
            self.tags = {}
            self.build_tag_set = True
            data = []
            for file in files:
                print "reading train data " + file
                _data = self._read_train_data_(file)
                data += _data
            # print('Fitting scaler')
            self.scaler = MinMaxScaler(feature_range=(-1, 1)).fit(data)

            # print(self.scaler)
            # Sprint('---')
        self.n_class = len(self.tags)
        self.transform = [Normalization(self.scaler),
                          ToTensor()]
        #self.print_summary()

    def _read_train_data_(self, path):
        with open(path) as f:
            lines = f.readlines()
        current_label, tmp = None, []
        all_data = []

        # lines = [x for x in lines if 'unknow' not in x]
        for line in lines[1:]:
            parts = line.strip().split(',')
            _tmp = []
            for i in range(199):
                _tmp.append([float(x) / 1024 for x in parts[79 * i:79 * (i + 1)]])
            if current_label == None:
                tmp.append(np.mean(_tmp, axis=0))
                current_label = parts[-1]
            elif current_label == parts[-1]:
                tmp.append(np.mean(_tmp, axis=0))
            else:
                if len(tmp) >= self.window_size:
                    for i in range(0, len(tmp) - self.window_size, self.skip):
                        self.data.append(tmp[i:i + self.window_size])
                        self.label.append(current_label)
                current_label = parts[-1]
                all_data += tmp
                tmp = []
        if len(tmp) >= self.window_size:
            for i in range(0, len(tmp) - self.window_size, self.skip):
                self.data.append(tmp[i:i + self.window_size])
                self.label.append(current_label)

        self.n_feature = len(self.data[0][0])
        tag_set = set(self.label)
        for idx, tag in enumerate(list(tag_set)):
            self.tags[tag] = idx
        self.n_class = len(self.tags)

        return all_data

    def _read_test_data_(self, path):
        with open(path) as f:
            lines = f.readlines()
        # lines = [x for x in lines if 'unknow' not in x]
        current_label, tmp = None, []
        for line in lines[1:]:
            parts = line.strip().split(',')
            _tmp = []
            for i in range(199):
                _tmp.append([float(x) / 1024 for x in parts[79 * i:79 * (i + 1)]])
            if current_label == None:
                tmp.append(np.mean(_tmp, axis=0))
                current_label = parts[-1]
            elif current_label == parts[-1]:
                tmp.append(np.mean(_tmp, axis=0))
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
        self.n_feature = len(self.data[0][0])

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
        print('  | Tags: %d (%s)' % (self.n_class, ', '.join(self.tags.keys())))
        # for tag, index in self.tags.items():
        #     print('   %d  %s'%(index, tag))
        print('  | Source files: %s' % ('; '.join(self.files)))

        print("Label distribution")

        from collections import Counter

        label_counter = Counter(self.label)
        print(label_counter)


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


class Argument():
    def __init__(self):
        self.tpoint = 3
        self.step = 3


if __name__ == '__main__':
    files = ['data/sample.csv']
    args = Argument()
    dataset = TimeSeriesDataset(files, args)
    dataset.print_summary()

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)

    for idx, (feature, label) in enumerate(dataloader):
        print(feature.shape)
        print(label.shape)