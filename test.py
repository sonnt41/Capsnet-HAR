import pandas as pd
import os
import sys

dataset_dir = '/home/quyendb/breath-classifier/csv/'

def convert(datafile):

    datas = pd.read_csv(dataset_dir + datafile, header=None, chunksize=4096)
    # res = None
    for data in datas:
        print set(data[15722])
    #     sys.exit(0)
    #     temp = data.iloc[:,:39].join(data.iloc[:,(39*199):(39*199+40)])
    #     for i in range(1, 199):
    #         temp = temp.join(data.iloc[:, 39*i:39*(i+1)])
    #         temp = temp.join(data.iloc[:,(39*199 + 40 * i):(39*199 + 40 * (i + 1))])
    #
    #     temp = temp.join(data.iloc[:,(79*199):])
    #
    #     if res is None:
    #         res = temp
    #     else:
    #         res = res.append(temp, ignore_index=True)
    #     print res.shape
    # res.to_csv('breath/' + datafile)


# convert('data_Feature_SILENT7.csv')

for datafile in os.listdir(dataset_dir):
    if os.path.isfile(os.path.join(dataset_dir, datafile)) and datafile.split('.')[1] == 'csv':
        print 'converting ' + datafile
        convert(datafile)