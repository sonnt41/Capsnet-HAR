import pandas
import numpy as np


path = '/Users/vietld/PycharmProjects/har/data/001/in/data.csv'
with open(path, 'r') as f:
    lines = f.readlines()
data=[]
for line in lines:
    parts = line.strip().split(',')
    _d = (int(parts[0]),[float(x) for x in parts[1:-1]], parts[-1])
    data.append(_d)

print('Number of samples: ')
len(data)


print(data[0])

label = 'upstair'
tp = [x[0] for x in data if x[2]==label]
val = [x[1][0] for x in data if x[2]==label]

import matplotlib.pyplot as plt

fig = plt.plot(tp, val)
fig.show()