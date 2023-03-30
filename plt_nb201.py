import collections

from datasets.nb201_dataset import NasBench201Dataset
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = NasBench201Dataset(start=0, end=15624, hp='200', seed=777)
    d = {}
    for i in data:
        a = int(i.y[1][-1])
        if a <= 80 :
            continue
        if not d.get(a):
            d[a] = 1
        else:
            d[a] += 1

    od = collections.OrderedDict(sorted(d.items()))
    print(od)
    print(od.keys())
    print(od.values())
    plt.bar(od.keys(), od.values())
    plt.show()