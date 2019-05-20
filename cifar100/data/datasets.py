import numpy as np
import pickle

class Cifar100:
    def __init__(self):
        raw = pickle.load(open('/home/liuyu/Documents/code/data/cifar-100-python/train','rb'), encoding='iso-8859-1')
        self.size = 50000
        self.data = raw['data'].reshape(self.size, 3, 32, 32)
        self.fine_labels = np.array(raw['fine_labels'])
        self.coarse_labels = np.array(raw['coarse_labels'])
        print('done')
    
        raw = pickle.load(open('/home/liuyu/Documents/code/data/cifar-100-python/test','rb'), encoding='iso-8859-1')
        self.test_size = 10000
        self.test_data = raw['data'].reshape(self.test_size, 3, 32, 32)
        self.test_labels = np.array(raw['fine_labels'])
        self.test_cls = np.array(raw['coarse_labels'])
        self.test_start = 0
        print('done')

    def gen(self, batch):
        idx = np.random.randint(0, self.size, (batch, ))
        data = self.data[idx]
        labels = self.fine_labels[idx]
        cls = self.coarse_labels[idx]
        return (data, labels, cls)

    def test(self, batch):
        start = self.test_start
        end = min(self.test_start + batch, self.test_size)
        self.test_start = end
        data = self.test_data[start:end]
        labels = self.test_labels[start:end]
        cls = self.test_cls[start:end]
        return (data, labels, cls)


if __name__ == '__main__':
    pass
