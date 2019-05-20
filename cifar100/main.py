import torch
from config import config 
from tqdm import tqdm
import numpy as np
import os
from utils import Network

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print(torch.cuda.is_available(), torch.cuda.device_count())
    model = Network()
    print('make network complete')
    optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.99, weight_decay=config.weight_decay)
    loss = torch.nn.CrossEntropyLoss()

    from data import Cifar100  
    data = Cifar100()
    print('load date complete')

    epoch = 0
    best = 0
    for epoch in tqdm(range(config.epoch)):
        with tqdm(range(data.size // config.batch_size), postfix=0) as t:
            for minibatch in t:
                optimizer.zero_grad()
                train_data, train_labels, train_cls = data.gen(config.batch_size)
                pred_coarse, pred_fine = model(torch.autograd.Variable(torch.from_numpy(train_data)).float().cuda())
                A = loss(pred_coarse, torch.autograd.Variable(torch.from_numpy(train_cls)).cuda())
                B = loss(pred_fine, torch.autograd.Variable(torch.from_numpy(train_labels)).cuda())
                H = A + B
                H.backward()
                optimizer.step()
                t.postfix = '%s %s' % (round(float(A.data.cpu().numpy()), 5), round(float(B.data.cpu().numpy()), 5))
                t.update()


if __name__ == '__main__':
    main()

