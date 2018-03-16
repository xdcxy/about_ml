import os
import sys
import time
import datetime
import random
import math
from itertools import islice

import numpy
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import MyModel

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

DATA_DIR = 'data/v1/'

print('Loading word2vec')
vocab = torch.load('./vocab.pt')
print('Loaded word2vec. Time cost:', time.clock())

print('Loading corpus')
corpus = torch.load(DATA_DIR + 'train_ex_norm.pt')
print('Loaded corpus. Time cost:', time.clock())
batch_size = 128
SHOW_NUM = 1000

print('Loading valid corpus')
valid_corpus = torch.load(DATA_DIR + 'valid_ex_norm.pt')
valid_batch_size = 128
print('Loaded valid corpus. Time cost:', time.clock())

print('Preparing model')
ex_user_dim = 2
ex_item_dim = 3

model = MyModel(vocab,ex_user_dim,ex_item_dim,128)
loss_function = nn.CrossEntropyLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
learning_rate = 0.5
optimizer = optim.SGD(parameters, lr=learning_rate)
print(model)

print('Start training')
for epoch in range(50):
    # TRAIN
    i = 0
    total_loss = 0.
    random.shuffle(corpus)
    for batch_data in batch(corpus, batch_size):
        model.zero_grad()
        user_vecs, title_idxs,i_meta = list(zip(*batch_data))
        user_vecs = torch.stack(user_vecs, dim=0)
        flat_title_idxs = [idx for title_idx in title_idxs for idx in title_idx]
        item_meta = torch.FloatTensor(i_meta).view(-1,ex_item_dim)

        out = model(user_vecs, flat_title_idxs,item_meta)
        target_idx = Variable(torch.LongTensor([0] * len(batch_data)).cuda())
        loss = loss_function(out, target_idx)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.data[0])

        i += 1
        if i % SHOW_NUM == 0:
            print(datetime.datetime.now().time(), 'Training epoch %d iter %d  loss: %.6f  ppl: %.5f' % (epoch, i, total_loss/SHOW_NUM, math.exp(total_loss/SHOW_NUM)))
            total_loss = 0.
    learning_rate *= 0.9
    optimizer.lr = learning_rate
    print('Start saving model')
    torch.save(model, DATA_DIR + "model/ex-fea-norm-step%d.pth" % epoch)


    # VALID
    total = 0
    correct = 0
    for batch_data in batch(valid_corpus, valid_batch_size):
        i += 1
        user_vecs, title_idxs,i_meta = list(zip(*batch_data))
        user_vecs = torch.stack(user_vecs, dim=0)
        flat_title_idxs = [idx for title_idx in title_idxs for idx in title_idx]
        item_meta = torch.FloatTensor(i_meta).view(-1,ex_item_dim)

        out = model(user_vecs, flat_title_idxs,item_meta)
        out = out.data.cpu().numpy().tolist()
        result = [int(numpy.argmax(scores)==0) for scores in out]
        total += len(result)
        correct += sum(result)
    print(datetime.datetime.now().time(), 'Valid epoch: %d acc: %.4f' % (epoch, 1.*correct/total))
    print('-'*50)
