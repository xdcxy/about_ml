import os
import sys
import time
import datetime
import random
import math
from itertools import islice

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import MyModule

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

DATA_DIR = 'data/v1/'
print('Preparing model')
model = MyModule(128)
loss_function = nn.CrossEntropyLoss()
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = optim.SGD(parameters, lr=0.5)

print('Loading corpus')
corpus = torch.load(DATA_DIR + 'train.pt')
print('Loaded corpus. Time cost:', time.clock())
batch_size = 5
SHOW_NUM = 5000

print('Start training')
for epoch in range(50):
    # with open(train_data) as f:
    i = 0
    total_loss = 0.
    random.shuffle(corpus)
    for batch_data in batch(corpus, batch_size):
        # print(' '.join([idx2w[x] for x in item_title_idxs]))
        model.zero_grad()
        user_vecs, title_vecs = list(zip(*batch_data))
        user_vecs = torch.stack(user_vecs, dim=0)
        title_vecs = torch.stack(title_vecs, dim=0)

        out = model(user_vecs, title_vecs)
        target_idx = Variable(torch.LongTensor([0] * len(batch_data)).cuda())
        # print(out.size())
        # print(target_idx.size())
        # quit()
        loss = loss_function(out, target_idx)
        loss.backward()
        optimizer.step()
        # quit()
        # output.write('%.3f %.3f\n' % (log_probs.data[0,0], log_probs.data[0,1]))
        total_loss += float(loss.data[0])

        i += 1
        if i % SHOW_NUM == 0:
            print(datetime.datetime.now().time(), 'Training epoch %d iter %d  loss: %.6f  ppl: %.5f' % (epoch, i, total_loss/SHOW_NUM, math.exp(total_loss/SHOW_NUM)))
            total_loss = 0.
        # sys.stderr.write('training %d sample' % i)
    print('Start saving model')
    torch.save(model, DATA_DIR + "model/2-layer-step%d.pth" % epoch)
# output.close()
