import os
import sys
import time
import datetime
import random
import numpy
from itertools import islice

import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from model import MyModule

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

DATA_DIR = 'data/v1/'
model_file = sys.argv[1]
print('Preparing model')
model = torch.load(model_file)
loss_function = nn.CrossEntropyLoss()

print('Loading corpus')
corpus = torch.load(DATA_DIR + 'valid.pt')
print('Loaded corpus. Time cost:', time.clock())
batch_size = 20

i = 0
total = 0
correct = 0
total_loss = 0.
for batch_data in batch(corpus, batch_size):
    # print(' '.join([idx2w[x] for x in item_title_idxs]))
    i += 1
    user_vecs, title_vecs = list(zip(*batch_data))
    user_vecs = torch.stack(user_vecs, dim=0)
    title_vecs = torch.stack(title_vecs, dim=0)

    out = model(user_vecs, title_vecs)
    target_idx = Variable(torch.LongTensor([0] * len(batch_data)).cuda())
    loss = loss_function(out, target_idx)
    total_loss += float(loss.data[0])

    out = out.data.cpu().numpy().tolist()
    result = [int(numpy.argmax(scores)==0) for scores in out]
    total += len(result)
    correct += sum(result)


print(total, correct, 1.*correct/total)
print(total_loss/i)
