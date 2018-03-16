# encoding: utf-8

import os
import sys
import re
import time
import random
from collections import Counter

import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchwordemb
from get_w2v import Vocab

DATA_DIR = 'data/v1/'

print('Loading word2vec')
vocab = torch.load('./vocab.pt')
embeddings = nn.Embedding(vocab.num, vocab.dim).cuda()
embeddings.weight.data.copy_(vocab.i2v)
embeddings.weight.requires_grad = False
print('Loaded word2vec. Time cost:', time.clock())

total = 0
ban_title = 0

train_corpus = []
valid_corpus = []
for i, line in enumerate(open('pairs_norm.txt')):
    total += 1
    components = line.strip().split('\t')
    if len(components) != 10:
        continue

    preference = components[0]
    profile = components[1]
    u_profile = Variable(torch.FloatTensor(list(map(float,profile.split('\001')))).unsqueeze(0).cuda(),requires_grad=False)
    wordcounts = [wordcount.split('\002') for wordcount in preference.split('\001')]
    u_idxs, u_counts = zip(*[(vocab.w2i[word], float(count)) for word, count in wordcounts if word in vocab.w2i])
    u_idxs = Variable(torch.LongTensor([u_idxs]).cuda(), requires_grad=False)
    u_counts = Variable(torch.FloatTensor(u_counts).cuda(), requires_grad=False).view(-1, 1)
    u_embs = embeddings(u_idxs).squeeze()
    u_emb = torch.sum(u_embs*u_counts, dim=0, keepdim=True)
    u_emb = torch.nn.functional.normalize(u_emb, p=2, dim=1)

    titles = [title.split() for title in components[2:6]]
    t_idxs = [[vocab.w2i[word] for word in title if word in vocab.w2i] for title in titles]
    i_meta = [tuple(map(float,meta.split('\001'))) for meta in components[6:]]
    if [] in t_idxs:
        ban_title += 1
        continue
    corpus = (torch.cat((u_emb, u_profile),1),t_idxs,i_meta)
    if i % 10 == 0:
        valid_corpus.append(corpus)
    else:
        train_corpus.append(corpus)

    if (i+1) % 100 == 0:
        print('Generating corpus: %d' % i, end='\r')
print()
print('Generated Corpus. Time cost:', time.clock())
print(total, ban_title)

print('Saving Corpus')
torch.save(valid_corpus, DATA_DIR + 'valid_ex_norm.pt')
torch.save(train_corpus, DATA_DIR + 'train_ex_norm.pt')
print('Finished. Time cost:', time.clock())
