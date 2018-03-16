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

DATA_DIR = 'data/v1/'

print('Loading word2vec')
vocab, w2v = torchwordemb.load_word2vec_text('/home/sakya.cxy/wkp/gitlab/item_rank_model/model/tensorflow/data/word_vec.txt')
vocab_size = w2v.size(0)
embedding_dim = w2v.size(1)
embeddings = nn.Embedding(vocab_size, embedding_dim).cuda()
embeddings.weight.data.copy_(w2v)
embeddings.weight.requires_grad = False
print('Loaded word2vec. Time cost:', time.clock())

total = 0
ban_title = 0

train_corpus = []
valid_corpus = []
for i, line in enumerate(open('./pairs.txt')):
    total += 1
    components = line.strip().split('\t')

    preference = components[0]
    wordcounts = [wordcount.split('\002') for wordcount in preference.split('\001')]
    u_idxs, u_counts = zip(*[(vocab[word], float(count)) for word, count in wordcounts if word in vocab])
    u_idxs = Variable(torch.LongTensor([u_idxs]).cuda(), requires_grad=False)
    u_counts = Variable(torch.FloatTensor(u_counts).cuda(), requires_grad=False).view(-1, 1)
    u_embs = embeddings(u_idxs).squeeze()
    u_emb = torch.sum(u_embs*u_counts, dim=0, keepdim=True)
    u_emb = torch.nn.functional.normalize(u_emb, p=2, dim=1)

    titles = [title.split() for title in components[1:]]
    t_idxs = [[vocab[word] for word in title if word in vocab] for title in titles]
    if [] in t_idxs:
        ban_title += 1
        continue

    t_idxs = [Variable(torch.LongTensor(t_idx).cuda(), requires_grad=False) for t_idx in t_idxs]
    t_embs = [embeddings(t_idx).squeeze() for t_idx in t_idxs]
    t_embs = [torch.sum(emb, dim=0) for emb in t_embs]
    try:
        t_embs = torch.stack(t_embs, dim=0)
    except:
        ban_title += 1
        continue
    t_emb = torch.nn.functional.normalize(t_embs, p=2, dim=1).squeeze()
    corpus = (u_emb, t_emb)
    if i % 10 == 0:
        valid_corpus.append(corpus)
    else:
        train_corpus.append(corpus)

    if (i+1) % 100 == 0:
        print('Generating corpus: %d' % i, end='\r')
        # break
print()
print('Generated Corpus. Time cost:', time.clock())
print(total, ban_title)

print('Saving Corpus')
torch.save(valid_corpus, DATA_DIR + 'valid.pt')
torch.save(train_corpus, DATA_DIR + 'train.pt')
print('Finished. Time cost:', time.clock())
