from itertools import islice, zip_longest, groupby, accumulate
import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SentEncoder(nn.Module):
    def __init__(self, vocab, hidden_dim=500):
        super(SentEncoder, self).__init__()
        self.lstm = nn.LSTM(vocab.dim, hidden_dim, num_layers=2,bidirectional=True).cuda()
        self.vocab = vocab

        # 词语Embedding数据
        self.embeddings = nn.Embedding(vocab.num, vocab.dim, padding_idx=vocab.padding_idx).cuda()
        # 这里设置为使用word2vec结果作为pretrain输入，可配置训练中是否更新
        self.embeddings.weight.data.copy_(vocab.i2v)
        self.embeddings.weight.requires_grad = True

    def _sort_and_pad(self, seq):
        seq_info = [(s, len(s), i) for i, s in enumerate(seq)]
        sorted_seq, sorted_len, reverse_map = zip(*sorted(seq_info, key=lambda x:x[1], reverse=True))
        _, sorted_map = list(zip(*sorted([(x, i) for i, x in enumerate(reverse_map)], key=lambda x:x[0])))
        sorted_seq = list(zip(*zip_longest(*sorted_seq, fillvalue=self.vocab.padding_idx)))
        sorted_seq = Variable(torch.LongTensor(sorted_seq).cuda())
        return sorted_seq, sorted_len, sorted_map

    @staticmethod
    def _recover(sorted_seq, sorted_map):
        return sorted_seq[:, sorted_map, :]

    def forward(self, seq):
        sorted_seq, sorted_len, sorted_map = self._sort_and_pad(seq)
        # print(sorted_seq)
        # print(sorted_len)
        sorted_seq_emb = self.embeddings(sorted_seq) # sorted_seq_emb: batch * max_src_length * emb_dim
        # 这里使用了pytorch0.2版本新增的packed_sequence功能，可在encoder中处理变长序列
        packed_sequence = torch.nn.utils.rnn.pack_padded_sequence(sorted_seq_emb,
                sorted_len,
                batch_first=True)
        packed_lstm_out, (ht, ct) = self.lstm(packed_sequence) # ht/ct : layer_num * batch * hidden_dim
        context_vec, context_len = torch.nn.utils.rnn.pad_packed_sequence(packed_lstm_out) # context_vec: max_src_length * batch * hidden_dim

        context_vec = self._recover(context_vec, sorted_map)
        ht = self._recover(ht, sorted_map)
        ct = self._recover(ct, sorted_map)

        return context_vec, (ht, ct)

class MyModel(nn.Module):
    def __init__(self, vocab,ex_user_dim=0,ex_item_dim=0, hidden_dim=500):
        super(MyModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = SentEncoder(vocab, hidden_dim)
        self.u1 = nn.Linear(vocab.dim + ex_user_dim, hidden_dim).cuda()
        self.u2 = nn.Linear(hidden_dim, hidden_dim).cuda()
        self.t1 = nn.Linear(hidden_dim + ex_item_dim, hidden_dim).cuda()
        self.t2 = nn.Linear(hidden_dim, hidden_dim).cuda()

    # 训练过程
    def forward(self, user_vecs, title_idxs,item_ex_feas):
        batch_size = user_vecs.size(0)
        context, hidden = self.encoder(title_idxs)
        ht = torch.cat((hidden[0][-1],Variable(item_ex_feas.cuda())),1)

        user = F.tanh(self.u2(F.tanh(self.u1(user_vecs))))
        title = F.tanh(self.t2(F.tanh(self.t1(ht)))).view(batch_size, -1, self.hidden_dim)
        #user = F.relu(self.u2(F.relu(self.u1(user_vecs))))
        #title = F.relu(self.t2(F.relu(self.t1(ht)))).view(batch_size, -1, self.hidden_dim)
        embeds = user * title
        out = torch.sum(embeds, dim=-1)

        return out

    def user_inference(self,user_vecs):
        user = F.tanh(self.u2(F.tanh(self.u1(user_vecs)))).data.cpu().numpy()
        return user


    def item_inference(self,item_fea,batch_size):
        context, hidden = self.encoder(item_fea)
        ht = hidden[0][-1]
        #item_vec = F.tanh(self.t2(F.tanh(self.t1(ht)))).view(batch_size, -1, self.hidden_dim).data.cpu().numpy()
        item_vec = F.tanh(self.t2(F.tanh(self.t1(ht)))).data.cpu().numpy()
        return item_vec

