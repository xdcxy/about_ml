
import numpy as np
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MyModule(nn.Module):
    def __init__(self, embedding_dim):
        super(MyModule, self).__init__()
        # self.linear1 = nn.Linear(500, 500).cuda()
        # self.linear2 = nn.Linear(500, 1).cuda()
        self.u1 = nn.Linear(embedding_dim, 500).cuda()
        self.u2 = nn.Linear(500, 500).cuda()
        self.t1 = nn.Linear(embedding_dim, 500).cuda()
        self.t2 = nn.Linear(500, 500).cuda()

    def forward(self, user_vecs, title_vecs):
        user = F.tanh(self.u2(F.tanh(self.u1(user_vecs))))
        title = F.tanh(self.t2(F.tanh(self.t1(title_vecs))))
        # print(user.size())
        # print(title.size())
        embeds = user * title
        # print(embeds.size())
        out = torch.sum(embeds, dim=-1) #.squeeze(-1)
        # print(out.size())
        # quit()

        return out
