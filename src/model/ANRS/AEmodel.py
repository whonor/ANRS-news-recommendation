import torch.nn as nn
import torch
from torch.nn.functional import normalize, softmax

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def max_margin_loss(r_s, z_s, z_n):
    pos = torch.bmm(z_s.unsqueeze(1), r_s.unsqueeze(2)).squeeze(2)
    negs = torch.mm(z_n, r_s.t()).squeeze(1)
    J = torch.ones(negs.shape).to(device) - pos.expand(negs.t().shape).t() + negs
    return torch.sum(torch.clamp(J, min=0.0))


def orthogonal_regularization(T):
    T_n = normalize(T, dim=1)
    I = torch.eye(T_n.shape[0]).to(device)
    return torch.norm(T_n.mm(T_n.t()) - I)

class attention(nn.Module):

    def __init__(self, d_embed):
        super(attention, self).__init__()
        self.M = nn.Linear(d_embed, d_embed)
        self.M.weight.data.uniform_(-0.1, 0.1)

    def forward(self, e_i):
        y_s = torch.mean(e_i, dim=-1)
        d_i = torch.bmm(e_i.transpose(1, 2), self.M(y_s).unsqueeze(2)).tanh()
        a_i = torch.exp(d_i) / torch.sum(torch.exp(d_i))
        return a_i.squeeze(1)


class abae(nn.Module):

    def __init__(self, config, E, T):
        super(abae, self).__init__()
        self.config = config
        config.num_words, config.word_embedding_dim = E.shape
        config.n_aspects, config.word_embedding_dim = T.shape
        if E is None:
            self.E = nn.Embedding(config.num_words, config.word_embedding_dim)
        else:
            self.E = nn.Embedding.from_pretrained(
                torch.from_numpy(E), freeze=False, padding_idx=0)
        assert config.window_size >= 1 and config.window_size % 2 == 1
        self.T = nn.Embedding(config.n_aspects, config.word_embedding_dim)
        self.attention = attention(config.word_embedding_dim)
        self.Linear = nn.Linear(config.word_embedding_dim, config.n_aspects)
        self.E.weight = nn.Parameter(torch.from_numpy(E))
        self.T.weight = nn.Parameter(torch.from_numpy(T))

    def forward(self, pos, negs):
        cate_tensor = []
        content_tensor = []
        for i in negs:
            for s in i.values():
                if isinstance(s, list):
                    for j in s:
                        content_tensor.append(j)
                else:
                    cate_tensor.append(s)
        news = cate_tensor + content_tensor
        n = torch.stack(news, dim=1)

        p_t, z_s = self.predict(pos)
        r_s = normalize(torch.mm(self.T.weight.t(), p_t.t()).t(), dim=-1)
        e_n = self.E(n.to(device)).transpose(-2, -1).to(device)
        z_n = normalize(torch.mean(e_n, dim=-1), dim=-1).to(device)

        return r_s, z_s, z_n

    def predict(self, x):

        cate_tensor = []
        content_tensor = []
        for i in x:
            for s in i.values():
                if isinstance(s, list):
                    for j in s:
                        content_tensor.append(j)
                else:
                    cate_tensor.append(s)
        news = cate_tensor + content_tensor
        n = torch.stack(news, dim=0)
        e_i = self.E(n.to(device)).transpose(1, 2).to(device)
        a_i = self.attention(e_i)
        z_s = normalize(torch.bmm(e_i, a_i).squeeze(2), dim=-1).to(device)
        p_t = softmax(self.Linear(z_s), dim=1).to(device)

        return p_t, z_s

    def aspects(self):
        E_n = normalize(self.E.weight, dim=1)
        T_n = normalize(self.T.weight, dim=1)
        projection = torch.mm(E_n, T_n.t()).t()
        return projection


