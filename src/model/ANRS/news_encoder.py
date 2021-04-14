import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import get_params
from src.model.ANRS.additive import AdditiveAttention

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NewsEncoder(torch.nn.Module):
    def __init__(self, config, pretrained_word_embedding, writer):
        super(NewsEncoder, self).__init__()
        self.config = config
        self.window_size = vars(get_params())['window_size']
        self.dropout_rate = vars(get_params())['dropout_rate']

        if pretrained_word_embedding is None:
            self.word_embedding = nn.Embedding(config.num_words,
                                               config.word_embedding_dim,
                                               padding_idx=0)
        else:
            self.word_embedding = nn.Embedding.from_pretrained(
                pretrained_word_embedding, freeze=False, padding_idx=0)
        self.category_embedding = nn.Embedding(config.num_categories,
                                               config.category_embedding_dim,
                                               padding_idx=0)
        self.category_linear = nn.Linear(config.category_embedding_dim,
                                         config.num_filters)
        self.subcategory_linear = nn.Linear(config.category_embedding_dim,
                                            config.num_filters)

        assert self.window_size >= 1 and self.window_size % 2 == 1

        self.title_CNN = nn.Conv2d(
            1,
            config.num_filters,
            (self.window_size, config.word_embedding_dim),
            padding=(int((self.window_size - 1) / 2), 0))
        self.title_attention = AdditiveAttention(config.query_vector_dim,
                                                 config.num_filters)
        self.abstract_CNN = nn.Conv2d(
            1,
            config.num_filters,
            (self.window_size, config.word_embedding_dim),
            padding=(int((self.window_size - 1) / 2), 0))
        self.abstract_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.num_filters)
        self.final_attention = AdditiveAttention(config.query_vector_dim,
                                                 config.num_filters, writer,
                                                 'Train/NewsAttentionWeight',
                                                 ['category', 'subcategory',
                                                  'title', 'abstract'])

    def forward(self, news):

        category_vector = self.category_embedding(news['category'].to(device))

        activated_category_vector = F.relu(
            self.category_linear(category_vector))

        subcategory_vector = self.category_embedding(
            news['subcategory'].to(device))

        activated_subcategory_vector = F.relu(
            self.subcategory_linear(subcategory_vector))

        title_vector = F.dropout(self.word_embedding(
            torch.stack(news['title'], dim=1).to(device)),
            p=self.dropout_rate,
            training=self.training)

        convoluted_title_vector = self.title_CNN(
            title_vector.unsqueeze(dim=1)).squeeze(dim=3)

        activated_title_vector = F.dropout(F.relu(convoluted_title_vector),
                                           p=self.dropout_rate,
                                           training=self.training)

        weighted_title_vector = self.title_attention(
            activated_title_vector.transpose(1, 2))

        abstract_vector = F.dropout(self.word_embedding(
            torch.stack(news['abstract'], dim=1).to(device)),
            p=self.dropout_rate,
            training=self.training)

        convoluted_abstract_vector = self.abstract_CNN(
            abstract_vector.unsqueeze(dim=1)).squeeze(dim=3)

        activated_abstract_vector = F.dropout(
            F.relu(convoluted_abstract_vector),
            p=self.dropout_rate,
            training=self.training)

        weighted_abstract_vector = self.abstract_attention(
            activated_abstract_vector.transpose(1, 2))

        stacked_news_vector = torch.stack([
            activated_category_vector, activated_subcategory_vector,
            weighted_title_vector, weighted_abstract_vector
        ],
            dim=1)

        news_vector = self.final_attention(stacked_news_vector)

        return news_vector
