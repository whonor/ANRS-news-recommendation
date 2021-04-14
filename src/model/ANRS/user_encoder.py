import torch

from src.model.ANRS.additive import AdditiveAttention

class UserEncoder(torch.nn.Module):
    def __init__(self, config):
        super(UserEncoder, self).__init__()
        self.additive_attention = AdditiveAttention(config.query_vector_dim,
                                                    config.num_filters)

    def forward(self, clicked_news_vector):
        user_vector = self.additive_attention(clicked_news_vector)
        return user_vector
