import torch

class DotProductClickPredictor(torch.nn.Module):
    def __init__(self):
        super(DotProductClickPredictor, self).__init__()

    def forward(self, candidate_news_vector, user_vector):
        probability = torch.bmm(
            user_vector.unsqueeze(dim=1),
            candidate_news_vector.unsqueeze(dim=2)).flatten()
        return probability
