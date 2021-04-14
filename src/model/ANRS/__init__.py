import torch

from src.model.ANRS.AEmodel import abae, max_margin_loss, orthogonal_regularization
from src.model.ANRS.news_encoder import NewsEncoder
from src.model.ANRS.user_encoder import UserEncoder
from src.model.ANRS.dot_product import DotProductClickPredictor
from src.model.ANRS.w2v import word2vec

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ANRS(torch.nn.Module):

    def __init__(self, config, pretrained_word_embedding=None, writer=None):
        super(ANRS, self).__init__()
        self.config = config
        self.news_encoder = NewsEncoder(config, pretrained_word_embedding, writer)
        self.user_encoder = UserEncoder(config)
        self.click_predictor = DotProductClickPredictor()

        self.w2v = word2vec(config.source)
        self.w2v.embed(config.source, config.w2v_path)
        self.w2v.aspect(config.n_aspects)
        self.aspect_encoder = abae(config, self.w2v.E, self.w2v.T)


    def forward(self, candidate_news, clicked_news):
        candidate_news_vector = torch.stack(
            [self.news_encoder(x) for x in candidate_news])
        clicked_news_vector = torch.stack(
            [self.news_encoder(x) for x in clicked_news], dim=1)
        rs, zs, zn = self.aspect_encoder(clicked_news, candidate_news)
        stacked_clicked_news_vector = torch.cat([
            clicked_news_vector, rs.view(-1, 50, 300)
        ], dim=0)
        crs, czs, czn = self.aspect_encoder(candidate_news, clicked_news)
        stacked_candidate_news_vector = torch.cat([
            candidate_news_vector, crs.view(5, -1, 300)
        ], dim=1)
        user_vector = self.user_encoder(stacked_clicked_news_vector)
        click_probability = torch.stack([
            self.click_predictor(x, user_vector) for x in stacked_candidate_news_vector
        ], dim=1)
        J = max_margin_loss(rs, zs, zn)
        F = orthogonal_regularization(self.aspect_encoder.T.weight)
        loss = J + self.config.ortho_reg * self.config.batch_size * F

        return click_probability, loss

    def get_news_vector(self, news):

        return self.news_encoder(news)

    def get_user_vector(self, clicked_news_vector):
        return self.user_encoder(clicked_news_vector)

    def get_prediction(self, news_vector, user_vector):
        click_probability = self.click_predictor(
            news_vector.unsqueeze(dim=0),
            user_vector.unsqueeze(dim=0)).squeeze(dim=0)
        return click_probability
