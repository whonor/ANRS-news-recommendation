import csv
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import codecs
from tqdm import tqdm
import os

class word2vec:

    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.n_vocab = 0

    def __iter__(self):
        with codecs.open(self.corpus_path, 'r', 'utf-8') as f:
            for line in tqdm.tqdm(f, desc='training'):
                yield line.split()

    def add_word(self, *words):
        for word in words:
            if not word in self.w2i:
                self.w2i[word] = self.n_vocab
                self.i2w[self.w2i[word]] = word
                self.n_vocab += 1

    def embed(self, source, word2int_path, d_embed =300):
        self.d_embed = d_embed
        word2int = dict(
            pd.read_table(word2int_path, na_filter=False).values.tolist())
        source_embedding = pd.read_table(source,
                                         index_col=0,
                                         sep=' ',
                                         header=None,
                                         quoting=csv.QUOTE_NONE)
        self.E = np.random.normal(size=(1 + len(word2int), d_embed))
        self.E[0] = 0
        word_missed = 0

        if os.access("GE.npy", mode=os.R_OK):
            print("Load word embedding...")
            self.E = np.load("GE.npy")
        else:
            with tqdm(total=len(word2int), desc="Generate Embedding E") as pbar:
                for k, v in word2int.items():
                    if k in source_embedding.index:
                        self.E[v] = source_embedding.loc[k].tolist()
                    else:
                        word_missed += 1

                    pbar.update(1)

            self.E = np.asarray(self.E).astype(np.float32)
            with open("GE.npy", 'wb') as f:
                np.save(f, self.E)

        return self

    def aspect(self, n_aspects):
        self.n_aspects = n_aspects
        km = KMeans(n_clusters=n_aspects, random_state=0)
        km.fit(self.E)
        self.T = km.cluster_centers_.astype(np.float32)
        self.T /= np.linalg.norm(self.T, axis=-1, keepdims=True)
        return self


