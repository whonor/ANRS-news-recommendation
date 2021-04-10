# ANRS-news-recommendation
Pytorch code for paper ''Aspect-driven User Preference and News Representation Learning for News Recommendation''

# Requirement
- pytorch~=1.5.0
- numpy~=1.19.2
- pandas~=1.1.3
- tensorboard
- tqdm~=4.46.0
- nltk~=3.5
- scikit-learn~=0.23.2

# Dataset
```bash
# Download GloVe pre-trained word embedding
https://nlp.stanford.edu/data/glove.840B.300d.zip

# By downloading the dataset, you agree to the [Microsoft Research License Terms](https://go.microsoft.com/fwlink/?LinkID=206977). For more detail about the dataset, see https://msnews.github.io/.
```
# Run
```bash
# Preprocess data into appropriate format
python3 src/data_preprocess_large.py
# Train and save checkpoint into `checkpoint/{model_name}/` directory
python3 src/train1.py
# Load latest checkpoint and evaluate on the test set
python3 src/evaluate.py
```

## Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
