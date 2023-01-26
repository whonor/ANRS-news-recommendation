# ANRS-news-recommendation
Pytorch code for paper ''Aspect-driven User Preference and News Representation Learning for News Recommendation''


## Requirement
- pytorch~=1.5.0
- numpy~=1.19.2
- pandas~=1.1.3
- tensorboard
- tqdm~=4.46.0
- nltk~=3.5
- scikit-learn~=0.23.2

## Dataset
```bash
# Download GloVe pre-trained word embedding
https://nlp.stanford.edu/data/glove.840B.300d.zip

# Download MIND dataset
https://msnews.github.io/.
```
## Run
```bash
# Preprocess data into appropriate format
python3 src/data_preprocess_large.py
# Train and save checkpoint into `checkpoint/{model_name}/` directory
python3 src/train1.py
# Load latest checkpoint and evaluate on the test set
python3 src/evaluate.py
```

## Acknowledgement
Any scientific publications that use our codes and datasets should cite the following paper as the reference:
```bash
@article{lu2022aspect,
  title={Aspect-driven user preference and news representation learning for news recommendation},
  author={Lu, Wenpeng and Wang, Rongyao and Wang, Shoujin and Peng, Xueping and Wu, Hao and Zhang, Qian},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={23},
  number={12},
  pages={25297--25307},
  year={2022},
  publisher={IEEE}
}
```

### Credits

- Dataset by **MI**crosoft **N**ews **D**ataset (MIND), see <https://msnews.github.io/>.
