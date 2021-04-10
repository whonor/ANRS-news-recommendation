'''
Note: barch_size of textCNN must be 16 or you can modify yourself..
'''
import argparse
import os

model_name = os.environ['MODEL_NAME'] if 'MODEL_NAME' in os.environ else 'ANRS'

assert model_name in ['ANRS']

def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dropout_rate", type=float, default=0.2, help="dropout rate")
    parser.add_argument("--window_size", type=int, default=7)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=4)
    args, _ = parser.parse_known_args()
    return args


class BaseConfig():

    accumulation_steps = 2

    num_batches = 80000  # Number of batches to train 60000
    num_batches_show_loss = 100  # Number of batchs to show loss
    # Number of batchs to check metrics on validation dataset
    num_batches_validate = 800
    batch_size = 16  # textCNN:16
    learning_rate = 0.0001
    validation_proportion = 0.1
    num_workers = 0  # Number of workers for data loading
    num_clicked_news_a_user = 50  # Number of sampled click history for each user
    # Whether try to load checkpoint
    load_checkpoint = 0 # os.environ['LOAD_CHECKPOINT'] == '1' if 'LOAD_CHECKPOINT' in os.environ else True

    num_words_title = 20
    num_words_abstract = 50
    word_freq_threshold = 3
    entity_freq_threshold = 3
    entity_confidence_threshold = 0.5
    negative_sampling_ratio = 4  # K
    dropout_probability = 0.2
    # Modify the following by the output of `data_preprocess.py`
    num_words = 1 + 44774
    num_categories = 1 + 295
    num_entities = 1 + 14697
    num_users = 1 + 711222
    word_embedding_dim = 300
    category_embedding_dim = 100
    # Modify the following only if you use another dataset
    entity_embedding_dim = 100
    # For additive attention
    query_vector_dim = 200

class ANRSConfig(BaseConfig):
    dataset_attributes = {"news": ['category', 'subcategory', 'title', 'abstract'], "record": []}
    source = "data/glove.840B.300d.txt"
    w2v_path = "data/train_l/word2int.tsv"
    # For CNN
    num_filters = 300
    window_size = 5
    aspect_classification_loss_weight = 0.1

    n_aspects = 40
    negsize = 20
    ortho_reg = 0.1
