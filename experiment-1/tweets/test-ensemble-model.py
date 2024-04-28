import os
import pandas as pd
import nltk
from sklearn.metrics import classification_report, confusion_matrix
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from bs4 import BeautifulSoup as beauty
import random
from nltk.corpus import stopwords
import re
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import scikitplot
import matplotlib.pyplot as plt

stop_words = set(stopwords.words('english'))
SEED = 42
random.seed(SEED)

TESTING_DATASETS = {
    "large_bilstm" : 'experiment-1/tweets/datasets/big_tweets_bilstm.csv',
    "large_ann":'experiment-1/tweets/datasets/big_tweets_ANN.csv',
    "small_bilstm" : 'experiment-1/tweets/datasets/small_tweets_bilstm.csv',
    "small_ann" : 'experiment-1/tweets/datasets/small_tweets_ANN.csv'
}

def read_data(datasets_dict):
    dfs = {}
    for key, file_path in datasets_dict.items():
        if "bilstm" in key:
            # For BiLSTM files
            with open(file_path, 'r') as file:
                lines = file.readlines()
                data = [{"sentiment": line.split(",")[0], "text": [word.strip() for word in line.strip().split(",")[1:]]} for line in lines]
                df = pd.DataFrame(data)
        elif "ann" in key:
            # For ANN files
            with open(file_path, 'r') as file:
                lines = file.readlines()
                data = [{"hypernyms": [word.strip() for word in line.strip().split(",")]} for line in lines]
                df = pd.DataFrame(data)
        else:
            raise ValueError("Invalid key in datasets_dict. Must contain 'bilstm' or 'ann'.")
        dfs[key] = df
    return dfs

def provide_embeddings_tfidf(texts):
    embeddings = Word2Vec(size=200, min_count=1)
    embeddings.build_vocab(texts)
    embeddings.train(texts,
                    total_examples=embeddings.corpus_count,
                    epochs=embeddings.epochs)
    print('word embedding model train done.')

    flattened_texts = [word for sentence in texts for word in sentence]
    gen_tfidf = TfidfVectorizer()
    gen_tfidf.fit_transform(flattened_texts)
    tfidf_map = dict(zip(gen_tfidf.get_feature_names_out(), gen_tfidf.idf_))

    return embeddings, tfidf_map


def encode_text_hypernyms(tfidf_map, word_embedding_model, text_hypernyms):
    EMBEDDING_SIZE = 200
    vector = np.zeros((1, EMBEDDING_SIZE))
    length = 0
    for word in text_hypernyms:
        try:
            vector += word_embedding_model.wv[word].reshape((1, EMBEDDING_SIZE)) * tfidf_map[word]
            length += 1
        except KeyError:
            continue
        break

    if length > 0:
        vector /= length
    return vector


def print_dataframes(dataframes):
    for key, df in dataframes.items():
        print(f"DataFrame: {key}")
        print(df)
        print("\n")


def get_testing_data(df_dict):
    big_bilstm_df = pd.concat([df_dict["large_bilstm"], df_dict["small_bilstm"]], ignore_index=True)
    big_ann_df = pd.concat([df_dict["large_ann"], df_dict["small_ann"]], ignore_index=True)
    #print(big_bilstm_df.head(5))
    #print(big_ann_df.head(5))
    result_of_filtering_tags = big_bilstm_df["text"].tolist()
    text_hypernyms = big_ann_df["hypernyms"].tolist()
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(result_of_filtering_tags)
    wv_model, tf_idf_map = provide_embeddings_tfidf(result_of_filtering_tags)

    texts = tokenizer.texts_to_sequences(result_of_filtering_tags)
    
    x_text_bilstm = pad_sequences(texts, maxlen=80, padding='post')
    
    x_test_ann = []
    for sentence in text_hypernyms:
        encoded_embeddings = encode_text_hypernyms(tfidf_map=tf_idf_map, word_embedding_model=wv_model,
                                                 text_hypernyms=sentence)
        x_test_ann.append(encoded_embeddings)
    
    x_test_ann = np.asarray(x_test_ann).reshape(len(big_ann_df), 200)
   
    y_true = big_bilstm_df['sentiment'].astype(int).to_numpy().reshape(-1, 1)

    for i in range(0,len(y_true)):
        if int(y_true[i])<=4:
            y_true[i] = 0
        else:
            y_true[i] = 1
    
    return x_text_bilstm, x_test_ann, y_true


if __name__ == '__main__':
    df_dict = read_data(TESTING_DATASETS)
    #print_dataframes(df_dict)
    bilstm_input, ann_input, y_true = get_testing_data(df_dict)
    # bilstm_input, ann_input, y_true = get_testing_data()
    print('preprocessing done.')

    filepath = 'experiment-1/models/binary_hybrid_larger.h5'
    ensemble_model = load_model(filepath=filepath)
    print('loaded the model')
    print(bilstm_input.shape)
    print(ann_input.shape)

    y_pred = ensemble_model.predict([bilstm_input, ann_input])
    y_pred = [0 if pred[0] < 0.5 else 1 for pred in y_pred]

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_true, y_pred), '\n')

    print("=== Classification Report ===")
    print(classification_report(y_true, y_pred, zero_division=0), '\n')
    import scikitplot
    import matplotlib.pyplot as plt
    scikitplot.metrics.plot_confusion_matrix(y_true, y_pred, title='Hybrid Bi-LSTM and ANN model on Tweets Dataset Confusion Matrix')
    plt.savefig("experiment-1/results/binary_tweets_hybrid_classified.png")

