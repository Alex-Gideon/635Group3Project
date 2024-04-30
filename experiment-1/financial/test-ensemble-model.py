"""test-ensemble-model.py
"""

import os
import pandas as pd
import nltk
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
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
import matplotlib.pyplot as plt



stop_words = set(stopwords.words('english'))
SEED = 42
random.seed(SEED)

TESTING_DATASETS = [
    'experiment-1/financial/datasets/financial-news-pretty-clean.csv',
    'experiment-1/financial/datasets/financial-phrase-bank-all.csv',
    'experiment-1/financial/datasets/twitter_financial_sentiment_data/data.csv'
]


def tokenize(input_text):
    """Tokenizes the sentence string into individual words
    Args:
        input_text (string): sentence string
    Returns:
        [str]: list of words (tokens)
    """

    tokens = re.sub('[^a-zA-Z]', ' ', input_text).lower().split()
    return tokens


def remove_stop_words(input_text_vector):
    """Removes stop words from a list of words/tokens
    Args:
        input_text_vector ([str]): list of words/tokens
    Returns:
        [str]: tokens with no stop words
    """

    filtered = []
    for word in input_text_vector:
        if word.isalpha() and word not in stop_words:
            filtered.append(word)
    return filtered


def get_hypernym(word):
    """Retrieves the hypernym of a given token

    Args:
        word (str): token

    Returns:
        list[str]: hypernyms
    """
    for hyps in wn.synsets(word):
        for hyp in hyps.hypernyms():
            return hyp.lemma_names()[0]
    return None


def standardize(input_label):
    """Standardizes the labels such that 0:negative, 1:positive
    Args:
        input_label (str): string that is either negative or positive
    Returns:
        int: label
    """

    if isinstance(input_label, int): return input_label
    input_label = input_label.strip().lower()
    labels = {
        'positive': 1,
        'negative': 0,
        'neutral': -1
    }
    return labels[input_label]


def provide_embeddings_tfidf(texts):
    """Generate word embeddings and calculate TF-IDF (Term Frequency-Inverse Document Frequency) scores for the given text.

    Args:
        texts (list): Texts to build embeddings for

    Returns:
        Word2Vec, dict: embeddings, tfidf
    """
    
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
    """Encode text hypernyms into a vector representation.

    Args:
        tfidf_map (dict): A dictionary mapping words to their TF-IDF scores.
        word_embedding_model (Word2Vec): A trained Word2Vec model for word embeddings
        text_hypernyms (list): A list of hypernyms to be encoded.

    Returns:
        ndarray: vector encoding of hypernyms
    """

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


def provide_text_sentiment_df(dataset_idx=0):
    """Reads different datasets based on the index provided and prepares DataFrame for text sentiment analysis.

    Args:
        dataset_idx (int, optional): Index of data to be used. Defaults to 0.

    Returns:
        DataFrame: text and sentiment columns
    """

    if dataset_idx == 1:
        df = pd.read_csv(TESTING_DATASETS[dataset_idx], encoding="ISO-8859-1", header=None)
        df.columns = ['sentiment', 'text']
        df = df[df['sentiment'] != 'neutral']
        return df
    elif dataset_idx == 0:
        df = pd.read_csv(TESTING_DATASETS[dataset_idx], encoding="ISO-8859-1")
        print(df.columns)
        df = df[['Headline', 'Final Status']]
        df.columns = ['text', 'sentiment']
        return df
    elif dataset_idx == 2:
        df = pd.read_csv(TESTING_DATASETS[dataset_idx], encoding="ISO-8859-1")
        df.columns = ['text', 'sentiment']
        df = df[df['sentiment'] != 2]
        print(df.shape)
        return df



def get_testing_data(dataset):
    """Prepare testing data for sentiment analysis

    Args:
        dataset (int): Index of the dataset

    Returns:
        _type_: BI-LSTM Test Data, ANN Test Data, Ground Truth
    """
    
    df = provide_text_sentiment_df(dataset_idx=dataset)

    # remove stop words
    financial_text_tokens = (df['text']
                        .apply(tokenize)
                        .apply(remove_stop_words)).tolist()
    
    # pos tagging and filtering them
    tagged_texts = nltk.pos_tag_sents(financial_text_tokens)
    tags_to_ignore = set(['IN' ,'DT' ,'CD' ,'CC' ,'EX' ,'MD','WDT' ,'WP' ,'UH' ,'TO' ,'RP' ,'PDT' ,'PRP' ,'PRP$','co'])
    
    result_of_filtering_tags = []
    for sentence in tagged_texts:
        filtered_sentence = []
        for word, tag in sentence:
            if len(word) > 1 and tag not in tags_to_ignore:
                filtered_sentence.append(word.rstrip(".,?!"))

        result_of_filtering_tags.append(filtered_sentence)

    # generate hypernyms
    text_hypernyms = []
    for sentence in result_of_filtering_tags:
        sentence_hypernyms = []
        for word in sentence:
            hypernym = get_hypernym(word)
            if hypernym:
                sentence_hypernyms.append(hypernym)
            else:
                sentence_hypernyms.append(word)
        text_hypernyms.append(sentence_hypernyms)


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
    
    x_test_ann = np.asarray(x_test_ann).reshape(len(df), 200)
   
    y_true = df['sentiment'].apply(standardize).to_numpy().reshape(-1, 1)

    return x_text_bilstm, x_test_ann, y_true


if __name__ == '__main__':

    print('preprocessing done.')

    filepath = 'experiment-1/models/binary_hybrid_larger.h5'
    ensemble_model = load_model(filepath=filepath)

    for dataset in range(0,3):
        bilstm_input, ann_input, y_true = get_testing_data(dataset)
        print('preprocessing done.')
        print('loaded the model')
        print(bilstm_input.shape)
        print(ann_input.shape)

        y_pred = ensemble_model.predict([bilstm_input, ann_input])
        y_pred = [0 if pred[0] < 0.5 else 1 for pred in y_pred]

        cm = confusion_matrix(y_true, y_pred)
        display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
        display.plot(cmap=plt.cm.Blues)
        plt.savefig(f'experiment-1/financial/results/ensemble/ensemble_model_financial{dataset}_confusion_matrix.png')

        print("=== Classification Report ===")
        print(classification_report(y_true, y_pred, zero_division=0), '\n')

