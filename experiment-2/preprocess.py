import numpy as np
import tensorflow as tf
from bs4 import BeautifulSoup as beauty
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
import nltk
import re

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
tokenizer = tf.keras.preprocessing.text.Tokenizer()


def remove_html_tags(row):
    """removes html tags from a sentence

    Args:
        row (pandas row:string): a pandas row containing the sentence to clean

    Returns:
        string: cleaned string
    """
    return beauty(row, 'html.parser').text


def tokenize(input_text):
    """tokenizes the sentence string into individual words

    Args:
        input_text (string): sentence string

    Returns:
        [str]: list of words (tokens)
    """
    tokens = re.sub('[^a-zA-Z]', ' ', input_text).lower().split()
    return tokens


def remove_stop_words(input_text_vector):
    """removes stop words from a list of words/tokens

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


def standardize(input_label):
    """standardizes the labels such that 0:negative, 1:positive

    Args:
        input_label (str): string that is either negative or positive

    Returns:
        int: label
    """
    return 1 if input_label == 'positive' else 0


def all_at_once(input_text):
    """does all the preprocessing all at once

    Args:
        input_text (str): sentence string

    Returns:
        [str]: list of cleaned tokens
    """
    cleaned = remove_html_tags(input_text)
    cleaned = tokenize(cleaned)
    cleaned = remove_stop_words(cleaned)
    return cleaned


def provide_word_embeddings(text_sequences):
    """generates word embeddings using the glove model

    Args:
        text_sequences ([str]): list of tokens

    Returns:
        [[float]]: word embeddings
    """
    wv = KeyedVectors.load('experiment-2/models/w2v.glove.model')
    sequence_max_len = 80
    word_embeddings = []
    for seq in text_sequences:
        vector = [wv[word] for word in seq if word in wv]
        if len(vector) < sequence_max_len:
            vector += [[0.0] * wv.vector_size for _ in range(sequence_max_len - len(vector))]
        word_embeddings.append(vector[:sequence_max_len])

    return np.array(word_embeddings)
