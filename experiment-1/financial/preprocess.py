"""preprocess.py
"""

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
    """
    Args:
        row: Input text

    Returns:
        Input text without html tags
    """
    return beauty(row, 'html.parser').text


def tokenize(input_text):
    """Splits input text into individual word elements (tokens) using regular expressions.

    Args:
        input_text (str): Text to tokenize

    Returns:
        list[str]: List of tokens
    """
    tokens = re.sub('[^a-zA-Z]', ' ', input_text).lower().split()
    return tokens


def remove_stop_words(input_text_vector):
    """Removes words that have no semantic value.

    Args:
        input_text_vector (list[str]): Input text

    Returns:
        list[str]: Input text without stop words.
    """
    filtered = []
    for word in input_text_vector:
        if word.isalpha() and word not in stop_words:
            filtered.append(word)
    return filtered


def standardize(input_label):
    """Converts positive/negative label into 0/1 label.

    Args:
        input_label (str): Input label

    Returns:
        int: Binary label
    """
    return 1 if input_label == 'positive' else 0


def all_at_once(input_text):
    """Performs all preprocessing steps on given input text.

    Args:
        input_text (list[str]): Input text

    Returns:
        list[str]: Text that is tokenized, without HTML tags and stop words.
    """
    cleaned = remove_html_tags(input_text)
    cleaned = tokenize(cleaned)
    cleaned = remove_stop_words(cleaned)
    return cleaned


def provide_word_embeddings(text_sequences):
    """Collects word embeddings for a given sequence of tokens.

    Args:
        text_sequences (list[str]): List of tokens

    Returns:
        ndarray: Numpy array of word embeddings
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
