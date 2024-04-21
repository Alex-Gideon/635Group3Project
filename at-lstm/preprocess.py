import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
from bs4 import BeautifulSoup as beauty
from gensim.models import KeyedVectors
from nltk.corpus import stopwords

# nltk.download('stopwords')
stop_words = stopwords.words('english')

word_tokenizer = tf_text.UnicodeScriptTokenizer()
tokenizer = tf.keras.preprocessing.text.Tokenizer()


def remove_html_tags(row):
    return beauty(row, 'html.parser').text


def tokenize(input_text):
    tokens = word_tokenizer.tokenize(input_text)
    return [token.numpy().decode('utf-8') for token in tokens]


def remove_stop_words(input_text_vector):
    filtered = list(filter(lambda word: word.isalnum() and word.lower() not in stop_words, input_text_vector))
    return list(map(lambda word: word.lower(), filtered))


def standardize(input_label):
    return 1 if input_label == 'positive' else 0


def provide_word_embeddings(text_sequences):
    wv = KeyedVectors.load('models/w2v.glove.model')
    sequence_max_len = 80
    word_embeddings = []
    for seq in text_sequences:
        vector = [wv[word] for word in seq if word in wv]
        if len(vector) < sequence_max_len:
            vector += [[0.0] * wv.vector_size for _ in range(sequence_max_len - len(vector))]
        word_embeddings.append(vector[:sequence_max_len])

    return np.array(word_embeddings)
