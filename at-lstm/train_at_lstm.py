import time

import at_lstm
from preprocess import *


def get_training_data(sample_size=50):
    filename = 'datasets/IMDB Dataset.csv'
    df = pd.read_csv(filename, nrows=sample_size)

    tokens = (df['review']
              .apply(remove_html_tags)
              .apply(tokenize)
              .apply(remove_stop_words))

    labels = df['sentiment'].apply(standardize)
    return tokens, labels


if __name__ == '__main__':
    start = time.time()
    cleaned_texts, y_train = get_training_data(sample_size=1000)

    x_train = provide_word_embeddings(cleaned_texts)
    elapsed = time.time() - start
    print(f'data preparation time = {elapsed}; done.')

    model = at_lstm.build_model()
    model.fit(x_train, y_train,
              batch_size=100,
              epochs=10,
              validation_split=0.3,
              use_multiprocessing=True)

    print('training done.')
