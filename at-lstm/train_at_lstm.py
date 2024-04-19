import at_lstm
from preprocess import *


def get_training_data(sample_size=50, sequence_max_len=80):
    filename = 'datasets/IMDB Dataset.csv'
    df = pd.read_csv(filename, nrows=sample_size)

    tokens = (df['review']
              .apply(remove_html_tags)
              .apply(tokenize)
              .apply(remove_stop_words))

    tokenizer.fit_on_texts(tokens)
    encoded_sentences = tokenizer.texts_to_sequences(tokens)
    padded_sentences = (tf.keras.preprocessing.sequence.
                        pad_sequences(encoded_sentences,
                                      maxlen=sequence_max_len,
                                      padding='post'))
    labels = df['sentiment'].apply(standardize)
    return padded_sentences, labels


if __name__ == '__main__':
    padded_text_sequences, y_train = get_training_data()
    x_train = provide_word_embeddings(padded_text_sequences)

    model = at_lstm.build_model()
    model.fit(x_train, y_train, epochs=5)

    print('training done.')
