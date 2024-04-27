import time
import pandas as pd
import at_lstm
from preprocess import *
import random
from sklearn.model_selection import train_test_split
import keras
import at_lstm_metric


SEED = 42
random.seed(SEED)


def get_training_data(sample_size=50):
    filepath = 'experiment-2/datasets/IMDBDataset.csv'
    df = pd.read_csv(filepath)
    
    sampled_df = df.sample(n=sample_size, random_state=SEED)

    tokens = (sampled_df['review']
              .apply(remove_html_tags)
              .apply(tokenize)
              .apply(remove_stop_words))

    labels = sampled_df['sentiment'].apply(standardize)
    return tokens, labels


if __name__ == '__main__':
    start = time.time()
    cleaned_texts, sentiments = get_training_data(sample_size=50000)

    X_train, X_test, y_train, y_test = train_test_split(cleaned_texts, sentiments, 
                                                         test_size=0.5)

    X_train = provide_word_embeddings(X_train)
    print('x train embeddings loaded')
    X_test = provide_word_embeddings(X_test)
    print('x test embeddings loaded')

    elapsed = time.time() - start
    print(f'data preparation time = {elapsed}; done.')

    checkpoint_filepath = 'experiment-2/models/binary_atlstm.model.keras'
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
    )

    model = at_lstm.build_model()
    model.fit(X_train, y_train,
              batch_size=512,
              epochs=10,
              validation_data=(X_test, y_test),
              use_multiprocessing=True,
              callbacks=[model_checkpoint_callback],)
 
    print('training done. model saved.')

    at_lstm_metric.gather_training_metrics(X_test, y_test)
    print('experiment results gathered. saved to /results')
