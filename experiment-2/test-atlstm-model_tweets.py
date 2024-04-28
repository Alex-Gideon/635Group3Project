import matplotlib.pyplot as plt
import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
import preprocess
import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm

SEED = 42
TWEETS_DATASETS = 'experiment-1/tweets/datasets/atlstm_tweets.csv'


def gather_training_metrics(X_test, y_test,name = "test"):

    # cleaned_texts, sentiments = train_at_lstm.get_training_data(sample_size=50000)

    # X_train, X_test, y_train, y_test = train_test_split(cleaned_texts, sentiments, 
    #                                                      test_size=0.5)
    
    
    
    print("loading model..")
    model = keras.models.load_model('experiment-2/models/binary_atlstm.model.keras')
    print("model loaded... predictings...")
    y_pred = model.predict(X_test)
    print("fixing predictions...")
    y_pred = [0 if pred[0] < 0.5 else 1 for pred in y_pred]
    print("Generating confusion matrix.")
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
    display.plot(cmap=plt.cm.Blues)
    plt.savefig(f'experiment-2/results/atlstm_validation_{name}_confusion_matrix.png')
    
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred), '\n')

    class_report = classification_report(y_test, y_pred)
    print(f'===Classification Report===\n {class_report}')



def provide_word_embeddings_fast(text_sequences, wv):
    sequence_max_len = 80
    num_sequences = len(text_sequences)
    word_embeddings = np.zeros((num_sequences, sequence_max_len, wv.vector_size))
    
    missing_words = set()
    for i, seq in tqdm(enumerate(text_sequences), total=num_sequences):
        vector = []
        for word in seq:
            if word in wv:
                vector.append(wv[word])
            else:
                missing_words.add(word)
                vector.append([0.0] * wv.vector_size)
        
        vector += [[0.0] * wv.vector_size for _ in range(sequence_max_len - len(vector))]
        word_embeddings[i] = vector[:sequence_max_len]
    
    print("Stopped")
    if missing_words:
        print(f"Missing words: {missing_words}")
    
    return word_embeddings

def get_testing_data(df):
    print("Tokenizing")
    tokens = (df['review']
              .apply(preprocess.remove_html_tags)
              .apply(preprocess.tokenize)
              .apply(preprocess.remove_stop_words))
    print("Embedding...")
    wv = KeyedVectors.load('experiment-2/models/w2v.glove.model')
    embeddings = provide_word_embeddings_fast(tokens,wv)
    print("fixing labels")
    labels = df['sentiment'].apply(preprocess.standardize) #i doint think we need this..?
    
    return embeddings, labels
    


if __name__ == '__main__':
    #dataset_idx = 2
    #dataset = provide_text_sentiment_df(dataset_idx=dataset_idx)
    #print(dataset)
    tweets = pd.read_csv(TWEETS_DATASETS)
    tweets10k = tweets.sample(n=10000, random_state=42)
    print("Read Data")
    print("Cleaning...")
    X_test_tweets, y_test_tweets = get_testing_data(tweets10k)
    print("Cleaning Complete, Testing now.")
    gather_training_metrics(X_test_tweets,y_test_tweets,name = "Tweets")
    
