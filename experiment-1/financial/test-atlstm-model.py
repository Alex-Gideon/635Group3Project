import matplotlib.pyplot as plt
import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pandas as pd
import preprocess

TESTING_DATASETS = [
    'experiment-1/financial/datasets/financial-news-pretty-clean.csv',
    'experiment-1/financial/datasets/financial-phrase-bank-all.csv',
    'experiment-1/financial/datasets/twitter_financial_sentiment_data/data.csv'
]

def provide_text_sentiment_df(dataset_idx=1):
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
    

def get_testing_data(df):

    tokens = (df['text']
              .apply(preprocess.remove_html_tags)
              .apply(preprocess.tokenize)
              .apply(preprocess.remove_stop_words))
    embeddings = preprocess.provide_word_embeddings(tokens)

    labels = df['sentiment'].apply(preprocess.standardize)
    return embeddings, labels
    


if __name__ == '__main__':
    dataset_idx = 2
    dataset = provide_text_sentiment_df(dataset_idx=dataset_idx)
    X_test, y_test = get_testing_data(dataset)

    model = keras.models.load_model('experiment-2/models/binary_atlstm.model.keras')
    y_pred = model.predict(X_test)
    y_pred = [0 if pred[0] < 0.5 else 1 for pred in y_pred]

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
    display.plot(cmap=plt.cm.Blues)
    plt.savefig(f'experiment-1/financial/results/at_lstm/atlstm_financial{dataset_idx}_confusion_matrix.png')

    class_report = classification_report(y_test, y_pred)
    print(f'===Classification Report===\n {class_report}')