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


def gather_training_metrics(X_test, y_test,name = "test"):
    """Fits the model for a given set of input and output embeddings

    Args:
        X_test np array: array of inputs embedded vectors
        y_test np array: array out outputs
        name (str, optional): name of the output file and confusion matrix title. Defaults to "test".
    """
    
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

def provide_text_sentiment_df(dataset_idx=1):
    """Gets datasets from global, cleans them based on what is required for each.
    

    Args:
        dataset_idx (int, optional): which dataset to get. Defaults to 1.

    Returns:
        dataframe: cleaned dataframe (standardized sentiments only)
    """
    #ugly hardcoded values but it works.
    if dataset_idx == 1:
        df = pd.read_csv(TESTING_DATASETS[dataset_idx], encoding="ISO-8859-1", header=None)
        df.columns = ['sentiment', 'text']
        df = df[df['sentiment'] != 'neutral']
        df['sentiment'] = df['sentiment'].str.lower() #lowercase
        df['sentiment'] = df['sentiment'].apply(preprocess.standardize) #standarize sentiments
        return df
    elif dataset_idx == 0:
        df = pd.read_csv(TESTING_DATASETS[dataset_idx], encoding="ISO-8859-1")
        print(df.columns)
        df = df[['Headline', 'Final Status']]
        df.columns = ['text', 'sentiment']
        df['sentiment'] = df['sentiment'].str.lower() #lowercase
        df['sentiment'] = df['sentiment'].apply(preprocess.standardize) #standardize sentiments
        return df
    elif dataset_idx == 2:
        df = pd.read_csv(TESTING_DATASETS[dataset_idx], encoding="ISO-8859-1")
        df.columns = ['text', 'sentiment']
        df = df[df['sentiment'] != 2]
        print(df.shape)
        return df
    

def get_testing_data(df):
    """Cleans dataframe

    Args:
        df dataframe: dataset
    Returns:
        tuple(numpy array, numpy array): embeddings
    """

    tokens = (df['text'] #clean data
              .apply(preprocess.remove_html_tags)
              .apply(preprocess.tokenize)
              .apply(preprocess.remove_stop_words))
    embeddings = preprocess.provide_word_embeddings(tokens) #tokenize via the word embeddings

    labels = df['sentiment'] #rename the column...
    return embeddings, labels
    


if __name__ == '__main__':

    model = keras.models.load_model('experiment-2/models/binary_atlstm.model.keras')
    
    for dataset_idx in range(len(TESTING_DATASETS)): #loop through datasets, fit model
        dataset = provide_text_sentiment_df(dataset_idx=dataset_idx)
        X_test, y_test = get_testing_data(dataset)
        gather_training_metrics(X_test,y_test,name = f"Financial_{dataset_idx}") #fit model
    