import matplotlib.pyplot as plt
import keras
import random
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

SEED = 42
random.seed(SEED)


def gather_training_metrics(X_test, y_test):

    # cleaned_texts, sentiments = train_at_lstm.get_training_data(sample_size=50000)

    # X_train, X_test, y_train, y_test = train_test_split(cleaned_texts, sentiments, 
    #                                                      test_size=0.5)

    model = keras.models.load_model('experiment-2/models/binary_atlstm.model.keras')

    y_pred = model.predict(X_test)
    y_pred = [0 if pred[0] < 0.5 else 1 for pred in y_pred]

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
    display.plot(cmap=plt.cm.Blues)
    plt.savefig('experiment-2/results/atlstm_validation_confusion_matrix.png')

    class_report = classification_report(y_test, y_pred)
    print(f'===Classification Report===\n {class_report}')

