import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import scale

from sklearn.metrics import classification_report, confusion_matrix
import csv
from keras.models import load_model
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from collections import Counter

rowsx = []


with open("data/train_multiclass_hybrid_BiLSTM_data.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)
    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        for i in range(1, len(row)):
            for j in row[i].split("\n"):
                    rows1.append(j)



        del (row[0])
        rowsx.append(rows1)

rowsx1 = []


with open("data/train_multiclass_hybrid_ANN_data.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)
    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        for i in range(1, len(row)):
            for j in row[i].split("\n"):
                    rows1.append(j)
        del (row[0])
        rowsx1.append(rows1)

t = Tokenizer()

t.fit_on_texts(rowsx)
vocab_size = len(t.word_index) + 1
print(vocab_size)

embeddings = Word2Vec(size=200, min_count=3)
embeddings.build_vocab([sentence for sentence in rowsx1])
embeddings.train([sentence for sentence in rowsx1],
                 total_examples=embeddings.corpus_count,
                 epochs=embeddings.epochs)

gen_tfidf = TfidfVectorizer(analyzer=lambda x: x, min_df=3)
matrix = gen_tfidf.fit_transform([sentence   for sentence in rowsx1])
tfidf_map = dict(zip(gen_tfidf.get_feature_names(), gen_tfidf.idf_))
print(len(tfidf_map))

def encode_sentence(tokens, emb_size):
    _vector = np.zeros((1, emb_size))
    length = 0
    for word in tokens:
        try:
            _vector += embeddings.wv[word].reshape((1, emb_size)) * tfidf_map[word]
            length += 1
        except KeyError:
            continue
        break

    if length > 0:
        _vector /= length

    return _vector


model = load_model("data/multiclass_hybrid.h5")

rowsx_t_1 = []
y1 = []
y2 = []
y3 = []
y4 = []
y7 = []
y8 = []
y9 = []
y10 = []

with open("data/test_multiclass_hybrid_BiLSTM_data.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)
    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        for i in range(1, len(row)):
            for j in row[i].split("\n"):
                    rows1.append(j)


        if row[0] == "1":
            y1.append(1)
            y2.append(0)
            y3.append(0)
            y4.append(0)
            y7.append(0)
            y8.append(0)
            y9.append(0)
            y10.append(0)
        elif row[0] == "2":
            y1.append(0)
            y2.append(1)
            y3.append(0)
            y4.append(0)
            y7.append(0)
            y8.append(0)
            y9.append(0)
            y10.append(0)
        elif row[0] == "3":
            y1.append(0)
            y2.append(0)
            y3.append(1)
            y4.append(0)
            y7.append(0)
            y8.append(0)
            y9.append(0)
            y10.append(0)
        elif row[0] == "4":
            y1.append(0)
            y2.append(0)
            y3.append(0)
            y4.append(1)
            y7.append(0)
            y8.append(0)
            y9.append(0)
            y10.append(0)
        elif row[0] == "7":
            y1.append(0)
            y2.append(0)
            y3.append(0)
            y4.append(0)
            y7.append(1)
            y8.append(0)
            y9.append(0)
            y10.append(0)
        elif row[0] == "8":
            y1.append(0)
            y2.append(0)
            y3.append(0)
            y4.append(0)
            y7.append(0)
            y8.append(1)
            y9.append(0)
            y10.append(0)
        elif row[0] == "9":
            y1.append(0)
            y2.append(0)
            y3.append(0)
            y4.append(0)
            y7.append(0)
            y8.append(0)
            y9.append(1)
            y10.append(0)
        elif row[0] == "10":
            y1.append(0)
            y2.append(0)
            y3.append(0)
            y4.append(0)
            y7.append(0)
            y8.append(0)
            y9.append(0)
            y10.append(1)
        del (row[0])
        rowsx_t_1.append(rows1)


rowsx_t_2 = []


with open("data/test_multiclass_hybrid_ANN_data.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)
    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        for i in range(1, len(row)):
            for j in row[i].split("\n"):
                    rows1.append(j)



        del (row[0])
        rowsx_t_2.append(rows1)

encoded_train_set = t.texts_to_sequences(rowsx_t_1)
SEQ_LEN = 80


padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

train_docs = [list(doc) for doc in padded_train]
x_train = np.array([np.array(token) for token in train_docs])
x_train1 = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, rowsx_t_2)]))


count = 0

pred = []

predicted = model.predict([x_train,x_train1])
print(len(predicted))
ytrue = []

for i in range(0,len(y1)):
    # ytrue = 0
    values = []
    for j in range(0,8):
        if predicted[j][i][0] > 0.5:
            if j<4:
                values.append(j+1)
            else:
                values.append(j+3)
            
    flag = 0
    if not values:
        flag =1
    if y1[i] == 1:
        ytrue.append(1)
        if flag == 1:
            values.append(1)
    if y2[i] == 1:
        ytrue.append(2)
        if flag == 1:
            values.append(2)
    if y3[i] == 1:
        ytrue.append(3)
        if flag == 1:
            values.append(3)
    if y4[i] == 1:
        ytrue.append(4)
        if flag == 1:
            values.append(4)
    if y7[i] == 1:
        ytrue.append(7)
        if flag == 1:
            values.append(7)
    if y8[i] == 1:
        ytrue.append(8)
        if flag == 1:
            values.append(8)
    if y9[i] == 1:
        ytrue.append(9)
        if flag == 1:
            values.append(9)
    if y10[i] == 1:
        ytrue.append(10)
        if flag == 1:
            values.append(10)
    pred.append(values[0])
    if ytrue[i] == pred[i]:
        count+=1

print(count)
print((count/len(y1))*100)

import scikitplot
import matplotlib.pyplot as plt
print("=== Confusion Matrix ===")
print(confusion_matrix(ytrue, pred))
print('\n')
print("=== Classification Report ===")
print(classification_report(ytrue, pred))
print('\n')
scikitplot.metrics.plot_confusion_matrix(ytrue, pred)
plt.savefig("results/multiclass_hybrid_confusion_matrix.png")