import csv
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,Model
from keras.layers import Dense, Input, Concatenate
from keras.layers import LSTM,Embedding,Bidirectional
from keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.preprocessing import scale

rowsx = []
y1 = []
y2 = []
y3 = []
y4 = []
y7 = []
y8 = []
y9 = []
y10 = []

with open("data/train_multiclass_hybrid_BiLSTM_data.csv", 'r', encoding='latin1') as csv1:
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
encoded_train_set = t.texts_to_sequences(rowsx)
SEQ_LEN = 80
padded_train = pad_sequences(encoded_train_set, maxlen=SEQ_LEN, padding='post')

train_docs = [list(doc) for doc in padded_train]

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


x_train1 = scale(np.concatenate([encode_sentence(ele, 200) for ele in map(lambda x: x, rowsx1)]))
print(x_train1.shape)

input_tensor = Input(shape=(SEQ_LEN,), dtype='int32')
e = Embedding(vocab_size, 300, input_length=SEQ_LEN, trainable=True)(input_tensor)
x = Bidirectional(LSTM(128, return_sequences=True))(e)
x = Bidirectional(LSTM(64, return_sequences=False))(x)
x = Dense(64, activation='relu')(x)
output_tensor = Dense(8, activation='relu')(x)
model = Model(input_tensor, output_tensor)

visible = Input(shape=(200,))

c1 = Dense(256,activation='relu')(visible)
c2 = Dense(256,activation='relu')(c1)
c3 = Dense(512,activation='relu')(c2)
c4 = Dense(1024,activation='relu')(c3)
c5 = Dense(2048,activation='relu')(c4)
c6 = Dense(1024,activation='relu')(c5)
c7 = Dense(512,activation='relu')(c6)
c8 = Dense(256,activation='relu')(c7)
s1 = Dense(8,activation='relu')(c8)
model1 = Model(inputs=visible,outputs=s1)

combined = Concatenate()([model.output, model1.output])
mix1 = Dense(100,activation='relu')(combined)
mix2 = Dense(50,activation='relu')(mix1)
out1 = Dense(1,activation="sigmoid")(mix2)
out2 = Dense(1,activation="sigmoid")(mix2)
out3 = Dense(1,activation="sigmoid")(mix2)
out4 = Dense(1,activation="sigmoid")(mix2)
out5 = Dense(1,activation="sigmoid")(mix2)
out6 = Dense(1,activation="sigmoid")(mix2)
out7 = Dense(1,activation="sigmoid")(mix2)
out8 = Dense(1,activation="sigmoid")(mix2)

model2 = Model(inputs=[input_tensor,visible],outputs=[out1,out2,out3,out4,out5,out6,out7,out8])

model2.compile(optimizer=Adam(lr=1e-3),
               loss='binary_crossentropy',
              metrics=['accuracy'])
x_train = np.array([np.array(token) for token in train_docs])
model2.fit([x_train,x_train1], [np.array(y1),np.array(y2),np.array(y3),np.array(y4),np.array(y7),np.array(y8),np.array(y9),np.array(y10)], epochs=10, batch_size=512)
model2.save("results/multiclass_hybrid.h5")