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
yx = []

with open("data/train_binary_hybrid_BiLSTM_data.csv", 'r', encoding='latin1') as csv1:
    # creating a csv reader object
    csvreader1 = csv.reader(csv1)
    # extracting each data row one by one
    for row in csvreader1:
        rows1 = []
        for i in range(1, len(row)):
            for j in row[i].split("\n"):
                    rows1.append(j)
        yx.append(row[0])
        del (row[0])


        rowsx.append(rows1)


for i in range(0,len(yx)):
    if int(yx[i])<=4:
        yx[i] = 0
    else:
        yx[i] = 1


rowsx1 = []


with open("data/train_binary_hybrid_ANN_data.csv", 'r', encoding='latin1') as csv1:
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
x_train = np.array([np.array(token) for token in train_docs])
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

# this is the same code as in bilstm
input_tensor = Input(shape=(SEQ_LEN,), dtype='int32')
e = Embedding(vocab_size, 300, input_length=SEQ_LEN, trainable=True)(input_tensor)
x = Bidirectional(LSTM(128, return_sequences=True))(e)
x = Bidirectional(LSTM(64, return_sequences=False))(x)
x = Dense(64, activation='relu')(x)
output_tensor = Dense(1, activation='sigmoid')(x)
model = Model(input_tensor, output_tensor)
####

yx = np.array(yx)
## ANN?
visible = Input(shape=(200,))

c1 = Dense(256,activation='relu')(visible)
c2 = Dense(256,activation='relu')(c1)
c3 = Dense(512,activation='relu')(c2)
c4 = Dense(1024,activation='relu')(c3)
c5 = Dense(2048,activation='relu')(c4)
c6 = Dense(1024,activation='relu')(c5)
c7 = Dense(512,activation='relu')(c6)
c8 = Dense(256,activation='relu')(c7)
s1 = Dense(1,activation='sigmoid')(c8)
model1 = Model(inputs=visible,outputs=s1)
##ANN
combined = Concatenate()([model.output, model1.output])
mix1 = Dense(100,activation='relu')(combined)
mix2 = Dense(50,activation='relu')(mix1)
out = Dense(1,activation="sigmoid")(mix2)

model2 = Model(inputs=[input_tensor,visible],outputs=out)

model2.compile(optimizer=Adam(lr=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model2.fit([x_train,x_train1], yx, epochs=10, batch_size=512)
model2.save("results/binary_hybrid.h5")