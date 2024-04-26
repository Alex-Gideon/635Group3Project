# -*- coding: utf-8 -*-
"""preprocessing_financial.ipynb
"""

import nltk
#if receiving message that stopwords, averaged_perceptron_tagger, or wordnet not found
#user commmand line: python -m nltk.downloader [resource name]
#Error from nltk_data might appear
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from os import walk

from nltk.corpus import stopwords

import csv

import pandas as pd

stop_words = set(stopwords.words('english'))

def check_for_hypernim(token):
    hypernims = []
    for i in range(15):
        try:
            hypernims1 = []
            for i, j in enumerate(wn.synsets(token)):
                for l in j.hypernyms():
                    hypernims1.append(l.lemma_names()[0])
            token = hypernims1[0]
            hypernims.append(hypernims1)
        except IndexError:
            hypernims.append([token])

    return hypernims

def process_ANN(fileName):
  df = pd.read_csv(fileName)

  processed = []
  
  sep_labels = []

  ind_words = []
  
  split_sentences = []
  labels = []

  for index, row in df.iterrows():
    label = row['label']
    line = row['text']
    words = line.split()

    
    split_sentences.append([x for x in words if x not in stop_words])
    labels.append(label)

    filtered_sentence = []

    x = []
    for r in words:

        if not r in stop_words:
            filtered_sentence.append(r)

    tagged = nltk.pos_tag(filtered_sentence)
    for i in tagged:
        if len(i[0]) != 0 or len(i[0]) != 1:
            if i[1] == 'IN' or i[1] == 'DT' or i[1] == 'CD' or i[1] == 'CC' or i[1] == 'EX' or i[1] == 'MD' or   i[1] == 'WDT' or i[1] == 'WP' or i[1] == 'UH' or i[1] == 'TO' or i[1] == 'RP' or i[1] == 'PDT' or i[1] == 'PRP' or i[1] == 'PRP$' or i[0] == 'co':
                # print(i[0])
                continue
            else:
                x.append(i[0].rstrip(".,?!"))
    for i in x:
        ind_words.append(i)
        l = []
        l.append(i)
        hype = check_for_hypernim(i)
        if len(hype) == 0:
            hype.append(i)  # 1
            hype.append(i)  # 2
            hype.append(i)  # 3
            hype.append(i)  # 4
            hype.append(i)  # 5
            hype.append(i)  # 6
            hype.append(i)  # 7
            hype.append(i)  # 8
            hype.append(i)  # 9
            hype.append(i)  # 10
            hype.append(i)  # 11
            hype.append(i)  # 12
            hype.append(i)  # 13
            hype.append(i)  # 14
            hype.append(i)  # 15
        for hyper in hype:
            l.append(hyper[0])
        processed.append(l)
        sep_labels.append(label)

  print(len(processed))
  print(len(sep_labels))

  print(f'Label 0: {sep_labels[0]}')
  print(f'Processed 0: {processed[0]}')

  word_only = [x[0] for x in processed]

  dict = {"label": sep_labels, "hypernyms":word_only}
  ind_dict = {"label": sep_labels, "words":ind_words}

  processed_df = pd.DataFrame(processed)
  individual_df = pd.DataFrame(ind_words)

  split_df = pd.DataFrame(split_sentences)
  labels_df = pd.DataFrame(labels)

  split_df = labels_df.join(split_df, lsuffix="_")

  sep_labels_df = pd.DataFrame(dict)
  sep_labels_df.head(5)

  joined_df = sep_labels_df.join(processed_df)
  joined_df.drop(['hypernyms'], axis=1, inplace=True)

  joined_ind = sep_labels_df.join(individual_df)
  joined_ind.drop(['hypernyms'], axis=1, inplace=True)

  if fileName == '../experiment-1/financial/datasets/clean_financialpc.csv':

    processed_df.to_csv(
        '../experiment-1/financial/datasets/financialpc_ann.csv',
        header=False,
        index=False
        )

    joined_df.to_csv(
        '../experiment-1/financial/datasets/financialpc_ann.csv',
        header=False,
        index=False
        )

    joined_ind.to_csv(
        '../experiment-1/financial/datasets/financialpc_bilstm.csv',
        header=False,
        index=False
        )

    split_df.to_csv(
        '../experiment-1/financial/datasets/financialpc_hybdrid_bilstm.csv',
        header=False,
        index=False
        )

    print("clean_financialpc has been processed")

  if fileName == '../experiment-1/financial/datasets/clean_financialfull.csv':
    processed_df.to_csv(
        '../experiment-1/financial/datasets/financialfull_ANN.csv',
        header=False,
        index=False
        )

    joined_df.to_csv(
        '../experiment-1/financial/datasets/financialfull_ANN.csv',
        header=False,
        index=False
        )

    joined_ind.to_csv(
        '../experiment-1/financial/datasets/financialfull_bilstm.csv',
        header=False,
        index=False
        )

    split_df.to_csv(
        '../experiment-1/financial/datasets/financialfull_hybrid_biltsm.csv',
        header=False,
        index=False
        )

    print("clean_financialfull has been processed")

datasets = ['../experiment-1/financial/datasets/clean_financialpc.csv',
            '../experiment-1/financial/datasets/clean_financialfull.csv']

for dataset in datasets:
  process_ANN(dataset)