# -*- coding: utf-8 -*-
"""financial_cleaning.ipynb

"""

import pandas as pd
import re

"""**Pretty Clean Dataset Cleaning**"""

pretty_df = pd.read_csv("financial-news-pretty-clean.csv")

pretty_df.head(5)

pretty_df.info()

pretty_df = pretty_df.drop(["Date_published", "Headline", "Synopsis"], axis=1)

pretty_df.head(5)

pretty_df["Score"] = pretty_df["Final Status"].apply(
    lambda x: "10" if x == "Positive" else "1" if x == "Negative" else "7"
    )

pretty_df.head(5)

pretty_df.info()

syn = set(pretty_df["Full_text"])

print(len(syn))

sample = pretty_df['Full_text'][0]

sample = ''.join(filter(lambda x: x.isalpha() or x.isspace(), sample))

sample = sample.replace("\n", " ")
sample = sample.replace("â", "")
sample = sample.split(" ")
print(sample)

word_list = []
labels = []

for index, row in pretty_df.iterrows():
  label = row['Score']
  line = ''.join(filter(lambda x: x.isalpha() or x.isspace(), row['Full_text']))
  line = line.replace("\n", " ")
  line = line.replace("â", "")
  word_list.append(line)
  labels.append(label)

print(len(word_list))
print(len(labels))

dict = {'label': labels, 'text': word_list}

cleaned_df = pd.DataFrame(dict)

cleaned_df.head(5)

cleaned_df.to_csv('../experiment-1/financial/datasets/clean_financialpc.csv', index=False)

"""**PhraseBook Cleaning**"""

df = pd.read_csv("financial-phrase-bank-all.csv")

df.head(5)

df.info()

df["Score"] = df["Status"].apply(
    lambda x: "10" if x == "positive" else "1" if x == "negative" else "7"
    )

df.head(5)

df.info()

syn = set(df["Text"])

print(len(syn))

df.drop_duplicates('Text', inplace = True)

df.info()

sample = df['Text'][0]

sample = ''.join(filter(lambda x: x.isalpha() or x.isspace(), sample))

sample = sample.replace("\n", " ")
sample = sample.replace("â", "")
sample = re.sub(' +', ' ', sample)
print(sample)

word_list = []
labels = []

for index, row in df.iterrows():
  label = row['Score']
  line = ''.join(filter(lambda x: x.isalpha() or x.isspace(), row['Text']))
  line = line.replace("\n", " ")
  line = line.replace("â", "")
  word_list.append(line)
  labels.append(label)

print(len(word_list))
print(len(labels))

dict = {'label': labels, 'text': word_list}

cleaned_df = pd.DataFrame(dict)

cleaned_df.head(5)

cleaned_df.to_csv('../experiment-1/financial/datasets/clean_financialfull.csv', index=False)