# Sentiment Analysis Using Ensemble-Hybrid Model with Hypernym Based Feature Engineering

## Table of contents
* [General info](#general-info)
* [Technologies](#technologies)
* [Usage](#Usage-Instructions)
* [Authors](#Authors)

## General info
<p>Sentimental analysis helps in interpretation and classification of textual data using various techniques applied in deep learning models. The processing of movie reviews given by users is enhanced using feature engineering by adding hypernyms to the data to enhance the semantic meaning and establish semantic relationships between words of the reviews. This would help the system to identify the context used by the user in his review as positive or negative. We have implemented a hybrid Bi-LSTM-ANN model to match the spatial aspect of hypernym features with the temporal aspects of the reviews. The Bi-LSTM part models the semantic meaning of the reviews and the ANN part improves the efficiency of the model by adding the hypernyms of words in each review. </p>


## Technologies
Project is created with:
* Python version: 3.7.3
* tensorflow-gpu version: 2.4.1
* keras: 2.4.3
	
## Usage
To run the project follow these steps:

```
python3 train_binary_hybrid.py
python3 train_multiclass_hybrid.py

```
which will train the hybrid models. 
Then run,

```
python3 test_binary_hybrid.py
python3 test_multiclass_hybrid.py

```
to test the hybrid models. <br/>

<br/>

Models Used for comparison

* ANN - neural.py
* LSTM - LSTM.py
* CNN - cnn.py
* BiLSTM - BiLSTM.py

<br/>
To create the dataset by incorporating hypernyms run,

```
python3 preprocess.py

```

## Authors

Contact us in case of any queries

* Sashank Sridhar - sashank.ssridhar@gmail.com
* Sowmya Sanagavarapu - sowmya.ssanagavarapu@gmail.com

