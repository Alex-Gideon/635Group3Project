# 635Group3Project

Initial Commit Grabbed from Code Ocean.. Adapted Dockerfile -> Conda environment.

# Create Conda environment:

```shell
conda env create -f environment.yml
conda activate 635ProjectGroup3
```
This may take 10-15 minutes.

# Don't have Conda?
On Linux run:
Run:
```shell
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

Aftewards run:

```shell
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

If you aren't on linux, follow the quick command line install links at the bottom of this page.
https://docs.anaconda.com/free/miniconda/index.html#latest-miniconda-installer-links


# Preprocessing Data
All extra data that we used for this project is small enough to be version controlled, including all preprocessed files.

Financial news is processed at runtime, the twitter dataset is processed prior. You can redo this preprocessing via running the notebook: experiment-1/tweets/preprocessingtweets.ipynb

All data should be version controlled and therefore does not need to be re-processed. 

# Data and Model downloads
The data is too large to be stored on git. You must download it manually from my super cool awesome google drive. Wow!

https://drive.google.com/drive/folders/1JATZ8WxXtZSCohncTbGh3T1IZhap6XMb?usp=sharing

(If you dont have access please let us know)

It contains 3 folders. 

1) Data must go into the SOURCE directory.

2) Move the contents from the folder named:  Experiment 1 Models into experiment-1/models

3) Move the contents from the foler named: Experiment 2 Models into experiment-2/models

## Can't Download? Train the models manually:


The original model can be trained with: 
```shell
python3 code/train_binary_hybrid.py
```

In order to generated the binary_hybrid_larger.h5 model, run this model is identical to the original papers however it is trained on the larger subset of data and tested on the smaller one, unlike the original papers.
```shell
python3 experiment-1/train_hybrid_binary_larger.py
``` 
it will save the model to: results/binary_hybrid_larger.h5 move the .h5 file into the experiment-1/models folder.

In order to get ready for experiment 2... 
Run the notebook: experiment-2/gensim_models.ipynb
Then run: 
```shell
python3 experiment-2/train_at_lstm.py
```

All of these may take 10 minutes to run each.

# Running the models and Experiments

## Original model:

Train:

```shell
python3 code/train_binary_hybrid.py
```

Test:

```shell
python3 code/testBinaryHybrid.py
```

## Experiment 1

### Twitter 

May take 3-4 minutes to run.

```shell
python3 experiment-1/tweets/test-ensemble-model.py
```

Results saved to:  experiment-1/results/binary_tweets_hybrid_classified.png

### Financial News

May take 1-2 minutes to run.

```shell
python3 experiment-1/financial/test-ensemble-model.py
```
Results saved to: experiment-1/financial/results/ensemble


## Experiment 2

### AT-LSTM Training and Validation on Movie Reviews:

This will run AND train the AT-LSTM model on movie reviews. Results saved to: experiment-2/results/atlstm_validation_confusion_matrix.png

```shell
python3 experiment-2/train_at_lstm.py
```
Results saved to: experiment-2/results/atlstm_validation_confusion_matrix.png


### Twitter Dataset

Tests the AT-LSTM on twitter dataset. 10,000 random entries.

May take 3-5 minutes to run.

```shell
python3 experiment-2/test-atlstm-model_tweets.py
```

### Financial News Dataset

Tests the AT-LSTM on Financial News dataset.

May take 3-5 minutes to run.

```shell
python3 experiment-2/test-atlstm-model_financial.py
```


