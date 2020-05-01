# Final Project Repo

## Overview

There were two primary scripts used for my analysis and that can be run on the command line are `finetune_bert.py` and `simple_feature_models.py`.  I go over how to set up your machine to run these scripts below.


## Dataset

Working with [this Kaggle NLP dataset of tweets](https://www.kaggle.com/c/nlp-getting-started/notebooks).  The task the data is set up for is sentiment analysis of whether or not a tweet is about a disaster.  


## Setting Up Environment

Follow these steps to have the proper environment for these scripts.

1. Create a Python virtual environment called "jack_finalproject" using `conda`

``` bash
$ conda create -n jack_finalproject pip python=3.6
$ conda activate jack_finalproject
```

2. Clone the repo and install required packages

``` bash
$ git clone https://github.com/jackhart/BERT_tuning
$ pip install -r requirements.txt
```

## Running simple_feature_models.py

The point of this script is to run multiple logistic regressions with simple feature representations of the dataset.  The script allows for holdout and 10-fold cv, and can run experiments with TfidfVectorizer, CountVectorizer, and HashingVectorizer.

As an example, to run the script with HashingVectorizer using holdout, use the following command:

``` bash
$ python simple_tfidf.py --feature_extractor HashingVectorizer
```

Also, to run the script with 10-fold CV and *not* save the results (just print them out), run the following command:

``` bash
$ python simple_tfidf.py --feature_extractor CountVectorizer --split_type cv 
```


## Running finetune_bert.py

This script requires more setup because in order for me to train specific layers in BERT I had to work directly with the model.

### Set-Up Environment for BERT Fine-Tuning

First, you need to be on a GPU ( :`( ), otherwise you'll run into memory errors and convert a lot of energy to heat.

Once on a GPU, follow these steps:

1. Clone the directory onto the GPU again if needed

``` bash
$ git clone https://github.com/jackhart/BERT_tuning
```

2. Import in the BERT-BASE model

``` bash
$ mkdir models
$ cd models
$ wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
$ unzip cased_L-12_H-768_A-12.zip
$ rm cased_L-12_H-768_A-12.zip
```


### Running the script

This script is my additions to the baseline `run_classifier.py` script Google gives in their repo to perform supervised finetuning.  I've adjusted the script to work with the Twitter data and allow for layer-specific tuning.

You can add a regular expression for teh `train_layers' parameter to parse out which layers you'd like to train.  For instance, to train a BERT model with all BERT layers frozen, run the following command:

``` bash
$ python finetune_bert.py --data_dir data \
--bert_config_file models/cased_L-12_H-768_A-12/bert_config.json \
--vocab_file models/cased_L-12_H-768_A-12/vocab.txt \
--output_dir models/bert_classifier_1 \
--init_checkpoint models/cased_L-12_H-768_A-12/bert_model.ckpt \
--do_train True
--train_layers '"^(?!bert)."'
```


After training the previous model, the checkpoint files will show up in the `output_dir`.  Plug that checkpoint file into the following command to predict:

``` bash
$ python finetune_bert.py --data_dir data \
--bert_config_file models/cased_L-12_H-768_A-12/bert_config.json \
--task_name twitter \
--do_lower_case=False \
--vocab_file models/cased_L-12_H-768_A-12/vocab.txt \
--output_dir models/bert_classifier_1 \
--init_checkpoint models/bert_classifier_1/model.ckpt-713 \ # adjust checkpoint file here
--do_predict True
```




