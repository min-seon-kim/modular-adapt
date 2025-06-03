# Modular Model Adaptation for Online Learning in Streaming Text Classification

## Data Preparation

Datasets are used in the following file structure:

```
│adaptive-model-update/
├──data/
│  ├── cybersecurity
│  │   ├── cybersecurity_source.csv
│  │   ├── cybersecurity_target.csv
│  ├── disaster
│  │   ├── disaster_source.csv
│  │   ├── disaster_target.csv
│  ├── review
│  │   ├── review_source.csv
│  │   ├── review_target.csv
│  ├── socialmedia
│  │   ├── socialmedia_source.csv
│  │   ├── socialmedia_target.csv
```

- `cs_source.csv`: You can download it from: [here](https://github.com/behzadanksu/cybertweets)
- `cs_target.csv`: You can download it from: [here](https://github.com/ndionysus/multitask-cyberthreat-detection)
- `disaster_source.csv`: You can download it from: [here](https://www.kaggle.com/competitions/nlp-getting-started/data)
- `disaster_target.csv`: Please refer to emergency.csv file.
- `hotel_review.csv`: You can download it from: [here](https://www.yelp.com/dataset)
- `review_source.csv` and `review_target.csv`: You can download it from: [here](https://msnews.github.io/)
- `socialmedia_source.csv` and `socialmedia_target.csv`: You can download `RS_2019-03.zst` and `RS_2019-04.zst` from: [here](https://academictorrents.com/details/ba051999301b109eab37d16f027b3f49ade2de13/tech&hit=1&filelist=1)

## Setups

All code was developed and tested on Nvidia RTX A4000 (48SMs, 16GB) the following environment.
- Ubuntu 18.04
- python 3.6.9
- gensim 3.8.3
- keras 2.6.0
- numpy 1.19.5
- pandas 1.1.5
- tensorflow 2.6.2

## Implementation

To pre-train the model, run the following script using command line:

```shell
sh run_pretrain_offline.sh
```

To adapt the model online, run the following script using command line:

```shell
sh run_update_online.sh
```

## Hyperparameters

The following options can be passed to `main.py`
- `-dataset`: Name of the dataset. (Supported names are cybersecurity, disaster, review)
- `-model`: Neural architecture of the _OnlineClassifier_. (Supported models are CNN, LSTM, Transformer)
- `-ood_detector`: Whether to enable the _OODDetector_ module. Default is `False`
- `-adjust_weight`: Relative importance between learning efficiency and accuracy. Default is 0.5.
- `-epochs`: Epochs for training model. Deault is 20.
- `-event_size`: Size of streaming batches.
- `-batch_size`: Size of batch to train the model.
- `-keyword_size`: Size of keyword set to calculate the frequency indicator. 
- `-embedding_size`: Size of embedding layer.
- `-output_path`: Path for the output results.
- `-token_path`: Path for saving and loading tokenizer.
- `-model_path`: Path for saving and loading machine learning-based _OnlineClassifier_.
- `-ml_path`: Path for saving and loading machine learning-based _AccPredictor_.
- `-pretrain`: Execute the model pre-training in offline.
- `-update`: Execute the model update in online.  
