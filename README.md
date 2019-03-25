# BertSum


This code is for the paper 'Fine-tune BERT for Extractive Summarization'.

**Python version**: This code is in Python3.6

## Data Preparation
### Option 1: download the processed data

### Option 2: process the data yourself


Follow steps in https://github.com/abisee/cnn-dailymail to generate CoreNLP tokenized and sentence-splitted datasets (`cnn_stories_tokenized` and `dm_stories_tokenized`), and merge them into one directory `merged_stories_tokenized`.

#### Step 1
 
```
python preprocess.py -mode format_to_lines -raw_path RAW_PATH -save_path JSON_PATH -map_path MAP_PATH -lower 
```

`RAW_PATH` is the directory containing tokenized files (`../merged_stories_tokenized`), `JSON_PATH` is the target directory to save the generated json files (`../json_data`), `MAP_PATH` is the  directory containing the urls files (`../urls`)

#### Step 2

```
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH -oracle_mode greedy -n_cpus 4 -log_file ../logs/preprocess.log
```

`JSON_PATH` is the directory containing json files (`../json_data`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`)

`-oracle_mod`e can be `greedy` or `combination`, where `combination` is more accurate but takes much longer time to process 
