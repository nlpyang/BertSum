# BertSum

**This code is for paper `Fine-tune BERT for Extractive Summarization`**(https://arxiv.org/pdf/1903.10318.pdf)

**!New: Please see our [full paper](https://arxiv.org/abs/1908.08345) with trained models**



Results on CNN/Dailymail (25/3/2019):

|  Models| ROUGE-1 | ROUGE-2 |ROUGE-L
| :---         |     :---      |         :--- |          :--- |
| Transformer Baseline   | 40.9     | 18.02    |37.17    |
| BERTSUM+Classifier     | 43.23       | 20.22    |39.60      |
| BERTSUM+Transformer     | 43.25      | 20.24    |39.63     |
| BERTSUM+LSTM     | 43.22       |  20.17    |39.59      |

**Python version**: This code is in Python3.6

**Package Requirements**: pytorch pytorch_pretrained_bert tensorboardX multiprocess pyrouge

Some codes are borrowed from ONMT(https://github.com/OpenNMT/OpenNMT-py)

## Data Preparation For CNN/Dailymail
### Option 1: download the processed data

download https://drive.google.com/open?id=1x0d61LP9UAN389YN00z0Pv-7jQgirVg6

unzip the zipfile and put all `.pt` files into `bert_data`

### Option 2: process the data yourself

#### Step 1 Download Stories
Download and unzip the `stories` directories from [here](http://cs.nyu.edu/~kcho/DMQA/) for both CNN and Daily Mail. Put all  `.story` files in one directory (e.g. `../raw_stories`)

####  Step 2. Download Stanford CoreNLP
We will need Stanford CoreNLP to tokenize the data. Download it [here](https://stanfordnlp.github.io/CoreNLP/) and unzip it. Then add the following command to your bash_profile:
```
export CLASSPATH=/path/to/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar
```
replacing `/path/to/` with the path to where you saved the `stanford-corenlp-full-2017-06-09` directory. 

####  Step 3. Sentence Splitting and Tokenization

```
python preprocess.py -mode tokenize -raw_path RAW_PATH -save_path TOKENIZED_PATH
```

* `RAW_PATH` is the directory containing story files (`../raw_stories`), `JSON_PATH` is the target directory to save the generated json files (`../merged_stories_tokenized`)


####  Step 4. Format to Simpler Json Files
 
```
python preprocess.py -mode format_to_lines -raw_path RAW_PATH -save_path JSON_PATH -map_path MAP_PATH -lower 
```

* `RAW_PATH` is the directory containing tokenized files (`../merged_stories_tokenized`), `JSON_PATH` is the target directory to save the generated json files (`../json_data/cnndm`), `MAP_PATH` is the  directory containing the urls files (`../urls`)

####  Step 5. Format to PyTorch Files
```
python preprocess.py -mode format_to_bert -raw_path JSON_PATH -save_path BERT_DATA_PATH -oracle_mode greedy -n_cpus 4 -log_file ../logs/preprocess.log
```

* `JSON_PATH` is the directory containing json files (`../json_data`), `BERT_DATA_PATH` is the target directory to save the generated binary files (`../bert_data`)

* `-oracle_mode` can be `greedy` or `combination`, where `combination` is more accurate but takes much longer time to process 

## Model Training

**First run**: For the first time, you should use single-GPU, so the code can download the BERT model. Change ``-visible_gpus 0,1,2  -gpu_ranks 0,1,2 -world_size 3`` to ``-visible_gpus 0  -gpu_ranks 0 -world_size 1``, after downloading, you could kill the process and rerun the code with multi-GPUs.


To train the BERT+Classifier model, run:
```
python train.py -mode train -encoder classifier -dropout 0.1 -bert_data_path ../bert_data/cnndm -model_path ../models/bert_classifier -lr 2e-3 -visible_gpus 0,1,2  -gpu_ranks 0,1,2 -world_size 3 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file ../logs/bert_classifier -use_interval true -warmup_steps 10000
```

To train the BERT+Transformer model, run:
```
python train.py -mode train -encoder transformer -dropout 0.1 -bert_data_path ../bert_data/cnndm -model_path ../models/bert_transformer -lr 2e-3 -visible_gpus 0,1,2  -gpu_ranks 0,1,2 -world_size 3 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file ../logs/bert_transformer -use_interval true -warmup_steps 10000 -ff_size 2048 -inter_layers 2 -heads 8
```

To train the BERT+RNN model, run:
```
python train.py -mode train -encoder rnn -dropout 0.1 -bert_data_path ../bert_data/cnndm -model_path ../models/bert_rnn -lr 2e-3 -visible_gpus 0,1,2  -gpu_ranks 0,1,2 -world_size 3 -report_every 50 -save_checkpoint_steps 1000 -batch_size 3000 -decay_method noam -train_steps 50000 -accum_count 2 -log_file ../logs/bert_rnn -use_interval true -warmup_steps 10000 -rnn_size 768 -dropout 0.1
```


* `-mode` can be {`train, validate, test`}, where `validate` will inspect the model directory and evaluate the model for each newly saved checkpoint, `test` need to be used with `-test_from`, indicating the checkpoint you want to use

## Model Evaluation
After the training finished, run
```
python train.py -mode validate -bert_data_path ../bert_data/cnndm -model_path MODEL_PATH  -visible_gpus 0  -gpu_ranks 0 -batch_size 30000  -log_file LOG_FILE  -result_path RESULT_PATH -test_all -block_trigram true
```
* `MODEL_PATH` is the directory of saved checkpoints
* `RESULT_PATH` is where you want to put decoded summaries (default `../results/cnndm`)


