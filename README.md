# FAN vs Transfomer
## Requirements
- Python 3
- PyTorch 0.4

## Data
First download the data from Linzen's [paper](https://arxiv.org/abs/1611.01368)
```
$ ./download_data.sh
```
Then split the data into train/dev/test to `expr_data` directory.
```
$ python split_data.py data/agr_50_mostcommon_10K.tsv expr_data
```

Create dictionary
```
$ python build_dictionary.py expr_data
```
This script will write a `vocab.pkl` file to `expr_data`

## Language Model
```
DATA=~/path/to/data
EXPR=~/path/to/save_models
```

Train FAN LM with `-arch fan`
```
$ python lm.py -train $DATA/train.tsv -valid $DATA/valid.tsv \
-dict $DATA/vocab.pkl -word_vec_size 32 -rnn_size 32 -optim adam -layers 2 -epochs 50 -dropout 0.2 \
-batch_size 16 -n_words -1 -tied -save_model $EXPR/lm_fan.pt  -arch fan -lr 0.001 -num_heads 2
```

## Verb Prediction

```
$ python vp.py -train $DATA/train.tsv -valid $DATA/valid.tsv \
    -dict $DATA/vocab.pkl -word_vec_size 32 -rnn_size 32 -optim adam -layers 2 -epochs 50 -dropout 0.2 \
    -batch_size 16 -n_words -1 -save_model $EXPR/vp_fan.pt  -arch rnn -lr 0.001 -num_heads 2
```
## Logical Inference
First, we generate the data for logical inference task
```
$ cd propositionallogic
$ python generate_neg_set_data.py
```
The script is a modified version of Bowman's [code](https://github.com/sleepinyourhat/vector-entailment/tree/master/propositionallogic)
```

$ python logic.py -word_vec_size 32 -rnn_size 32 -optim adam -layers 2 -epochs 50 -dropout 0.2  -batch_size 64 -data_dir $DATA/propositionallogic/ \
    -save_model $EXPR/logic_fan.pt  -arch fan -lr 0.001  -max_bin 13  -report_every 100 -num_heads 2
```
