## A Pytorch Implementation of the Transformer: Attention Is All You Need
My implementation is largely based on [this git](https://github.com/leviswind/pytorch-transformer)
## Requirements
  * NumPy >= 1.11.1
  * Pytorch >= 0.4.0
  * nltk


## File description
  * `hyperparams.py` includes all hyper parameters that are needed.
  * `prepro.py` get source language and target language,and then creates vocabulary files for the source and the target,last save train file and test file.
  * `data_load.py` transfer sentence to ont-hot
  * `modules.py` has all building blocks for encoder/decoder networks.
  * `AttModel`  create model
  * `train.py` is for train.
  * `eval.py` is for evaluation.

## Training
* STEP 1. Adjust hyper parameters in `hyperparams.py` if necessary.
* STEP 2. Run `prepro.py` to generate vocabulary files to the `preprocessed` folder.
* STEP 3. Run `train.py` 
* STEP 4. Show loss and accuracy in tensorboard
```sh
tensorboard --logdir runs
```

## Evaluation
  * Run `eval.py`.
  
##statement
This project need cuda support
