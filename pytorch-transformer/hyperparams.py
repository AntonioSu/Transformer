
class Hyperparams:
    '''Hyperparameters'''
    # data
    source='corpora/cmn.txt'
    #all the data including the train and the test
    eng_train = 'corpora/english.txt'
    #label
    chin_train = 'corpora/chinese.txt'

    source_train = 'corpora/X_train.tsv'
    target_train = 'corpora/y_train.tsv'

    source_test = 'corpora/X_test.tsv'
    target_test = 'corpora/y_test.tsv'

    # training
    batch_size = 32 # alias = N
    lr = 0.0001 # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = 'logdir' # log directory

    # model
    maxlen = 10 # Maximum number of words in a sentence. alias = T.
                # Feel free to increase this if you are ambitious.
    min_cnt = 1 # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512 # alias = C
    num_blocks = 6 # number of encoder/decoder blocks
    num_epochs = 20
    num_heads = 8
    sinusoid=True
    preload=False
    model_dir="checkpoints"
    dropout_rate = 0.1
