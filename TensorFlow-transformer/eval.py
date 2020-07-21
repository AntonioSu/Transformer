from __future__ import print_function
import codecs
import os

import tensorflow as tf
import numpy as np

from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_chi_vocab, load_en_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu
from  utils import del_chin_sign
import jieba
from nlgeval import compute_metrics
def eval():
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")

    # Load data
    X, Sources, Targets = load_test_data()
    chi2idx, idx2chi = load_chi_vocab()
    # Start session
    with g.graph.as_default():
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            ## Get model name
            mname = open(hp.logdir + '/checkpoint', 'r').read().split('"')[1] # model name
            ## Inference
            if not os.path.exists('results'): os.mkdir('results')

            tar=open('got.txt','w')
            with codecs.open("results/" + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // hp.batch_size):

                    ### Get mini-batches
                    x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
                    sources = Sources[i * hp.batch_size: (i + 1) * hp.batch_size]
                    targets = Targets[i * hp.batch_size: (i + 1) * hp.batch_size]

                    ### Autoregressive inference
                    preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]

                    for source, target, pred in zip(sources, targets, preds):  # sentence-wise
                        got = "".join(idx2chi[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write(target + "\n")
                        tar.write(got + "\n")
                        #fout.flush()

                #     ### Write to file
                #     for source, target, pred in zip(sources, targets, preds):  # sentence-wise
                #         got = "".join(idx2chi[idx] for idx in pred).split("</S>")[0].strip()
                #         fout.write("- source: " + source + "\n")
                #         fout.write("- expected: " + target + "\n")
                #         fout.write("- got: " + got + "\n\n")
                #         fout.flush()
                #
                #         # bleu score
                #         ref=[]
                #         hypothesis=[]
                #         text=jieba.cut(target)
                #         for word in text:
                #             ref+=[word]
                #
                #         text = jieba.cut(got)
                #         for word in text:
                #             hypothesis += [word]
                #
                #         if len(ref) > 3 and len(hypothesis) > 3:
                #             list_of_refs.append([ref])
                #             hypotheses.append(hypothesis)
                #
                # ## Calculate bleu score
                # score = corpus_bleu(list_of_refs, hypotheses)
                #s=compute_metrics(ref,hypotheses)
                #fout.write("Bleu Score = " + str(100 * score))
                #fout.write(s)


if __name__ == '__main__':
    eval()
    print("Done")