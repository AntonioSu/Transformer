#coding=utf-8
from __future__ import print_function
from hyperparams import Hyperparams as hp
import codecs
import os
import re
from collections import Counter
import  jieba
from sklearn.model_selection import train_test_split
#from utils import del_eng_sign
from utils import del_chin_sign
from utils import clear_data


#create the vocabulary
def make_vocab(fpath, fname,flag):
    '''Constructs vocabulary.

    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.

    Writes vocabulary line by line to `preprocessed/fname`
    '''
    word2cnt={}
    if flag=='eng':
        file = open(fpath, 'r').readlines()
        for line in file:
            text=line.strip().split(' ')
            for word in text:
                word2cnt[word]=word2cnt.get(word,0)+1
    else:
        file = codecs.open(fpath, 'r','utf-8').readlines()
        for line in file:
            text=jieba.cut(line.strip())
            for word in text:
                word2cnt[word]=word2cnt.get(word,0)+1

    word2cnt=list(word2cnt.items())
    word2cnt.sort(key=lambda x: x[1], reverse=True)

    if not os.path.exists('preprocessed'):
        os.mkdir('preprocessed')

    with codecs.open('preprocessed/{}'.format(fname), 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word in word2cnt:
            fout.write(u"{}\t{}\n".format(word[0], word[1]))

#data file,split file into two files,one is english,one is chinese
def split_eng_chi(path,eng,chin):
    file=codecs.open(path,'r','utf-8').readlines()
    english=open(eng,'w')
    chinese=codecs.open(chin,'w','utf-8')
    for seq in file:
        lang=seq.split('\t')
        if len(lang)==2:
            lang[0]=clear_data(lang[0])
            english.write(lang[0]+'\n')
            lang[1] = del_chin_sign(lang[1])
            chinese.write(lang[1]+'\n')


#split data into two parts,one is train,one is test
def train_test_splits(eng,chin):
    X =open(eng, 'r').readlines()
    y= codecs.open(chin, 'r', 'utf-8').readlines()
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
    save(X_train,hp.source_train)
    save(y_train, hp.target_train)
    save(X_test, hp.source_test)
    save(y_test, hp.target_test)


#save every line to the file
def save(data,path):
    file = codecs.open(path, 'w', 'utf-8')
    for x in data:
        file.write(x)

if __name__ == '__main__':
    split_eng_chi(hp.source,hp.eng_train,hp.chin_train)
    #create the vocabula
    make_vocab(hp.eng_train, "eng.tsv",'eng')
    make_vocab(hp.chin_train, "chi.tsv",'chi')
    train_test_splits(hp.eng_train,hp.chin_train)
    print("Done")
