from sklearn.feature_extraction.text import CountVectorizer
import os
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer
import cPickle as pickle
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.neural_network import MLPClassifier
from tflearn.layers.normalization import local_response_normalization
from tensorflow.contrib import learn

max_features=5000
max_document_length=100

data_pkl_file="data-spam-mail.pkl"
label_pkl_file="label-spam-mail.pkl"

def load_one_file(filename):
    x=""
    with open(filename) as f:
        for line in f:
            line=line.strip('\n')
            line = line.strip('\r')
            x+=line
    return x

def load_files_from_dir(rootdir):
    x=[]
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isfile(path):
            v=load_one_file(path)
            x.append(v)
    return x

def load_all_files():
    ham=[]
    spam=[]
    for i in range(1,2):
        path="../data/mail/enron%d/ham/" % i
        print "Load %s" % path
        ham+=load_files_from_dir(path)
        path="../data/mail/enron%d/spam/" % i
        print "Load %s" % path
        spam+=load_files_from_dir(path)
    return ham,spam
    
def do_svm_tf(x_train, x_test, y_train, y_test):
    print "SVM and tf"
    clf = svm.SVC()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print metrics.accuracy_score(y_test, y_pred)
    print metrics.confusion_matrix(y_test, y_pred)
    
def  get_features_by_tf():
    global  max_document_length
    x=[]
    y=[]
    ham, spam=load_all_files()
    x=ham+spam
    y=[0]*len(ham)+[1]*len(spam)
    vp=tflearn.data_utils.VocabularyProcessor(max_document_length=max_document_length,
                                              min_frequency=0,
                                              vocabulary=None,
                                              tokenizer_fn=None)
    x=vp.fit_transform(x, unused_y=None)
    x=np.array(list(x))
    return x,y   

if __name__ == "__main__":
    print "Hello spam-mail"
    print "get_features_by_tf"
    x,y=get_features_by_tf()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 0)

    do_svm_tf(x_train, x_test, y_train, y_test)
