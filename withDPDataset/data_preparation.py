import pandas as panda
import re
import numpy as np # linear algebra
import string
import gensim
import nltk
#nltk.download() # to download stopwords corpus
from nltk.corpus import stopwords
#stopwords = nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import glob
import collections

def changes_to_list(files, list):
    for f in files:
        list.append(f)
    return list

def single_email(emails):
    for i in range(len(emails)):
        f = open(emails, "r")
        contents = f.read()
    print (contents)
    #clean_data(contents)
    return contents

def multiple_email(emails):
    contents_emails = []
    for i in range(len(emails)):
        f = open(emails[i], "r")
        contents = f.read()
        contents_emails.append(contents)
    #print (contents)
    #clean_data(contents)
    return contents

 #f = open(emails, "r")
def remove_stop_words(emails):
    stop_words = set(stopwords.words('english'))
    #for i in range(len(emails)):
    f = open(emails, "r")
    contents = f.read()
   #print(contents)
    word_tokens = word_tokenize(contents)
    no_stopwords = []
    for w in word_tokens:
       if w not in stop_words:
           no_stopwords.append(w)
    return no_stopwords


def remove_punctuation(words):
    punctuation = set(string.punctuation)
    no_punc = []
    for word in words:
        if word not in punctuation:
            no_punc.append(word)
    return no_punc

def lemmatation(list):
    normalized = []
    lemma = WordNetLemmatizer()
    for word in list:
        normalized.append(lemma.lemmatize(word).lower())
    return normalized

def remove_numbers(words):
    numbers = set(string.numbers)
    no_numbers = []
    for word in words:
        if word not in numbers:
            no_numbers.append(word)
    return no_numbers


def data_prep(emails):
    no_stopwords = remove_stop_words(emails)
    no_punc = remove_punctuation(no_stopwords)
    data = lemmatation(no_punc)
    print (len(data))
    #frequency(data)
    return data

def clean(doc):
    words_to_exclude = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    word_free = " ".join([i for i in doc.lower().split() if i not in words_to_exclude])
    punc_free = ''.join(ch for ch in word_free if ch not in exclude)
    print (punc_free)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized

def clean_data(data):
    return [data_prep(doc).split(' ') for doc in data]

def create_dictionary(data):
    #dic = gensim.corpora.Dictionary([data.split()])
    #print (dic)
    #print (data[0:100])
    dictionary = gensim.corpora.Dictionary(data)
    dictionary.filter_extremes(no_below=20, no_above=0.1)
    #dictionary.filter_extremes(20, 0.1)
    print (dictionary)
    return dictionary

def frequency(data):
    c = collections.Counter(data)
    cnt = collections.Counter()
    for word in data:
        cnt[word] += 1
    print(len(c))
    print(len(c.items()))

    return c
    '''
    print ("The most common words are", c.most_common())
    print (len(c))
    print (c[1][1])
    return c
    '''

test_spam_files = sorted(glob.glob('spam-non-spam-dataset/test-mails/spmsgc*.txt'))
test_spam = []
test_spam = changes_to_list(test_spam_files, test_spam)
email_contents = multiple_email(test_spam)
test_spam_data = clean_data(email_contents)
dictionary = create_dictionary(test_spam_data)
#test_spam_data = data_prep(test_spam[2])


'''
test_non_spam_files = sorted(glob.glob('spam-non-spam-dataset/test-mails/*-*msg*.txt'))
test_non_spam = []
test_non_spam = changes_to_list(test_non_spam_files, test_non_spam)
test_non_spam_data = data_prep(test_non_spam)

train_spam_files = sorted(glob.glob('spam-non-spam-dataset/train-mails/spmsgc*.txt'))
train_spam = []
train_spam = changes_to_list(train_spam_files, train_spam)
train_spam_data = data_prep(train_spam_files)

train_non_spam_files = sorted(glob.glob('spam-non-spam-dataset/train-mails/*-*msg*.txt'))
train_non_spam = []
train_non_spam = changes_to_list(train_non_spam_files, train_non_spam)
train_non_spam_data = data_prep(train_non_spam_files)
'''
