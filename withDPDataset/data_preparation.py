import pandas as panda
import re
import numpy as np  # linear algebra
import string
import gensim
import nltk
# nltk.download() # to download stopwords corpus
from nltk.corpus import stopwords
# stopwords = nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import glob
import collections

from sklearn import svm

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split


def plot_decision_function(classifier, sample_weight, axis, title):
    # plot the decision function
    xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot the line, the points, and the nearest vectors to the plane
    axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    axis.scatter(X[:, 0], X[:, 1], c=y, s=100 * sample_weight, alpha=0.9,
                 cmap=plt.cm.bone, edgecolors='black')

    axis.axis('off')
    axis.set_title(title)


def changes_to_list(files, list):
    for f in files:
        list.append(f)
    return list


def single_email(emails):
    for i in range(len(emails)):
        f = open(emails, "r")
        contents = f.read()
    print (contents)
    # clean_data(contents)
    return contents


def multiple_email(emails):
    contents_emails = []
    for i in range(len(emails)):
        f = open(emails[i], "r")
        contents = f.read()
        contents_emails.append(contents)
    # print (contents)
    # clean_data(contents)
    return contents


# f = open(emails, "r")
def remove_stop_words(emails):
    stop_words = set(stopwords.words('english'))
    # for i in range(len(emails)):
    # f = open(emails, "r")
    # contents = f.read()
    # print(contents)
    # word_tokens = word_tokenize(emails)
    word_tokens = word_tokenize(emails)
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
    numbers = set(string.digits)
    no_numbers = []
    for word in words:
        if word not in numbers:
            no_numbers.append(word)
    return no_numbers


def data_prep(emails):
    no_stopwords = remove_stop_words(emails)
    no_punc = remove_punctuation(no_stopwords)
    no_numbers = remove_numbers(no_punc)
    data = lemmatation(no_numbers)
    # frequency(data)
    return data


def frequency(data):
    c = collections.Counter(data)
    cnt = collections.Counter()
    for word in data:
        cnt[word] += 1
    print(len(c))
    print(len(c.items()))
    print(c.values())

    return c
    '''
    print ("The most common words are", c.most_common())
    print (len(c))
    print (c[1][1])
    return c
    '''


'''
Part 3


def feature_exectration():
  '''



def support_vector_machine(data, is_spam):
    train_data, test_data, train_is_spam, test_is_spam = train_test_split(data, is_spam, test_size=0.2)

    print(train_data)
    print(train_is_spam)
    clf = svm.SVC(kernel='linear')
    clf.fit(train_data, train_is_spam)
    # plot_decision_function(train_spam_set, train_non_spam_set, test_spam_set , test_non_spam_set , clf)

    # Accuracy
    clf_predictions = clf.predict(test_data)
    print("Accuracy: {}%".format(clf.score(test_data, test_is_spam) * 100))


def get_common_words():
    output = []
    with open("annaFile30.txt", "r") as file:
        for line in file:
            output.extend(word_tokenize(line))
    return output

def feature_extraction(test_spam, common_words):
    data = []
    is_spam = []
    for email in test_spam[0:3]:
        data_sample = [0] * 30000
        email_contents = multiple_email([email])
        test_spam_data = data_prep(email_contents)
        dictionary = frequency(test_spam_data)
        print (dictionary)
        i = 0
        for j in dictionary.values():
            i +=j
        for key, value in dictionary.items():
            if key in common_words:
                value = value/i
                index = common_words.index(key)
                data_sample[index] = value
        data.append(data_sample)
        is_spam.append(-1)
    return data, is_spam


common_words = get_common_words()

test_spam_files = sorted(glob.glob('spam-non-spam-dataset/test-mails/spmsgc*.txt'))
test_spam = []
test_spam = changes_to_list(test_spam_files, test_spam)

data, is_spam = feature_extraction(test_spam, common_words)

test_non_spam_files = sorted(glob.glob('spam-non-spam-dataset/test-mails/*-*msg*.txt'))
test_non_spam = []
test_non_spam = changes_to_list(test_non_spam_files, test_non_spam)
test_non_spam_data = data_prep(test_non_spam)

data_non_spam, is_non_spam = feature_extraction(test_non_spam, common_words )

data.extend(data_non_spam)
is_spam.extend(is_non_spam)
support_vector_machine(data, is_spam)






# train_set = [
#     [0, 1],
#     [1, 1],
#     [0, 0],
#     [1, 0],
#     [0, 1],
#     [1, 1],
#     [0, 0],
#     [1, 0],
#     [0, 1],
#     [1, 1],
#     [0, 0],
#     [1, 0],
#     [0, 1],
#     [1, 1],
#     [0, 0],
#     [1, 0],
#              ]
# train_answer_set = [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1]
#
# test_set = train_set
# test_answer_set = train_answer_set
# clf = svm.SVC(kernel='linear')
# clf.fit(train_set, train_answer_set)
# # plot_decision_function(train_set, train_answer_set, test_set, test_answer_set , clf)
# print (clf.predict([[0, 0]]))
#
# support_vector_machine(train_set, train_answer_set)
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
