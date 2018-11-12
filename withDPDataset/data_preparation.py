import collections
import glob
import string

from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn import naive_bayes
from sklearn.naive_bayes import MultinomialNB
import numpy as np

from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
#import matplotlib.pyplot as plt

#Author: Anna Frances Rasburn
#Python 3.6.4

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

    return contents


def remove_stop_words(emails):
    stop_words = set(stopwords.words('english'))
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
    return c


def support_vector_machine(data, is_spam):
    train_data, test_data, train_is_spam, test_is_spam = train_test_split(data, is_spam, test_size=0.2)
    param_support_vector_machine(train_data, test_data, train_is_spam, test_is_spam)


def param_support_vector_machine(train_data, test_data, train_is_spam, test_is_spam):
    print("Training data size: ", len(train_data))
    print("Test data size: ", len(test_data))
    clf = svm.LinearSVC()
    clf.fit(train_data, train_is_spam)
    y_score = clf.decision_function(test_data)
    precision, recall, F1 = measurement_metric(test_is_spam, y_score)
    print("Accuracy of SVM: {}%".format(clf.score(test_data, test_is_spam) * 100))

def naive_bayes(train_data, test_data, train_is_spam, test_is_spam):
    #list_alpha = np.arange(1 / 100000, 20, 0.11)
    #for alpha in list_alpha:
    bayes = MultinomialNB()
    bayes.fit(train_data, train_is_spam)
    print("Accuracy of Naive Bayes: {}%".format(bayes.score(test_data, test_is_spam) * 100))

def measurement_metric(test_is_spam, y_score):
    average_precision = average_precision_score(test_is_spam, y_score)
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision))
    precision, recall, _ = precision_recall_curve(test_is_spam, y_score)
    print (precision)
    F1 = 2 * (precision * recall) / (precision + recall)

    '''
        step_kwargs = ({'step': 'post'}
                       if 'step' in signature(plt.fill_between).parameters
                       else {})
        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2,where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
                .format(average_precision))
                '''
    return precision, recall, F1

def get_common_words():
    output = []
    with open("commonWords30k.txt", "r") as file:
        for line in file:
            output.extend(word_tokenize(line))
    return output


def feature_extraction(emails_list, common_words, spamClassifier):
    data = []
    is_spam = []
    for email in emails_list:
        data_sample = [0] * 30000
        email_contents = multiple_email([email])
        test_spam_data = data_prep(email_contents)
        dictionary = frequency(test_spam_data)
        i = 0
        for j in dictionary.values():
            i +=j
        for key, value in dictionary.items():
            if key in common_words:
                value = value/i
                index = common_words.index(key)
                data_sample[index] = value
        data.append(data_sample)
        is_spam.append(spamClassifier)
    return data, is_spam


def main():
    common_words = get_common_words()

    test_spam_files = sorted(glob.glob('spam-non-spam-dataset/test-mails/spmsgc*.txt'))
    print ("Test Spam files found: ", len(test_spam_files))
    test_spam = []
    test_spam = changes_to_list(test_spam_files, test_spam)
    test_spam_data, test_spam_is_spam = feature_extraction(test_spam, common_words, 1)
    print ("Test Spam processed")

    test_non_spam_files = sorted(glob.glob('spam-non-spam-dataset/test-mails/*-*msg*.txt'))
    print ("Test Non-Spam files found: ", len(test_non_spam_files))
    test_non_spam = []
    test_non_spam = changes_to_list(test_non_spam_files, test_non_spam)
    test_not_spam_data, test_not_spam_is_spam = feature_extraction(test_non_spam, common_words, -1)
    print ("Test Non-Spam processed")

    train_spam_files = sorted(glob.glob('spam-non-spam-dataset/train-mails/spmsg*.txt'))
    print ("Train Spam files found: ", len(train_spam_files))
    train_spam = []
    train_spam = changes_to_list(train_spam_files, train_spam)
    train_spam_data, train_spam_is_spam = feature_extraction(train_spam, common_words, 1)
    print ("Train Spam processed")

    train_non_spam_files = sorted(glob.glob('spam-non-spam-dataset/train-mails/*-*msg*.txt'))
    print ("Train Non-Spam files found: ", len(train_non_spam_files))
    train_non_spam = []
    train_non_spam = changes_to_list(train_non_spam_files, train_non_spam)
    train_not_spam_data, train_not_spam_is_spam = feature_extraction(train_non_spam, common_words, -1)
    print ("Train Non-Spam processed")

    train_data = []
    train_is_spam = []

    train_data.extend(train_spam_data)
    train_data.extend(train_not_spam_data)
    train_is_spam.extend(train_spam_is_spam)
    train_is_spam.extend(train_not_spam_is_spam)

    test_data = []
    test_is_spam = []

    test_data.extend(test_spam_data)
    test_data.extend(test_not_spam_data)
    test_is_spam.extend(test_spam_is_spam)
    test_is_spam.extend(test_not_spam_is_spam)

    param_support_vector_machine(train_data, test_data, train_is_spam, test_is_spam)
    naive_bayes(train_data, test_data, train_is_spam, test_is_spam)



main()

