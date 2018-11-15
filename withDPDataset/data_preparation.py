import collections
import glob
import string
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import naive_bayes
from sklearn import svm
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

# Author: Anna Frances Rasburn
# Python 3.6.4

#Function that receieves a set of files and converts them to a list
def changes_to_list(files, list):
    for f in files:
        list.append(f)
    return list

#Function to gather the contents from  multiple emails
def multiple_email(emails):
    contents_emails = []
    for i in range(len(emails)):
        f = open(emails[i], "r")
        contents = f.read()
        contents_emails.append(contents)
    return contents

#Function that removes stopwords and adds non stop words to a list
def remove_stop_words(emails):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(emails)
    no_stopwords = []
    for w in word_tokens:
        if w not in stop_words:
            no_stopwords.append(w)
    return no_stopwords

#Function that removes punctuation and adds non punctuation to a list
def remove_punctuation(words):
    punctuation = set(string.punctuation)
    no_punc = []
    for word in words:
        if word not in punctuation:
            no_punc.append(word)
    return no_punc

#Function that lemmatisers  and adds lemmas to a list
def lemmatation(list):
    normalized = []
    lemma = WordNetLemmatizer()
    for word in list:
        normalized.append(lemma.lemmatize(word).lower())
    return normalized

#Function that removes numbers and adds non numbers to a list
def remove_numbers(words):
    numbers = set(string.digits)
    no_numbers = []
    for word in words:
        if word not in numbers:
            no_numbers.append(word)
    return no_numbers

#This calls the above methods to prep the data before its processed
def data_prep(emails):
    no_stopwords = remove_stop_words(emails)
    no_punc = remove_punctuation(no_stopwords)
    no_numbers = remove_numbers(no_punc)
    data = lemmatation(no_numbers)
    return data

#Finds the frequency of the words from all the words in the emails
def create_dictionary(data):
    c = collections.Counter(data)
    cnt = collections.Counter()
    for word in data:
        cnt[word] += 1
    return c

#The SVM Classifier preparation whuch gets a train, test split
def support_vector_machine(data, is_spam):
    train_data, test_data, train_is_spam, test_is_spam = train_test_split(data, is_spam, test_size=0.2)
    param_support_vector_machine(train_data, test_data, train_is_spam, test_is_spam)

#This is where the SVM takes place. It is a linear SVM
def param_support_vector_machine(train_data, test_data, train_is_spam, test_is_spam):
    print("")
    svm_classifier = svm.LinearSVC()
    svm_classifier.fit(train_data, train_is_spam)
    predict = svm_classifier.predict(test_data)
    print ("SVM RESULTS")
    precision, recall, F1 = measurement_metric(test_is_spam, predict)
    #Calculation of accuracy
    print("Accuracy of SVM: {}%".format(svm_classifier.score(test_data, test_is_spam) * 100))
    return precision, recall, F1

#This is the naive bayes classifier. It is a multinomial naive bayes
def naive_bayes(train_data, test_data, train_is_spam, test_is_spam):
    print("")
    bayes = MultinomialNB()
    bayes.fit(train_data, train_is_spam)
    predict = bayes.predict(test_data)
    print ("NAIVE BAYES RESULTS")
    precision, recall, F1 = measurement_metric(test_is_spam, predict)
    # Calculation of accuracy
    print("Accuracy of Naive Bayes: {0:0.2f}%".format(bayes.score(test_data, test_is_spam) * 100))
    return precision, recall, F1

def random_forest(train_data, test_data, train_is_spam, test_is_spam):
    print("")
    random_forest = RandomForestClassifier(n_estimators=10)
    random_forest.fit(train_data, train_is_spam)
    predict = random_forest.predict(test_data)
    print ("RANDOM FOREST RESULTS")
    precision, recall, F1 = measurement_metric(test_is_spam, predict)
    # Calculation of accuracy
    print("Accuracy of Random Forest: {0:0.2f}%".format(random_forest.score(test_data, test_is_spam) * 100))
    return precision, recall, F1

#This methods works out the precision, recall and F1 of the above classifiers
def measurement_metric(test_is_spam, predict):
    average_precision = average_precision_score(test_is_spam, predict)
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision))
    precision = precision_score(test_is_spam, predict)
    recall = recall_score(test_is_spam, predict)
    print ("Precision: ", precision)
    print ("Recall: ", recall)
    F1 = 2 * (precision * recall) / (precision + recall)
    print("F1:", F1)
    return precision, recall, F1

#A file that contains the 30,000 most common english words and adds this to a list
def get_common_words():
    output = []
    with open("commonWords30k.txt", "r") as file:
        for line in file:
            output.extend(word_tokenize(line))
    return output

#This is where we create a dictionary for each email and compares the dictionary values to the most common words
#Spam classifier: -1 for Spam and 1 for Non Spam
def feature_extraction(emails_list, common_words, spamClassifier):
    data = []
    is_spam = []
    for email in emails_list:
        data_sample = [0] * 30000
        email_contents = multiple_email([email])
        test_spam_data = data_prep(email_contents)
        dictionary = create_dictionary(test_spam_data)
        i = 0
        for j in dictionary.values():
            i += j
        for key, value in dictionary.items():
            if key in common_words:
                value = value / i
                index = common_words.index(key)
                data_sample[index] = value
        data.append(data_sample)
        is_spam.append(spamClassifier)
    return data, is_spam

#This plots the precison, recall and F1 of the classifiers
def plot_results(SVM_precision, SVM_recall, SVM_F1, NB_precision, NB_recall, NB_F1, rf_precision, rf_recall, rf_F1):
    n_groups = 3
    results_svm = (SVM_precision, SVM_recall, SVM_F1)
    results_nb = (NB_precision, NB_recall, NB_F1)
    results_rf = (rf_precision, rf_recall, rf_F1)

    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.25
    opacity = 0.8

    rects1 = plt.bar(index - 0.25, results_svm, bar_width, alpha=opacity, color='r', label='SVM')
    rects2 = plt.bar(index, results_nb, bar_width, alpha=opacity, color='b', label='Naive Bayes')
    rects3 = plt.bar(index + 0.25, results_rf, bar_width, alpha=opacity, color='g', label='Random Forest')

    plt.title('Measurement Metrics for Different Classifiers')
    plt.xlabel('Measurement Metrics')
    plt.ylabel('Score')
    plt.xticks(index + bar_width - 0.25, ('Precision', 'Recall', 'F1'))
    plt.legend()
    plt.tight_layout()
    plt.show()

#Main Function
def main():
    print("**********************************")
    print("***** SPAM FILTER Assignment *****")
    print("**********************************")
    print("")
    common_words = get_common_words()

    print("*** Pre Processing ***")
    #TEST SPAM DATA
    test_spam_files = sorted(glob.glob('spam-non-spam-dataset/test-mails/spmsgc*.txt'))
    print ("Test Spam files found: ", len(test_spam_files))
    test_spam = []
    test_spam = changes_to_list(test_spam_files, test_spam)
    test_spam_data, test_spam_is_spam = feature_extraction(test_spam, common_words, 1)
    print ("Test Spam processed")

    # TEST NON SPAM DATA
    test_non_spam_files = sorted(glob.glob('spam-non-spam-dataset/test-mails/*-*msg*.txt'))
    print ("Test Non-Spam files found: ", len(test_non_spam_files))
    test_non_spam = []
    test_non_spam = changes_to_list(test_non_spam_files, test_non_spam)
    test_not_spam_data, test_not_spam_is_spam = feature_extraction(test_non_spam, common_words, -1)
    print ("Test Non-Spam processed")

    # TRAIN SPAM DATA
    train_spam_files = sorted(glob.glob('spam-non-spam-dataset/train-mails/spmsg*.txt'))
    print ("Train Spam files found: ", len(train_spam_files))
    train_spam = []
    train_spam = changes_to_list(train_spam_files, train_spam)
    train_spam_data, train_spam_is_spam = feature_extraction(train_spam, common_words, 1)
    print ("Train Spam processed")

    # TRAIN NON SPAM DATA
    train_non_spam_files = sorted(glob.glob('spam-non-spam-dataset/train-mails/*-*msg*.txt'))
    print ("Train Non-Spam files found: ", len(train_non_spam_files))
    train_non_spam = []
    train_non_spam = changes_to_list(train_non_spam_files, train_non_spam)
    train_not_spam_data, train_not_spam_is_spam = feature_extraction(train_non_spam, common_words, -1)
    print ("Train Non-Spam processed")

    #Creation of lists
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

    print("Training data size: ", len(train_data))
    print("Test data size: ", len(test_data))

    #Classifiers
    print("")
    print("*** Classifiers ***")
    SVM_precision, SVM_recall, SVM_F1 = param_support_vector_machine(train_data, test_data, train_is_spam, test_is_spam)
    NB_precision, NB_recall, NB_F1 = naive_bayes(train_data, test_data, train_is_spam, test_is_spam)
    rf_precision, rf_recall, rf_F1 = random_forest(train_data, test_data, train_is_spam, test_is_spam)
    #Plot the measurement metrics results.
    plot_results(SVM_precision, SVM_recall, SVM_F1, NB_precision, NB_recall, NB_F1, rf_precision, rf_recall, rf_F1)


main()
