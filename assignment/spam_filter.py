
import pandas as panda
import re

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

import string

import gensim


import nltk
from nltk.corpus import stopwords
stopwords = nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer


#STEPS

def get_data(filename):
    emails = panda.read_csv(filename)
    #print (emails)

    cols_to_clean = []
    exclude = [[]]

    for i, col in enumerate(cols_to_clean):
        exclude_pattern = re.compile('|'.join(exclude[i]))
        emails = emails[emails[col].str.contains(exclude_pattern) == False]

    #print (cols_to_clean)
    #print (exclude)

    #emails = remove_duplicates(emails)

    return emails

def remove_duplicates(data):
    processed = set()
    result = []
    pattern = re.compile('X-FileName: .*')
    pattern2 = re.compile('X-FileName: .*?  ')

    for doc in data:
        doc = doc.replace('\n', ' ')
        doc = doc.replace(' .*?nsf', '')
        match = pattern.search(doc).group(0)
        match = re.sub(pattern2, '', match)

        if match not in processed:
            processed.add(match)
            result.append(match)

    return result



'''
1. Preparation of Data
    a. Remove all Stop works e.g. 'and', 'of', 'the' etc
    b. Lemmatation - reducing words to original form. Remove suffix and prefix e.g. Include, Includes, Included --> Include
    c. Remove Punctionation

    LIBRARIES AVALIABLE FOR ALL OF THESE
    MORE IMPORTANT TO LEARN THESE PROCESSES
'''

def remove_stop_words(emails):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(emails)
    clean_code = [w for w in word_tokens if not w in stop_words]
    clean_code = []
    for w in word_tokens:
        if w not in stop_words:
            clean_code.append(w)
    print (word_tokens)
    print (clean_code)
    remove_punctuation(clean_code)
    return clean_code


def remove_punctuation(clean_code):
    punctuation = set(string.punctuation)
    words = [ch for ch in clean_code if ch not in punctuation]
    lemmatizer(words)
    return words


def lemmatizer(words):
    lemma = WordNetLemmatizer()
    normalised = [lemma.lemmatize(word)for word in words.split()]
    return normalised

'''
2. Create word dictionary
    - Each word has a frequency (How often the word is used)
'''

def create_dictionary(training_set):
    dictionary = gensim.corpora.Dictionary(training_set)
    dictionary.filter_extremes(no_below=20, no_above=0.1)

    return dictionary

'''
3. Feature Execution
    -For each email you want a set of data e.g. create a 30,000 for commonly used words
'''

'''
4. ML Classifiers
    -SVM
    -And another one
    -`Use body of the email
'''

'''
5. Accuracy 
    -How do you measure this?
    -Use Libraries
'''

#######MAIN#######

emails = get_data("emails.csv")

#emails = remove_duplicates(emails)

print (len(emails)) #Total Size is  517401 - TODO: need to calculate for duplicates

email_bodies = emails.message.as_matrix()

clean_code = remove_stop_words(email_bodies)

no_punctionation = remove_punctuation(clean_code)

words = lemmatizer(no_punctionation)

words_training_set = words[0:1000]

dictionary = create_dictionary(words_training_set)

emails_training_set = email_bodies[0:362181]  #70% of 517401 is

emails_testing_set = email_bodies[362181:]