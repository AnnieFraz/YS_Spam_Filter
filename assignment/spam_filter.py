
import pandas as panda
import re

import numpy as np # linear algebra
import pandas as panda # data processing, CSV file I/O (e.g. pd.read_csv)
import re

import string

import gensim


import nltk
#nltk.download() # to download stopwords corpus
from nltk.corpus import stopwords
#stopwords = nltk.download('stopwords')
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
    return clean_code


def remove_punctuation(clean_code):
    punctuation = set(string.punctuation)
    words = [ch for ch in clean_code if ch not in punctuation]
    return words


def lemmatizer(words):
    lemma = WordNetLemmatizer()
    normalised = [lemma.lemmatize(word)for word in words.split()]
    return normalised

'''
def clean_code(emails):
    
    excluded_stop_words = remove_stop_words(emails)
    excluded_punctuation = remove_punctuation(excluded_stop_words)
    end_result = lemmatizer(excluded_punctuation)

    return end_result


    words_to_exclude = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    word_free = " ".join([i for i in doc.lower().split() if i not in words_to_exclude])
    punc_free = ''.join(ch for ch in word_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

    return normalized
'''

def clean(doc):
    words_to_exclude = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    word_free = " ".join([i for i in doc.lower().split() if i not in words_to_exclude])
    #word_free = remove_stop_words(doc)
    print ("WORD FREE")
    print (word_free)
    punc_free = ''.join(ch for ch in word_free if ch not in exclude)
    print ("PUNC")
    print (punc_free)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())

    print ("NORMALIZED")
    print (normalized)
    return normalized

def clean_data(data):
    return [clean(doc).split(' ') for doc in data]



'''
2. Create word dictionary
    - Each word has a frequency (How often the word is used)
'''

def create_dictionary(training_set):
    dictionary = gensim.corpora.Dictionary(training_set)
    dictionary.filter_extremes(no_below=20, no_above=0.1)

    matrix = [dictionary.doc2bow(doc) for doc in training_set]

    tfidf_model = gensim.models.TfidfModel(matrix, id2word=dictionary)
    lsi_model = gensim.models.LsiModel(tfidf_model[matrix], id2word=dictionary, num_topics=100)

    topics = lsi_model.print_topics(num_topics=100, num_words=10)
    for topic in topics:
        print(topic)

    return matrix

def word_frequency():

'''
3. Feature Execution
    -For each email you want a set of data e.g. create a 30,000 for commonly used words
'''
#def feature_execution():


'''
4. ML Classifiers
    -SVM
    -And another one
    -`Use body of the email
'''

#def svm_classifer():

#def

'''
5. Accuracy 
    -How do you measure this?
    -Use Libraries
'''

#######MAIN#######

emails = get_data("emails.csv")



print (len(emails)) #Total Size is  517401 - TODO: need to calculate for duplicates

email_bodies = emails.message.as_matrix()
unique_emails = remove_duplicates(email_bodies)

print('There are a total of {} non-duplicate emails\n'.format(len(unique_emails))) #len - 248930, 70% is 174251
#unique_emails = remove_duplicates(emails_bodies)
#print (len(unique_emails))


training_set = clean_data(unique_emails[0:20])
testing_set = clean_data(unique_emails[20:30])


dictionary = create_dictionary(training_set)

print (dictionary)
'''
emails_training_set = email_bodies[0:362181]  #70% of 517401 is

emails_testing_set = email_bodies[362181:]

'''
