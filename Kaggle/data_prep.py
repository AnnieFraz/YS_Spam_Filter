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
    return [data_prep(doc).split(',') for doc in data]

def create_dictionary(data):
    #dic = gensim.corpora.Dictionary([data.split()])
    #print (dic)
    #print (data[0:100])
    dictionary = gensim.corpora.Dictionary(data)
    dictionary.filter_extremes(no_below=20, no_above=0.1)
    #dictionary.filter_extremes(20, 0.1)
    print (dictionary)
    return dictionary