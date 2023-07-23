import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
from collections import Counter
import numpy as np
import timeit
import pickle

stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))
pattern = r'\w+'
tokenizer = RegexpTokenizer(pattern)

def preprocess(text, stopword_removal=True):

    # Normalization and tokenlization - lowercase and remove punctuation
    tokens = tokenizer.tokenize(text.lower())

    # stopward_removal
    if stopword_removal == True:
        tokens = [w for w in tokens if not w in stop_words]

    # stemming
    tokens = [stemmer.stem(w) for w in tokens]

    return tokens


def inverted_index(dataframe):
    tokens = dataframe.iloc[:, 2].tolist()
    ids = dataframe.iloc[:, 0].tolist()
    vocabulary = []
    for i in range(len(tokens)):
        vocabulary += tokens[i]
    vocabulary = set(vocabulary)
    
    inverted = dict(Counter(vocabulary))
    for item in inverted:
        inverted[item]= dict()

    for i in range(len(tokens)):
        word_count = Counter(tokens[i])
        for word, number in word_count.items():
            inverted[word].update({ids[i]: number})

    return inverted



def BM25_score_toplist(dataset, inverted_passages, inverted_query, N, avdl):

    k1 = 1.2
    k2 = 100
    b = 0.75
    query_tokens_list = dataset.loc[:,'q_tokens'].tolist()
    passage_tokens_list = dataset.loc[:,'p_tokens'].tolist()
    qid_list = dataset.loc[:,'qid'].tolist()
    pid_list = dataset.loc[:,'pid'].tolist()

    scores = np.zeros(len(dataset))

    for i in range(len(dataset)):
        query_tokens = query_tokens_list[i]
        passage_tokens = passage_tokens_list[i]
        pid = pid_list[i]
        qid = qid_list[i]

        for word in query_tokens:
            if word in inverted_passages.keys():
                n = len(inverted_passages[word])
            else:
                n = 0        

            dl = len(passage_tokens)
            K = k1*((1-b)+b*dl/avdl)
            if word in passage_tokens:
                f = inverted_passages[word][pid]
            else:
                f = 0

            qf = inverted_query[word][qid]

            
            scores[i] += np.log(((0+0.5)/(0-0+0.5)) / ((n-0+0.5)/(N-n-0))
                                ) * (k1+1)*f*(k2+1)*qf / ((K+f)*(k2+qf))
            
    dataset['score'] = scores
    dataset.sort_values(by=['qid','score'],ascending=False, inplace=True)
    dataset.reset_index(drop=True, inplace= True)

    top_df = dataset.loc[:,['qid','pid','relevancy','score']]
    top_df.to_csv("bm25_toplist.csv", index=False)




if __name__ == "__main__":
    start = timeit.default_timer()

    validation_data = pd.read_csv('validation_data.tsv', sep='\t', header=0)
    

    print('passage preprocess........................')
    # passage preprocess
    validation_data['p_tokens'] = np.vectorize(
        preprocess, otypes=[list])(validation_data['passage'])
    
    passage_distinct = validation_data.drop_duplicates(
        subset='pid', keep='first', inplace=False).reset_index()[['pid', 'passage','p_tokens']]
    

    print('query preprocess........................')
    # query preprocess
    validation_data['q_tokens'] = np.vectorize(
        preprocess, otypes=[list])(validation_data['queries'])
    
    query_distinct = validation_data.drop_duplicates(
        subset='qid', keep='first', inplace=False).reset_index()[['qid', 'queries','q_tokens']]


    print('inverted dictionary generating........................')
    # inverted dictionary
    inverted_passage = inverted_index(passage_distinct)
    inverted_query = inverted_index(query_distinct)
    

    print('BM25 top list generating........................')
    # BM25
    N = len(passage_distinct)
    length = sum(passage_distinct['p_tokens'].map(len))
    avdl = length/N
    BM25_score_toplist(validation_data, inverted_passage, inverted_query, N, avdl)
    
    print("Time consumed {0} s".format(timeit.default_timer()-start))

    with open("vali_tokens.pkl", 'wb') as file:
        pickle.dump(validation_data, file)


