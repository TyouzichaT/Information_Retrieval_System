import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import numpy as np
import timeit
from gensim.models import Word2Vec
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


if __name__ == "__main__":
    start = timeit.default_timer()

    # read training data
    print('read train data........................')
    train_data = pd.read_csv('train_data.tsv', sep='\t',header=0)
    train_passage_distinct = train_data.drop_duplicates(subset='pid',keep='first',inplace=False).reset_index().loc[:,['pid','passage']]
    train_query_distinct = train_data.drop_duplicates(subset='qid',keep='first',inplace=False).reset_index().loc[:,['qid','queries']]

    # preprocess training data
    print('preprocessing train data........................')
    train_passage_distinct['p_tokens']= np.vectorize(preprocess, otypes=[list])(train_passage_distinct['passage'])
    train_query_distinct['q_tokens']=np.vectorize(preprocess, otypes=[list])(train_query_distinct['queries'])

    # read validation data
    print('read preprocessed validation data........................')
    with open("vali_tokens.pkl", 'rb') as file:
        vali_data = pickle.load(file)

    vali_passage_distinct = vali_data.drop_duplicates(subset='pid',keep='first',inplace=False).reset_index().loc[:,['pid','p_tokens']]

    vali_query_distinct = vali_data.drop_duplicates(subset='qid',keep='first',inplace=False).reset_index().loc[:,['qid','q_tokens']]


    # embedding_train
    embdedding_train = pd.concat([train_passage_distinct['p_tokens'],train_query_distinct['q_tokens'], vali_passage_distinct['p_tokens'], vali_query_distinct['q_tokens']])

    #train w2v
    print('Training W2V........................')
    W2V = Word2Vec(embdedding_train, negative=5, hs=0, sg=1,vector_size=100, window=5, min_count=1, workers=7)

    print("Time consumed {0} s".format(timeit.default_timer()-start))

    W2V.save('word2vec.model')

