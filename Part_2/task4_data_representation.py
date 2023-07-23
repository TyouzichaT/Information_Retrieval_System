import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer
import numpy as np
import timeit
from gensim.models import Word2Vec
import pickle
from keras import preprocessing
from keras import utils

stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))
pattern = r'\w+'
tokenizer = RegexpTokenizer(pattern)
W2V = Word2Vec.load('word2vec.model')


def preprocess(text, stopword_removal=True):

    # Normalization and tokenlization - lowercase and remove punctuation
    tokens = tokenizer.tokenize(text.lower())

    # stopward_removal
    if stopword_removal == True:
        tokens = [w for w in tokens if not w in stop_words]

    # stemming
    tokens = [stemmer.stem(w) for w in tokens]

    return tokens


def sample(dataset):
    relevant = dataset[dataset['relevancy'] > 0]
    nonrelevant = dataset[dataset['relevancy'] <= 0]

    unique_qids = nonrelevant.drop_duplicates(subset='qid', keep='first', inplace=False).reset_index(drop=True)['qid']

    train_sample = pd.DataFrame()
    for qid in unique_qids:
        nonrel_with_qid = nonrelevant[nonrelevant['qid'] == qid]
        if len(nonrel_with_qid) > 100:
            nonrel_with_qid = nonrel_with_qid.sample(100)

        train_sample = pd.concat([train_sample, nonrel_with_qid]).reset_index(drop=True)

    train_sample = pd.concat([train_sample, relevant]).sample(frac=1).reset_index(drop=True)

    return train_sample


if __name__ == "__main__":
    start = timeit.default_timer()

    # reading local trained Word2Vec
    print('reading local trained Word2Vec........................')
    W2V = Word2Vec.load('word2vec.model')
    vocab = W2V.wv.index_to_key
    
    # generate word_index dictionary for our vocabulary 
    print('generating word_index dictionary for our vocabulary........................')
    texttokenizer = preprocessing.text.Tokenizer()
    texttokenizer.fit_on_texts(vocab)

    # sample train data and preprocess
    print('sample train data and preprocess........................')
    train_data = pd.read_csv('train_data.tsv', sep='\t', header=0)
    train_sample = sample(train_data)
    train_sample['q_tokens']=np.vectorize(preprocess, otypes=[list])(train_sample['queries'])
    train_sample['p_tokens']=np.vectorize(preprocess, otypes=[list])(train_sample['passage'])
    train_label = np.array(train_sample['relevancy'])

    # generating sample data sequences with padding
    print('generating sample data sequences with padding........................')
    train_seq_passage= texttokenizer.texts_to_sequences(train_sample['p_tokens'])
    train_seq_padding_passage = utils.pad_sequences(train_seq_passage, maxlen=207, padding="post", truncating="post")

    train_seq_query= texttokenizer.texts_to_sequences(train_sample['q_tokens'])
    train_seq_padding_query = utils.pad_sequences(train_seq_query, maxlen=17, padding="post", truncating="post")

    # read preprocessed validation set
    print('reading preprocessed validation set........................')
    with open("vali_tokens.pkl", 'rb') as file:
        validation_data = pickle.load(file)

    # generating validation data sequences with padding
    print('generating validation data sequences with padding........................')
    vali_seq_passage= texttokenizer.texts_to_sequences(validation_data['p_tokens'])
    vali_seq_padding_passage = utils.pad_sequences(vali_seq_passage, maxlen=207, padding="post", truncating="post")

    vali_seq_query= texttokenizer.texts_to_sequences(validation_data['q_tokens'])
    vali_seq_padding_query = utils.pad_sequences(vali_seq_query, maxlen=17, padding="post", truncating="post")
    vali_label = np.array(validation_data['relevancy'])
    

    # read top1000 data and preprocess
    print('read top1000 data and preprocess........................')
    top1000 = pd.read_csv('candidate_passages_top1000.tsv', sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'] )
    top1000['q_tokens']=np.vectorize(preprocess, otypes=[list])(top1000['query'])
    top1000['p_tokens']=np.vectorize(preprocess, otypes=[list])(top1000['passage'])

    # generating top1000 data sequences with padding
    print('generating top1000 data sequences with padding........................')
    top1000_seq_passage= texttokenizer.texts_to_sequences(top1000['p_tokens'])
    top1000_seq_padding_passage = utils.pad_sequences(top1000_seq_passage, maxlen=207, padding="post", truncating="post")

    top1000_seq_query= texttokenizer.texts_to_sequences(top1000['q_tokens'])
    top1000_seq_padding_query = utils.pad_sequences(top1000_seq_query, maxlen=17, padding="post", truncating="post")

    # save sequences 
    print('save sequences........................')
    np.save('t4_train_passage.npy',train_seq_padding_passage)
    np.save('t4_train_query.npy',train_seq_padding_query)
    np.save('t4_vali_passage.npy',vali_seq_padding_passage)
    np.save('t4_vali_query.npy',vali_seq_padding_query)
    np.save('t4_train_label.npy', train_label)
    np.save('t4_vali_label.npy', vali_label)
    np.save('top1000_seq_padding_passage.npy',top1000_seq_padding_passage)
    np.save('top1000_seq_padding_query.npy',top1000_seq_padding_query)   



    # generating embedding matrix
    print('generating embedding matrix........................')
    size_vocab = len(texttokenizer.word_index)+1 # +1 for padding
    embedding_matrix = np.zeros((size_vocab, 100))
    for word, i in texttokenizer.word_index.items():
        #update the row with vector    
        try:
            embedding_matrix[i] =  W2V.wv[word]
        # if word not in model then skip and the row stays all 0s    
        except:
            pass
    np.save('embedding_matrix.npy',embedding_matrix)


    print("Time consumed {0} s".format(timeit.default_timer()-start))
    