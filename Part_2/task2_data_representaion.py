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

    unique_qids = nonrelevant.drop_duplicates(
        subset='qid', keep='first', inplace=False).reset_index(drop=True)['qid']

    train_sample = pd.DataFrame()
    for qid in unique_qids:
        nonrel_with_qid = nonrelevant[nonrelevant['qid'] == qid]
        if len(nonrel_with_qid) > 50:
            nonrel_with_qid = nonrel_with_qid.sample(50)

        train_sample = pd.concat(
            [train_sample, nonrel_with_qid]).reset_index(drop=True)

    train_sample = pd.concat([train_sample, relevant]).reset_index(drop=True).sort_values(by=['qid'],ascending=True)

    return train_sample

def average_embedding(tokens):
    text_vector = np.mean(W2V.wv[tokens],axis=0)
    return text_vector

def average_embedding_top(tokens):
    words =[word for word in tokens if word in W2V.wv.index_to_key]
    try:
        text_vector = np.mean(W2V.wv[tokens],axis=0)
    except:
        text_vector = np.zeros(100)
    return text_vector

def feature_label_genearation(data):
    features = np.zeros(shape=[len(data), 200]) 
    for row, content in data.iterrows():
        p_vec = content['p_vec']
        q_vec = content['q_vec']
        features[row,:] = np.concatenate((q_vec,p_vec))
    
    labels = np.array(data['relevancy'].tolist())[:,np.newaxis]
    return features, labels

def feature_genearation(data):
    features = np.zeros(shape=[len(data), 200]) 
    for row, content in data.iterrows():
        p_vec = content['p_vec']
        q_vec = content['q_vec']
        features[row,:] = np.concatenate((q_vec,p_vec))
    return features

if __name__ == "__main__":
    start = timeit.default_timer()

    # sample train data
    print('sampling train data........................')
    train_data = pd.read_csv('train_data.tsv', sep='\t', header=0)
    train_sample = sample(train_data)

    # preprocess sample data
    print('preprocessing sample data........................')
    train_sample['q_tokens']=np.vectorize(preprocess, otypes=[list])(train_sample['queries'])
    train_sample['p_tokens']=np.vectorize(preprocess, otypes=[list])(train_sample['passage'])

    # generating p_vec and q_vec for sample data
    print('generating p_vec and q_vec for sample data........................')
    train_sample['q_vec'] = np.vectorize(average_embedding, otypes=[np.ndarray])(train_sample['q_tokens'])
    train_sample['p_vec'] = np.vectorize(average_embedding, otypes=[np.ndarray])(train_sample['p_tokens'])

    # generate train_sample fearture and label
    print('generating train_sample fearture and label........................')
    train_feature, train_label = feature_label_genearation(train_sample)
    train_qids = np.array(train_sample['qid'].tolist())[:,np.newaxis]

    # read preprocessed validation set
    print('reading preprocessed validation set........................')
    with open("vali_tokens.pkl", 'rb') as file:
        validation_data = pickle.load(file)

    # generating p_vec and q_vec for validation data
    print('generating p_vec and q_vec for validation data........................')
    validation_data['q_vec'] = np.vectorize(average_embedding, otypes=[np.ndarray])(validation_data['q_tokens'])
    validation_data['p_vec'] = np.vectorize(average_embedding, otypes=[np.ndarray])(validation_data['p_tokens'])

    # generate validation fearture and label
    print('generating validation fearture and label........................')
    vali_feature, vali_label= feature_label_genearation(validation_data)
    vali_qids = np.array(validation_data['qid'].tolist())[:,np.newaxis]


    #read top1000 tsv
    print('read top1000.tsv........................')
    top1000 = pd.read_csv('candidate_passages_top1000.tsv', sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'] )
    # preprocess top1000
    print('preprocessing top1000........................')
    top1000['q_tokens']=np.vectorize(preprocess, otypes=[list])(top1000['query'])
    top1000['p_tokens']=np.vectorize(preprocess, otypes=[list])(top1000['passage'])

    # generating p_vec and q_vec for top1000
    print('generating p_vec and q_vec for top1000........................')
    top1000['q_vec'] = np.vectorize(average_embedding_top, otypes=[np.ndarray])(top1000['q_tokens'])
    top1000['p_vec'] = np.vectorize(average_embedding_top, otypes=[np.ndarray])(top1000['p_tokens'])

    # generate top1000 fearture 
    print('generating top1000 feature ........................')
    top1000_feature = feature_genearation(top1000)

    print("Time consumed {0} s".format(timeit.default_timer()-start))


    np.save('vali_feature.npy',vali_feature)
    np.save('vali_label.npy',vali_label)
    np.save('train_feature.npy',train_feature)
    np.save('train_label.npy',train_label)
    np.save('train_qids.npy', train_qids)
    np.save('vali_qids.npy', vali_qids)
    np.save('top1000_feature.npy', top1000_feature)



