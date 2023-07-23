import task1
import task2
import pandas as pd
import numpy as np

def passages_preprocess():

    # read csv
    candidate_passages = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', header=None, names=['qid','pid','query','passage'])

    # remove duplicated passages
    candidate_passages_distinct = candidate_passages.drop_duplicates(subset='pid',keep='first',inplace=False).reset_index()

    # document preprocess
    candidate_passages_tokens = task1.preprocess(candidate_passages_distinct.passage,stopword_removal=True)

    # return tokenized passages, corresponding pid and original dataframe 
    return candidate_passages_tokens, candidate_passages_distinct.pid, candidate_passages

# generate a series of tfidf vectors according to every unique passage
def tfidf_passages(pid_passage_dict, inverted_index): 

    number_passage = len(pid_passage_dict)
    
    IDF = {}
    TF_IDF_vectors = {}
    for word, pid_counts in inverted_index.items():
        n = len(pid_counts)
        idf = np.log10(number_passage/n)
        IDF[word] = idf
        for pid, number in pid_counts.items():
            tf = number/len(pid_passage_dict[pid])
            if pid not in TF_IDF_vectors.keys():
                TF_IDF_vectors[pid] = {word:tf*idf}
            else:
                TF_IDF_vectors[pid].update({word:tf*idf})

    return IDF, TF_IDF_vectors


def query_preprocess():
    test_queries = pd.read_csv('test-queries.tsv', sep='\t', header=None, names=['qid','query'])
    queries_tokens = task1.preprocess(test_queries['query'], stopword_removal=True)
    return queries_tokens, test_queries.qid


# generate a series of tfidf vectors according to every unique query
def tfidf_query(qid_query_dict, inverted_queries, IDF):

    TF_IDF_query_vectors = {}
    for word, qid_counts in inverted_queries.items():
        for qid, number in qid_counts.items():
            tf = number/len(qid_query_dict[qid])
            if word in IDF.keys():
                if qid not in TF_IDF_query_vectors.keys():
                    TF_IDF_query_vectors[qid]={word:tf*IDF[word]}
                else:
                    TF_IDF_query_vectors[qid].update({word:tf*IDF[word]})

    return TF_IDF_query_vectors

# similarity between one query and one passage
def TF_IDF_score(query_vector, passage_vector):

    common_vector_word = query_vector.keys() & passage_vector.keys()
    inner_product = 0
    for word in common_vector_word:
        inner_product += query_vector[word]*passage_vector[word]
    
    norm_query = np.linalg.norm(list(query_vector.values())) 
    norm_passage = np.linalg.norm(list(passage_vector.values()))
    if inner_product == 0:
        score = 0
    else:
        score = inner_product / (norm_query * norm_passage)  
    return score   

def TF_IDF_score_top100(series_query_vectors, series_passage_vectors, candidate_passages):

    for i in range(len(candidate_passages)):
        qid = candidate_passages.loc[i,'qid']
        pid = candidate_passages.loc[i,'pid']
        candidate_passages.loc[i,'score'] = TF_IDF_score(series_query_vectors[qid],series_passage_vectors[pid])

    top_100_df = pd.DataFrame()
    for qid in series_query_vectors.keys():
        query_passage_score = candidate_passages.loc[candidate_passages['qid']==qid,['qid','pid','score']].sort_values(by=['score'],ascending=False, inplace=False)
        if len(query_passage_score) >= 100:
            query_passage_score = query_passage_score.iloc[:100,:]

        top_100_df = top_100_df.append(query_passage_score, ignore_index = True)

    top_100_df.to_csv("tfidf.csv", index = False, header = False)

# similarity between one query and one passage
def BM25_score(qid, pid, qid_query_dict, pid_passage_dict, inverted_passages, inverted_query, N, avdl):

    k1 = 1.2
    k2 = 100
    b = 0.75

    score = 0
    for word in qid_query_dict[qid]:
        if word in inverted_passages.keys():
            n = len(inverted_passages[word])
        else:
            n = 0
        dl = len(pid_passage_dict[pid])
        K = k1*((1-b)+b*dl/avdl)
        
        if word in pid_passage_dict[pid]:
            f = inverted_passages[word][pid]
        else:
            f = 0

        qf = inverted_query[word][qid]

        score += np.log( ((0+0.5)/(0-0+0.5)) /((n-0+0.5)/(N-n-0))) * (k1+1)*f*(k2+1)*qf / ((K+f)*(k2+qf))
    return score


def BM25_score_top100(pid_passage_dict, qid_query_dict, candidate_passages, inverted_passages, inverted_query):
    N = len(pid_passage_dict.keys())
    length = 0
    for pid in pid_passage_dict.keys():
        length += len(pid_passage_dict[pid])
    avdl = length/N
    
    for i in range(len(candidate_passages)):
        qid = candidate_passages.loc[i,'qid']
        pid = candidate_passages.loc[i,'pid']
        candidate_passages.loc[i,'score'] = BM25_score(qid, pid, qid_query_dict, pid_passage_dict, inverted_passages, inverted_query, N, avdl)

    top_100_df = pd.DataFrame()
    for qid in qid_query_dict.keys():
        query_passage_score = candidate_passages.loc[candidate_passages['qid']==qid,['qid','pid','score']].sort_values(by=['score'],ascending=False, inplace=False)
        if len(query_passage_score) >= 100:
            query_passage_score = query_passage_score.iloc[:100,:]

        top_100_df = top_100_df.append(query_passage_score, ignore_index = True)

    top_100_df.to_csv("bm25.csv", index = False, header = False)




if __name__ == "__main__":

    tokenized_passages, pids, candidate_passages = passages_preprocess()
    tokenized_queries, qids = query_preprocess()
    pid_passage_dict = dict(zip(pids,tokenized_passages))
    qid_query_dict = dict(zip(qids,tokenized_queries))
    inverted_passages = task2.inverted_index(pids, tokenized_passages)
    inverted_queries = task2.inverted_index(qids, tokenized_queries)

    # TF_IDF
    IDF, TF_IDF_vectors = tfidf_passages(pid_passage_dict, inverted_passages)
    TF_IDF_query_vectors = tfidf_query(qid_query_dict, inverted_queries,IDF)

    TF_IDF_score_top100(TF_IDF_query_vectors, TF_IDF_vectors, candidate_passages)
    
    # BM25
    BM25_score_top100(pid_passage_dict, qid_query_dict, candidate_passages, inverted_passages, inverted_queries)




