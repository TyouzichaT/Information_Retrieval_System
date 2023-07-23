import task2
import task3
import pandas as pd
import numpy as np

def Query_likelihood(qid, pid, qid_query_dict, pid_passage_dict, inverted_passages, smooth_mode, V, tol_coll_word):

    score = 0
    D = len(pid_passage_dict[pid])

    for word in qid_query_dict[qid]:

        if word in pid_passage_dict[pid]:
            m = inverted_passages[word][pid]
        else:
            m =0

        if word in inverted_passages.keys(): 
            cq = float(sum(inverted_passages[word].values()))
        else:
            cq = 0
    
        if smooth_mode == 'laplace':
            score += np.log((m+1)/(D+V)) 
        elif smooth_mode == 'lidstone':
            epsilon = 0.1
            score += np.log((m+epsilon)/(D+V*epsilon)) 
        elif smooth_mode == 'dirichlet':
            miu = 50
            score += np.log((D/(D+miu))*(m/D)+(miu/(D+miu))*(cq/tol_coll_word))

    return score

def Query_likelihood_top100(pid_passage_dict, qid_query_dict, inverted_passages, candidate_passages, smooth_mode):
    tol_coll_word = 0
    for pid_count in inverted_passages.values():
        word_occur = float(sum(pid_count.values()))
        tol_coll_word += word_occur 

    V = len(inverted_passages)
    for i in range(len(candidate_passages)):
        qid = candidate_passages.loc[i,'qid']
        pid = candidate_passages.loc[i,'pid']
        candidate_passages.loc[i,'score'] = Query_likelihood(qid, pid, qid_query_dict, pid_passage_dict, inverted_passages, smooth_mode, V, tol_coll_word)

    top_100_df = pd.DataFrame()
    for qid in qid_query_dict.keys():
        query_passage_score = candidate_passages.loc[candidate_passages['qid']==qid,['qid','pid','score']].sort_values(by=['score'],ascending=False, inplace=False)
        if len(query_passage_score) >= 100:
            query_passage_score = query_passage_score.iloc[:100,:]

        top_100_df = top_100_df.append(query_passage_score, ignore_index = True)

    if smooth_mode=='laplace':
        top_100_df.to_csv("laplace.csv", index = False, header = False)
    elif smooth_mode=='lidstone':
        top_100_df.to_csv("lidstone.csv", index = False, header = False)
    elif smooth_mode=='dirichlet':
        top_100_df.to_csv("dirichlet.csv", index = False, header = False)


if __name__ == "__main__":

    tokenized_passages, pids, candidate_passages = task3.passages_preprocess()
    tokenized_queries, qids = task3.query_preprocess()
    pid_passage_dict = dict(zip(pids,tokenized_passages))
    qid_query_dict = dict(zip(qids,tokenized_queries))
    inverted_passages = task2.inverted_index(pids, tokenized_passages)
    inverted_queries = task2.inverted_index(qids, tokenized_queries)

    # Laplace smoothing
    Query_likelihood_top100(pid_passage_dict, qid_query_dict, inverted_passages, candidate_passages, smooth_mode='laplace')

    # Lidstone smoothing
    Query_likelihood_top100(pid_passage_dict, qid_query_dict, inverted_passages, candidate_passages, smooth_mode='lidstone')

    # Dirichlet smoothing
    Query_likelihood_top100(pid_passage_dict, qid_query_dict, inverted_passages, candidate_passages, smooth_mode='dirichlet')

