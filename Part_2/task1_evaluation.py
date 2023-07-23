import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


def mean_AP(result_df, ranknum):
    AP = []
    unique_qid = result_df.drop_duplicates(subset=['qid'], keep='first', inplace=False)
    for qid in unique_qid['qid']:
        collection = result_df[result_df['qid'] == qid]
        if len(collection) <= ranknum:
            retrived_collection = collection
        else:
            retrived_collection = collection[:ranknum]

        # if for a specific query, no passage relevant is retrieved, set average precision =0
        if np.sum(retrived_collection['relevancy']) == 0:
            AP.append(0)
        else:
            current_rank = 0
            relevant_retrived = 0
            precision = 0

            for row, data in retrived_collection.iterrows():
                current_rank += 1
                # if relevancy >0
                if data['relevancy'] > 0:
                    relevant_retrived += 1
                    precision += relevant_retrived/current_rank

            AP.append(precision/np.sum(retrived_collection['relevancy'] > 0))

    return np.mean(AP)


def mean_NDCG(result_df, ranknum):
    NDCG = []
    unique_qid = result_df.drop_duplicates(subset=['qid'], keep='first', inplace=False)
    for qid in unique_qid['qid']:
        collection = result_df[result_df['qid'] == qid]
        if len(collection) <= ranknum:
            retrived_collection = collection
        else:
            retrived_collection = collection[:ranknum]

        # if for a specific query, no passage relevant is retrieved, set NDCG =0
        if np.sum(retrived_collection['relevancy']) == 0:
            NDCG.append(0)
        else:
            current_rank = 0
            DCG = 0

            for row, data in retrived_collection.iterrows():
                current_rank += 1
                DCG += (2**data['relevancy']-1) / np.log2(current_rank + 1)

            opt = retrived_collection.sort_values(
                by=['relevancy'], ascending=False, inplace=False)
            opt_current_rank = 0
            opt_DCG = 0
            for row, data in opt.iterrows():
                opt_current_rank += 1
                opt_DCG += (2**data['relevancy']-1) / np.log2(opt_current_rank + 1)

            NDCG.append(DCG/opt_DCG)

    return np.mean(NDCG)


if __name__ == "__main__":

    bm_result = pd.read_csv('bm25_toplist.csv', sep=',', header=0)

    meanAP_3 = mean_AP(bm_result, 3)
    meanAP_10 = mean_AP(bm_result,  10)
    meanAP_100 = mean_AP(bm_result,  100)

    print('The mean average precision @ 3 is', meanAP_3)
    print('The mean average precision @ 10 is', meanAP_10)
    print('The mean average precision @ 100 is', meanAP_100)

    meanNDCG_3 = mean_NDCG(bm_result,  3)
    meanNDCG_10 = mean_NDCG(bm_result,  10)
    meanNDCG_100 = mean_NDCG(bm_result,  100)

    print('The mean NDCG @ 3 is', meanNDCG_3)
    print('The mean NDCG @ 10 is', meanNDCG_10)
    print('The mean NDCG @ 100 is', meanNDCG_100)
