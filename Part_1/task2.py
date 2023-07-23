import task1
import pandas as pd
from collections import Counter


def documents_preprocess():

    # read csv
    candidate_passages = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', header=None, names=['qid','pid','query','passage'])

    # remove duplicated passages
    candidate_passages_distinct = candidate_passages.drop_duplicates(subset='pid',keep='first',inplace=False).reset_index()

    # document preprocess
    candidate_passages_tokens = task1.preprocess(candidate_passages_distinct.passage,stopword_removal=True)

    # return tokenized passages, corresponding pid 
    return candidate_passages_tokens, candidate_passages_distinct.pid


def inverted_index(pids, tokenized_passages):
    inverted = {}

    for i in range(len(tokenized_passages)):
        word_count = Counter(tokenized_passages[i])
        for word, number in word_count.items():
            if not word in inverted.keys():
                inverted[word] = {pids[i]:number}
            else:
                inverted[word].update({pids[i]:number})

    return inverted


if __name__ == "__main__":

    tokenized_passages, pids = documents_preprocess()

    inverted = inverted_index(pids,tokenized_passages)





