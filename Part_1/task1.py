from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize
from nltk.stem import SnowballStemmer
from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def read_file(filename):
    text = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            text.append(line)
    return text


def preprocess(text, stopword_removal=False):

    processed_text = []
    stemmer = SnowballStemmer('english')
    stop_words = set(stopwords.words('english')) 

    for line in text:
        # Normalization and tokenlization - lowercase and remove punctuation
        pattern = r'\w+'
        tokens = regexp_tokenize(line.lower(), pattern)

        # stopward_removal
        if stopword_removal==True:
            tokens = [w for w in tokens if not w in stop_words]

        # stemming
        tokens = [stemmer.stem(w) for w in tokens]

        processed_text.append(tokens)

    return processed_text


def counter_vocabulary(text):
    flat_tokens = [token for subtext in text for token in subtext]
    counts = Counter(flat_tokens)
    # sort the list by descending order
    sorted_words = counts.most_common(len(counts))

    return np.array(sorted_words)


def zipf_plot(text,title):

    word_counter = counter_vocabulary(text)
    print(f"identified index of terms: {word_counter.shape[0]}")

    frequency = word_counter[:,1].astype(np.int64)/np.sum(word_counter[:,1].astype(np.int64))
    rank = np.arange(1, word_counter.shape[0]+1)

    # generate data for standard Zipf's curve
    Hn = np.sum(1/rank)
    Zipf = 1 / (rank * Hn)

    # plot and save graphs
    plt.figure(figsize=(8, 4))
    plt.title("Zipf's Law Comparation "+title)
    plt.xlabel("Term frequency ranking")
    plt.ylabel("Term prob. of occurrence")
    plt.plot(rank, frequency, color='blue', label="data")
    plt.plot(rank, Zipf, linestyle="--",
             color='r', label="theory (Zipf's curve)")
    plt.legend()
    plt.savefig("Zipf'sLaw_plot_"+title+".pdf")
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.title("Zipf's Law Comparation(loglog) "+title)
    plt.xlabel("Term frequency ranking (log)")
    plt.ylabel("Term prob. of occurrence (log)")
    plt.loglog(rank, frequency, color='blue', label="data")
    plt.loglog(rank, Zipf, linestyle="--",
               color='r', label="theory (Zipf's curve)")
    plt.legend()
    plt.savefig("Zipf'sLaw_loglog_"+title+".pdf")
    plt.show()


if __name__ == "__main__":
    passage_collection = read_file('passage-collection.txt')
    
    #With stopword
    processed_text = preprocess(passage_collection)
    zipf_plot(processed_text, title='with_stopword')

    #Without stopword
    processed_text_no_stopword = preprocess(passage_collection, stopword_removal=True)
    zipf_plot(processed_text_no_stopword, title='without_stopword')