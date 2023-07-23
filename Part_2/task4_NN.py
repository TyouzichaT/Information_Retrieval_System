from keras.layers import Input, Embedding, Dense, LSTM, Dot, Dropout
from keras import Model
import tensorflow as tf
import numpy as np
import timeit
from matplotlib import pyplot as plt
import pandas as pd
from task1_evaluation import mean_AP, mean_NDCG
from keras.utils import plot_model
from keras.initializers import Constant

if __name__ == "__main__":
    start = timeit.default_timer()

    print('Define DSSM ........................')
    # load embedding_matrix and generate embedding layer
    embedding_matrix = np.load('embedding_matrix.npy', allow_pickle=True)
    vocab_size = embedding_matrix.shape[0]
    embed_dim = embedding_matrix.shape[1]

    # Define  model input layer
    query_len = 17
    doc_len = 207
    query_input = Input(shape=(query_len,))
    doc_input = Input(shape=(doc_len,))

    # Input layers
    query_input = Input(shape=(query_len,), name='query_input')
    doc_input = Input(shape=(doc_len,), name='doc_input')

    # Embedding layers
    query_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=query_len, trainable=False)(query_input)
    doc_embedding = Embedding(input_dim=vocab_size, output_dim=embed_dim,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=query_len, trainable=False)(doc_input)

    query_lstm = LSTM(128)(query_embedding)
    doc_lstm = LSTM(128)(doc_embedding)

    query_drop = Dropout(0.2)(query_lstm)
    doc_drop = Dropout(0.2)(doc_lstm)

    query_vec = Dense(64)(query_drop)
    doc_vec = Dense(64)(doc_drop)

    cos = Dot(axes=-1, normalize=True, name='cosine')([query_vec, doc_vec])
    out = Dense(1, activation='sigmoid', name='out')(cos)

    # Build the model
    model = Model(inputs=[query_input, doc_input], outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss='binary_crossentropy')
    plot_model(model, to_file='model.png')
    print('The architecture of DSSM is')
    print(model.summary())

    # Train model
    print('Training this model..........................')

    train_seq_padding_passage = np.load('t4_train_passage.npy', allow_pickle=True)
    train_seq_padding_query = np.load('t4_train_query.npy', allow_pickle=True)
    vali_seq_padding_passage = np.load('t4_vali_passage.npy', allow_pickle=True)
    vali_seq_padding_query = np.load('t4_vali_query.npy', allow_pickle=True)

    train_label = np.load('t4_train_label.npy', allow_pickle=True)[:, np.newaxis]
    vali_label = np.load('t4_vali_label.npy', allow_pickle=True)[:, np.newaxis]

    history = model.fit(x=[train_seq_padding_query, train_seq_padding_passage], y=train_label, batch_size=512, epochs=20, verbose=1)

    # predict on testing dataset
    print('predict on testing dataset........................')
    vali_pred = model.predict([vali_seq_padding_query, vali_seq_padding_passage])

    # sorting output
    validation_data = pd.read_csv('validation_data.tsv', sep='\t', header=0)
    validation_data['score'] = vali_pred
    validation_data.sort_values(by=['qid', 'score'], ascending=False, inplace=True)
    validation_data.reset_index(drop=True, inplace=True)

    # calculate mAP and mNDVG
    print('calculating mAP and mNDVG........................')
    meanAP_3 = mean_AP(validation_data, 3)
    meanAP_10 = mean_AP(validation_data,  10)
    meanAP_100 = mean_AP(validation_data,  100)

    print('The mean average precision @ 3 is', meanAP_3)
    print('The mean average precision @ 10 is', meanAP_10)
    print('The mean average precision @ 100 is', meanAP_100)

    meanNDCG_3 = mean_NDCG(validation_data,  3)
    meanNDCG_10 = mean_NDCG(validation_data,  10)
    meanNDCG_100 = mean_NDCG(validation_data,  100)

    print('The mean NDCG @ 3 is', meanNDCG_3)
    print('The mean NDCG @ 10 is', meanNDCG_10)
    print('The mean NDCG @ 100 is', meanNDCG_100)
    

    # predict top1000
    print('predicting top1000........................')
    top1000_seq_padding_passage = np.load('top1000_seq_padding_passage.npy', allow_pickle=True)
    top1000_seq_padding_query = np.load('top1000_seq_padding_query.npy', allow_pickle=True)
    top_pred = model.predict([top1000_seq_padding_query, top1000_seq_padding_passage])

    # sorting top1000
    print('sorting top1000........................')
    top1000 = pd.read_csv('candidate_passages_top1000.tsv', sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'] )
    top1000['score'] = top_pred
    qid_list = top1000.drop_duplicates(subset='qid',keep='first',inplace=False).reset_index(drop=True)['qid']

    toplist = pd.DataFrame()
    for qid in qid_list:
        qid_passages= top1000[top1000['qid']==qid].sort_values(by=['score'],ascending=False, inplace=False).reset_index(drop=True)
        qid_passages['rank']=np.arange(1,len(top1000[top1000['qid']==qid])+1)
        toplist=toplist.append(qid_passages, ignore_index = True)


    toplist['assignment'] = 'A2'
    toplist['algorithm'] = 'NN'
    toplist = toplist.loc[:,['qid','assignment','pid','rank','score','algorithm']]
    toplist.to_csv('NN.txt',index=False,header=False, sep=' ')



    print('All processes take %s minutes' %((timeit.default_timer() - start)/60))
