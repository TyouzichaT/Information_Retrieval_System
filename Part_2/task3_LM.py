import xgboost as xgb
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from task1_evaluation import mean_AP, mean_NDCG
from bayes_opt import BayesianOptimization
import time

def cross_validation(max_depth, min_child_weight, eta, subsample, colsample_bytree, gamma, num_boost_round):
    params = {
        # booster Parameters
        'booster': 'gbtree',
        "objective": "rank:pairwise",
        "tree_method": "hist",


        'max_depth': round(max_depth),
        'min_child_weight': min_child_weight,
        'eta': eta,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'gamma': gamma, }

    results = xgb.cv(params, training_data, num_boost_round=round(
        num_boost_round), nfold=6, metrics=['map'])

    return float(results['test-map-mean'].values[-1:])


if __name__ == "__main__":
    start = time.time()
    # load train feature and label
    print('load train feature and label........................')
    train_feature = np.load('train_feature.npy', allow_pickle=True)
    train_label = np.load('train_label.npy', allow_pickle=True)
    train_qids = np.load('train_qids.npy', allow_pickle=True)

    training_data = xgb.DMatrix(train_feature, label=train_label, qid=train_qids)


    # Run Bayesian Optimization
    print('Run Bayesian Optimization........................')
    start1 = time.time()
    params_gbm ={
        'max_depth':(3, 10),
        'min_child_weight':(0, 10),
        'eta':(0.01, 1),
        'subsample': (0.8, 1),
        'colsample_bytree': (0.8, 1), 
        'gamma': (1,9),
        'num_boost_round':(80, 200),
    }
    gbm_bo = BayesianOptimization(cross_validation, params_gbm, random_state=111)
    gbm_bo.maximize(init_points=20, n_iter=20)

    print('It takes %s minutes' % ((time.time() - start1)/60))

    # Train best model
    print('Train best model........................')
    params_gbm = gbm_bo.max['params']
    params_gbm['max_depth'] = round(params_gbm['max_depth'])
    params_gbm['num_boost_round'] = round(params_gbm['num_boost_round'])
    general_params = {
    'booster':'gbtree',
    "objective": "rank:pairwise", 
    "tree_method": "hist"}

    best_params = {**general_params,**params_gbm}
    del best_params['num_boost_round']
    best_model = xgb.train(best_params,training_data,num_boost_round=params_gbm['num_boost_round'])

    # load validation feature and label
    print('load validation feature and label........................')
    vali_feature = np.load('vali_feature.npy', allow_pickle=True)
    vali_label = np.load('vali_label.npy', allow_pickle=True)

    testing_data = xgb.DMatrix(vali_feature, label=vali_label)

    # predict on testing dataset
    print('predict on testing dataset........................')
    vali_pred = best_model.predict(testing_data)

    # sorting output
    validation_data = pd.read_csv('validation_data.tsv', sep='\t',header=0)
    validation_data['score'] = vali_pred
    validation_data.sort_values(by=['qid','score'],ascending=False, inplace=True)
    validation_data.reset_index(drop=True, inplace= True)

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
    top1000_feature = np.load('top1000_feature.npy',allow_pickle=True)
    top1000_Dmatix = xgb.DMatrix(top1000_feature)
    top_pred = best_model.predict(top1000_Dmatix)

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
    toplist['algorithm'] = 'LM'
    toplist = toplist.loc[:,['qid','assignment','pid','rank','score','algorithm']]
    toplist.to_csv('LM.txt',index=False,header=False, sep=' ')


    print('All processes take %s minutes' % ((time.time() - start)/60))




