import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from task1_evaluation import mean_AP, mean_NDCG



class LogisticRegression():
    def __init__(self, class_weight):
        self.class_weight_0 = class_weight[0]
        self.class_weight_1 = class_weight[1]

    def sigmoid(self,z):
        sig = 1/(1+np.exp(-z))
        return sig

    def initialize(self,X,y):
        self.weights = np.zeros((X.shape[1]+1,1))
        self.X = np.c_[np.ones((X.shape[0],1)),X]
        self.y = y
        self.m = len(y)
        return self.weights,self.X

    def cost(self):
        h = self.sigmoid(self.X@self.weights)
        cost0 = self.class_weight_0*(self.y.T.dot(np.log(h)))
        cost1 = self.class_weight_1*((1 - self.y).T.dot(np.log(1 - h)))
        cost = -((cost1 + cost0)) / self.m
        return cost

    # unweighted gradient of loss function
    # def grad(self):
    #     h = self.sigmoid(self.X@self.weights)
    #     grad = np.dot(self.X.T,h-self.y)/self.m
    #     return grad

    def grad(self):
        h = self.sigmoid(self.X@self.weights)
        tem = -self.class_weight_0*self.y - (self.class_weight_1-self.class_weight_0)*self.y*h+self.class_weight_1*h
        grad = np.dot(self.X.T,tem)/self.m
        return grad
    
    def fit(self,X,y,lr,iter):
        self.initialize(X,y)
        cost_list = np.zeros(iter,)
        for i in range(iter):
            self.weights -= lr*self.grad()
            cost_list[i] = self.cost()
        return cost_list

    def predict(self,X):
        X = np.c_[np.ones((X.shape[0],1)),X]
        z = np.dot(X,self.weights)
        y_pred = self.sigmoid(z)
        return y_pred

def loss_lr_relation(train_feature, train_label):
    lr_list = [8,5,0.1,0.01,0.001]
    plt.figure(figsize=(8,6))
    plt.xlabel('Epochs', fontsize=17)
    plt.ylabel('Training Loss', fontsize=17)
    plt.title('Training loss with different learning rates', fontsize=17)
    class_weight_1 = len(train_label)/(2*sum(train_label))
    class_weight_0 = len(train_label)/(2*(len(train_label)-sum(train_label)))
    for lr in lr_list:
        model = LogisticRegression([class_weight_0, class_weight_1])
        cost_list = model.fit(train_feature,train_label,lr=lr,iter=500)
        plt.plot(np.arange(len(cost_list)),cost_list,label='lr ='+ str(lr))

    plt.legend()
    plt.savefig('learning_rate.pdf')
    plt.show()



if __name__ == "__main__":

    # read train feature and label
    print('read train feature and label........................')
    train_feature = np.load('train_feature.npy',allow_pickle=True)
    train_label = np.load('train_label.npy',allow_pickle=True)

    # study the relationship between loss and learning rate
    print('plot relationship between loss and learning rate........................')
    loss_lr_relation(train_feature, train_label)

    # train model
    print('train model........................')
    class_weight_1 = len(train_label)/(2*sum(train_label))
    class_weight_0 = len(train_label)/(2*(len(train_label)-sum(train_label)))
    model = LogisticRegression([class_weight_0, class_weight_1])
    cost_list = model.fit(train_feature,train_label,lr=0.001, iter=400)

    # load validation data
    print('load validation data........................')
    vali_feature = np.load('vali_feature.npy',allow_pickle=True)
    vali_label = np.load('vali_label.npy',allow_pickle=True)

    # predict validation data
    print('predicting validation data........................')
    vali_pred = model.predict(vali_feature)

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
    top_pred = model.predict(top1000_feature)

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
    toplist['algorithm'] = 'LR'
    toplist = toplist.loc[:,['qid','assignment','pid','rank','score','algorithm']]
    toplist.to_csv('LR.txt',index=False,header=False, sep=' ')





