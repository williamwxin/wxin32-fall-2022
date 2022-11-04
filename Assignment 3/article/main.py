# This is a sample Python script.
# https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/#:~:text=A%20fundamental%20step%20for%20any%20unsupervised%20algorithm%20is,clustering%20technique%20using%20the%20Sklearn%20library%20of%20python.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
import numpy as np
import time
import copy
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.environ["OMP_NUM_THREADS"] = '5'

from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.cluster import KMeans
from matplotlib.ticker import MaxNLocator
from sklearn.mixture import GaussianMixture
import sklearn.decomposition as DC
import sklearn.random_projection as RP
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import scipy

data = loadtxt(open('OnlineNewsPopularity.csv'), delimiter=",", dtype=object)

# data = loadtxt(open('housing.csv'),delimiter = ",", dtype=object)
data = data[:, 1:]
cols = data[0, :]
data = data[1:, :].astype(float)
np.random.shuffle(data)
np.random.shuffle(data)
data_p1 = data[data[:,-1]>=5000]
data_p2 = data[data[:,-1]<5000]
data_p1[:,-1] = 1
data_p2[:,-1] = 0

p2_idx = np.random.choice(len(data_p2),2500,replace=False)
data_p2 = data_p2[p2_idx]

data_b = np.vstack((data_p1[:2500,:], data_p2))

data_x = data_b[:, :-1]
data_x = np.array(data_x, dtype=float)
data_y = data_b[:, -1]

data_y_class = data_y.copy()
# data_y_class[np.array(data_y, dtype=float)>=23]=1
data_y_class = np.array(data_y_class, dtype=int)
log_cols = [2,6,7,8,9,19,21,25,26,27,28,29]
for i in log_cols:
    data_x[:,i] = np.log(data_x[:,i]+1)
log_log = [9]
for i in log_log:
    data_x[:,i] = np.log(data_x[:,i]+1)
log_big = [46,50]
for i in log_big:
    data_x[:,i] = np.log(data_x[:,i]*100+1)

train_x, test_x, train_y, test_y = train_test_split(data_x, data_y_class, test_size=0.20, random_state=0)

txt_mean = train_x.mean(axis=0)
txt_std = train_x.std(axis=0)
train_x = (train_x - txt_mean) / txt_std
test_x = (test_x - txt_mean) / txt_std
samples, features = train_x.shape

k_range = list(range(2,41,2))
km_res = np.zeros((len(k_range),6))
em_res = np.zeros((len(k_range),6))

avg = 3
for i, k in enumerate(k_range):
    for t in range(avg):
        model = KMeans(n_clusters=k, random_state=t)
        model.fit(train_x)
        pred_c = model.predict(train_x)
        test_2 = model.cluster_centers_
        km_res[i,0] += metrics.silhouette_score(train_x, pred_c, metric='euclidean')/5
        km_res[i, 1] += metrics.silhouette_score(train_x, pred_c, metric='manhattan')/5
        km_res[i, 2] += metrics.silhouette_score(train_x, pred_c, metric='cosine')/5
        km_res[i,3] += (sum(np.min(cdist(train_x, model.cluster_centers_,'euclidean'), axis=1)) / train_x.shape[0])/5
        km_res[i, 4] += (sum(np.min(cdist(train_x, model.cluster_centers_, 'cityblock'), axis=1)) / train_x.shape[0])/5
        km_res[i, 5] += (sum(np.min(cdist(train_x, model.cluster_centers_, 'cosine'), axis=1)) / train_x.shape[0])/5
        # km_res[i,1] = metrics.calinski_harabasz_score(train_x, pred_c)
        # km_res[i,2] = metrics.davies_bouldin_score(train_x, pred_c)

        model = GaussianMixture(n_components=k, random_state=t)
        model.fit(train_x)
        pred_c = model.predict(train_x)
        test_1 = model.means_
        em_res[i, 0] += metrics.silhouette_score(train_x, pred_c, metric='euclidean')/5
        em_res[i, 1] += metrics.silhouette_score(train_x, pred_c, metric='manhattan')/5
        em_res[i, 2] += metrics.silhouette_score(train_x, pred_c, metric='cosine')/5
        em_res[i, 3] += (sum(np.min(cdist(train_x, model.means_,'euclidean'), axis=1)) / train_x.shape[0])/5
        em_res[i, 4] += (sum(np.min(cdist(train_x, model.means_, 'cityblock'), axis=1)) / train_x.shape[0])/5
        em_res[i, 5] += (sum(np.min(cdist(train_x, model.means_, 'cosine'), axis=1)) / train_x.shape[0])/5
    # em_res[i, 1] = metrics.calinski_harabasz_score(train_x, pred_c)
    # em_res[i, 2] = metrics.davies_bouldin_score(train_x, pred_c)

    print(k)

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Article K-Means: Silhouette Score')
plt.xlabel('K')
plt.ylabel('Metric Value')
sns.lineplot(y=km_res[:, 0], x=k_range, label='Euclidean')
sns.lineplot(y=km_res[:, 1], x=k_range, label='Manhattan')
sns.lineplot(y=km_res[:, 2], x=k_range, label='Cosine')
plt.savefig('Article p1 KM silhouette')

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Article EM: Silhouette Score')
plt.xlabel('# Clusters')
plt.ylabel('Metric Value')
sns.lineplot(y=em_res[:, 0], x=k_range, label='Euclidean')
sns.lineplot(y=em_res[:, 1], x=k_range, label='Manhattan')
sns.lineplot(y=em_res[:, 2], x=k_range, label='Cosine')
plt.savefig('Article p1 EM silhouette')

#distortion
if False:
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Article K-Means: Distortion')
    plt.xlabel('K')
    plt.ylabel('Metric Value')
    sns.lineplot(y=km_res[:, 3], x=k_range, label='Euclidean')
    sns.lineplot(y=km_res[:, 4], x=k_range, label='Manhattan')
    sns.lineplot(y=km_res[:, 5], x=k_range, label='Cosine')
    plt.savefig('Article p1 KM distortion')

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Article K-Means: Distortion')
    plt.xlabel('K')
    plt.ylabel('Metric Value')
    sns.lineplot(y=em_res[:, 3], x=k_range, label='Euclidean')
    sns.lineplot(y=em_res[:, 4], x=k_range, label='Manhattan')
    sns.lineplot(y=em_res[:, 5], x=k_range, label='Cosine')
    plt.savefig('Article p1 EM distortion')


k_range = list(range(2,61,2))
km_res_vote = np.zeros((len(k_range),2))
em_res_vote = np.zeros((len(k_range),2))

for i, k in enumerate(k_range):
    for t in range(avg):
        model = KMeans(n_clusters=k, random_state=t)
        model.fit(train_x)
        pred_c = model.predict(train_x)
        pred_y = np.zeros(len(pred_c))
        votes = np.zeros(k)
        for c in range(k):
            temp = train_y[pred_c==c]
            val = (np.mean(temp)>=0.5) * 1
            pred_y[pred_c==c] = val
            votes[c] = val
        km_res_vote[i,0] += accuracy_score(train_y, pred_y)/avg

        pred_c = model.predict(test_x)
        pred_y = np.zeros(len(pred_c))
        for c in range(k):
            pred_y[pred_c == c] = votes[c]
        km_res_vote[i, 1] += accuracy_score(test_y, pred_y)/avg

        model = GaussianMixture(n_components=k, random_state=t)
        model.fit(train_x)
        pred_c = model.predict(train_x)
        pred_y = np.zeros(len(pred_c))
        votes = np.zeros(k)
        for c in range(k):
            temp = train_y[pred_c == c]
            val = (np.mean(temp) >= 0.5) * 1
            pred_y[pred_c == c] = val
            votes[c] = val
        em_res_vote[i, 0] += accuracy_score(train_y, pred_y)/avg

        pred_c = model.predict(test_x)
        pred_y = np.zeros(len(pred_c))
        for c in range(k):
            pred_y[pred_c == c] = votes[c]
        em_res_vote[i, 1] += accuracy_score(test_y, pred_y)/avg

    print(k)

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Article K-Means: Majority Vote Acc')
plt.xlabel('K')
plt.ylabel('Accuracy')
sns.lineplot(y=km_res_vote[:, 0], x=k_range, label='Train')
sns.lineplot(y=km_res_vote[:, 1], x=k_range, label='Test')
plt.savefig('Article p1 KM vote')

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Article EM: Majority Vote Acc')
plt.xlabel('# Clusters')
plt.ylabel('Accuracy')
sns.lineplot(y=em_res_vote[:, 0], x=k_range, label='Train')
sns.lineplot(y=em_res_vote[:, 1], x=k_range, label='Test')
plt.savefig('Article p1 EM vote')


sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Article: Majority Vote Acc')
plt.xlabel('K')
plt.ylabel('Accuracy')
sns.lineplot(y=km_res_vote[:, 0], x=k_range, label='KM Train')
sns.lineplot(y=km_res_vote[:, 1], x=k_range, label='KM Test')
sns.lineplot(y=em_res_vote[:, 0], x=k_range, label='EM Train')
sns.lineplot(y=em_res_vote[:, 1], x=k_range, label='EM Test')
ax.legend(loc='lower center', fontsize=8)
plt.savefig('Article p1 vote')

if False:
    km_res_2 = np.zeros((len(k_range),2))
    em_res_2 = np.zeros((len(k_range),2))
    for i, k in enumerate(k_range):
        model = KMeans(n_clusters=k)
        model.fit(train_x)
        pred_c = model.predict(train_x)
        km_res_2[i,0] = metrics.adjusted_rand_score(pred_c, train_y)
        km_res_2[i,1] = metrics.adjusted_mutual_info_score(pred_c, train_y)

        model = GaussianMixture(n_components=k)
        model.fit(train_x)
        pred_c = model.predict(train_x)
        em_res_2[i, 0] = metrics.adjusted_rand_score(pred_c, train_y)
        em_res_2[i, 1] = metrics.adjusted_mutual_info_score(pred_c, train_y)

        print(k)

    #p1b
    if True:    #p1b graphs
        sns.set()
        fig, ax = plt.subplots(figsize=(4, 2))
        plt.subplots_adjust(bottom=.26)
        plt.subplots_adjust(left=.16)
        plt.title('Article: Adjusted Rand')
        plt.xlabel('K')
        plt.ylabel('Metric Value')
        sns.lineplot(y=km_res_2[:, 0], x=k_range, label='K-Means')
        sns.lineplot(y=em_res_2[:, 0], x=k_range, label='EM')
        plt.savefig('Article p1 Rand')

        sns.set()
        fig, ax = plt.subplots(figsize=(4, 2))
        plt.subplots_adjust(bottom=.26)
        plt.subplots_adjust(left=.16)
        plt.title('Article: Adjusted Mutual Info')
        plt.xlabel('K')
        plt.ylabel('Metric Value')
        sns.lineplot(y=km_res_2[:, 1], x=k_range, label='K-Means')
        sns.lineplot(y=em_res_2[:, 1], x=k_range, label='EM')
        plt.savefig('Article p1 MI')

        # sns.set()
        # fig, ax = plt.subplots(figsize=(4, 2))
        # plt.subplots_adjust(bottom=.26)
        # plt.subplots_adjust(left=.16)
        # plt.title('Article: Calinski-Harabaz')
        # plt.xlabel('K')
        # plt.ylabel('Metric Value')
        # sns.lineplot(y=km_res[:, 1], x=k_range, label='K-Means')
        # sns.lineplot(y=em_res[:, 1], x=k_range, label='EM')
        # plt.savefig('Article p1 CH')
        #
        # sns.set()
        # fig, ax = plt.subplots(figsize=(4, 2))
        # plt.subplots_adjust(bottom=.26)
        # plt.subplots_adjust(left=.16)
        # plt.title('Article: Davies-Bouldin')
        # plt.xlabel('K')
        # plt.ylabel('Metric Value')
        # sns.lineplot(y=km_res[:, 2], x=k_range, label='K-Means')
        # sns.lineplot(y=em_res[:, 2], x=k_range, label='EM')
        # plt.savefig('Article p1 DB')


dimred_time = np.zeros((len(list(range(1,51,2))),4))
scale = StandardScaler()
scale.fit(train_x)

pca_range = list(range(1,51,2))
pca_res = np.zeros((len(pca_range),2))

for i, c in enumerate(pca_range):
    start_time = time.time()
    model = DC.PCA(n_components=c)
    model.fit(train_x)
    pca_x = model.transform(train_x)
    var_ex = sum(model.explained_variance_ratio_)
    recon = scale.inverse_transform(model.inverse_transform(pca_x))
    recon_err = np.sum((train_x - recon) ** 2, axis=1).mean()
    pca_res[i,0] = var_ex
    pca_res[i,1] = recon_err
    dimred_time[i,0] = time.time() - start_time

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
ax.grid(False)
plt.subplots_adjust(bottom=.3)
plt.subplots_adjust(left=.13)
plt.subplots_adjust(right=.84)
#ax.lege
ax.set_ylabel('Reconstruction Error')
ax.set_xlabel('# Principal Components')
plt.title('Article: PCA')
sns.lineplot(y=pca_res[:, 1], x=pca_range, label='Expl Var', ax=ax, color='orange')
sns.lineplot(y=pca_res[:, 1], x=pca_range, label='Rec Err', ax=ax)
ax2 = ax.twinx()
ax2.grid(False)
ax2.set_ylabel('Explained Variance')
ax2.set_ylim([0,1.1])
sns.lineplot(y=pca_res[:, 0], x=pca_range, ax=ax2, color='orange')
ax.legend(loc='center right')
plt.savefig('Article pca')


ica_range =list(range(1,51,2))
ica_res = np.zeros((len(ica_range),2))

for p in range(5):
    for i, c in enumerate(ica_range):
        start_time = time.time()
        model = DC.FastICA(n_components=c, max_iter=1500)
        model.fit(train_x)
        ica_x = model.transform(train_x)
        recon = scale.inverse_transform(model.inverse_transform(ica_x))
        recon_err = np.sum((train_x - recon) ** 2, axis=1).mean()
        ica_res[i,0] = recon_err
        kurt = np.mean(scipy.stats.kurtosis(ica_x, axis=0))
        ica_res[i,1] = kurt
        print(c)
        dimred_time[i, 1] += (time.time() - start_time)/5

m0 = DC.PCA(n_components=3)
m1= DC.TruncatedSVD(n_components=3)
m0.fit(train_x)
m1.fit(train_x)
x0 = m0.transform(train_x)
x1 = m1.transform(train_x)
r0 = scale.inverse_transform(m0.inverse_transform(x0))
r1 = scale.inverse_transform(m1.inverse_transform(x1))


rp_range =list(range(1,51,2))
rp_res = np.zeros(len(rp_range))
for i, c in enumerate(rp_range):
    start_time = time.time()
    model = RP.GaussianRandomProjection(n_components=c)
    model.fit(train_x)
    rp_x = model.transform(train_x)
    recon = scale.inverse_transform(model.inverse_transform(rp_x))
    recon_err = np.sum((train_x - recon) ** 2, axis=1).mean()
    rp_res[i] = recon_err
    dimred_time[i, 2] = time.time() - start_time

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.3)
plt.subplots_adjust(left=.16)
#ax.lege
ax.set_ylabel('Reconstruction Error')
ax.set_xlabel('# Projections')
plt.title('Article: RP')
sns.lineplot(y=rp_res, x=rp_range, label='Rec Err', ax=ax)
ax.legend(loc='center right')
plt.savefig('Article rp')


tsvd_range = list(range(1,51,2))
tsvd_res = np.zeros(len(tsvd_range))
tsvd_train_x = train_x - np.min(train_x, axis=0)
tsvd_test_x = test_x - np.min(train_x, axis=0)

for i, c in enumerate(tsvd_range):
    start_time = time.time()
    model = DC.TruncatedSVD(n_components=c)
    model.fit(tsvd_train_x)
    tsvd_x = model.transform(tsvd_train_x)
    recon = scale.inverse_transform(model.inverse_transform(tsvd_x))
    recon_err = np.sum((tsvd_train_x - recon) ** 2, axis=1).mean()
    tsvd_res[i] = recon_err
    dimred_time[i, 3] = time.time() - start_time


sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.3)
plt.subplots_adjust(left=.16)
#ax.lege
ax.set_ylabel('Reconstruction Error')
ax.set_xlabel('# Components')
plt.title('Article: Reconstruction Error')
sns.lineplot(y=pca_res[:,1], x=tsvd_range, label='PCA=ICA', ax=ax)
sns.lineplot(y=rp_res, x=tsvd_range, label='RP', ax=ax)
sns.lineplot(y=tsvd_res, x=tsvd_range, label='TSVD', ax=ax)
ax.legend(loc='center')
plt.savefig('Article p2 rec error')


sns.set()
fig, ax = plt.subplots(figsize=(5, 2.5))
plt.subplots_adjust(bottom=.3)
plt.subplots_adjust(left=.16)
plt.subplots_adjust(right=.86)
#ax.lege
ax.set_ylabel('PCA+RP+TSVD Clock (ms)')
ax.set_xlabel('# Components')
ax2 = ax.twinx()
ax2.set_ylabel('ICA Clock (ms)')
ax.grid(False)
ax2.grid(False)
ax.set_ylim([0,np.max(dimred_time[:,2:]*1000*2)])
plt.title('Article: Dim Reduction Clock Times')
sns.lineplot(y=dimred_time[:,0]*1000, x=tsvd_range, label='PCA', ax=ax)
sns.lineplot(y=dimred_time[:,1]*1000, x=tsvd_range, label='ICA', ax=ax2, color='purple')
sns.lineplot(y=dimred_time[:,2]*1000, x=tsvd_range, label='RP', ax=ax)
sns.lineplot(y=dimred_time[:,3]*1000, x=tsvd_range, label='TSVD', ax=ax)
ax.legend(loc='center')
ax2.legend(loc='upper center')
plt.savefig('Article p2 clock')


model = DC.FastICA(n_components=30, random_state=0, max_iter=2000)
model.fit(train_x)
ica_x = model.transform(train_x)
recon = scale.inverse_transform(model.inverse_transform(ica_x))
recon_err = np.sum((train_x - recon) ** 2, axis=1).mean()
ica_x_tr = model.transform(train_x)
txt_mean = ica_x_tr.mean(axis=0)
txt_std = ica_x_tr.std(axis=0)
ica_x_tr = (ica_x_tr - txt_mean) / txt_std
ica_x_te = model.transform(test_x)
ica_x_te = (ica_x_te - txt_mean) / txt_std
ica_kurt = scipy.stats.kurtosis(ica_x_tr, axis=0)

kurt_order = np.argsort(ica_kurt)
ica_x_tr = ica_x_tr[:,kurt_order[20:]]
ica_x_te = ica_x_te[:,kurt_order[20:]]


sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
ax.grid(False)
plt.subplots_adjust(bottom=.3)
plt.subplots_adjust(left=.13)
plt.subplots_adjust(right=.84)
#ax.lege
ax.set_ylabel('Kurtosis')
ax.set_xlabel('Independent Component')
plt.title('Article: ICA')
sns.scatterplot(y=ica_kurt, x=range(1,31))
plt.savefig('Article p1 ica kurt')




sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
ax.grid(False)
plt.subplots_adjust(bottom=.3)
plt.subplots_adjust(left=.13)
plt.subplots_adjust(right=.84)
#ax.lege
ax.set_ylabel('Reconstruction Error')
ax.set_xlabel('# Independent Components')
plt.title('Article: ICA')
sns.lineplot(y=ica_res[:, 0], x=ica_range, label='Avg Kurt', ax=ax, color='orange')
sns.lineplot(y=ica_res[:, 0], x=ica_range, label='Rec Err', ax=ax)
ax2 = ax.twinx()
ax2.grid(False)
ax2.set_ylabel('Avg Kurtosis')
sns.lineplot(y=ica_res[:, 1], x=ica_range, ax=ax2, color='orange')
ax.legend(loc='center right')
plt.savefig('Article ica')



k_range = list(range(2,31))
km_res = np.zeros((len(k_range),2))
em_res = np.zeros((len(k_range),2))

for i, k in enumerate(k_range):
    for t in range(avg):
        model = KMeans(n_clusters=k, random_state=t)
        model.fit(train_x)
        pred_c = model.predict(train_x)
        pred_y = np.zeros(len(pred_c))
        votes = np.zeros(k)
        for c in range(k):
            temp = train_y[pred_c==c]
            val = (np.mean(temp)>=0.5) * 1
            pred_y[pred_c==c] = val
            votes[c] = val
        km_res[i,0] += accuracy_score(train_y, pred_y)/avg

        pred_c = model.predict(test_x)
        pred_y = np.zeros(len(pred_c))
        for c in range(k):
            pred_y[pred_c == c] = votes[c]
        km_res[i, 1] += accuracy_score(test_y, pred_y)/avg

        model = GaussianMixture(n_components=k, random_state=t)
        model.fit(train_x)
        pred_c = model.predict(train_x)
        pred_y = np.zeros(len(pred_c))
        votes = np.zeros(k)
        for c in range(k):
            temp = train_y[pred_c == c]
            val = (np.mean(temp) >= 0.5) * 1
            pred_y[pred_c == c] = val
            votes[c] = val
        em_res[i, 0] += accuracy_score(train_y, pred_y)/avg

        pred_c = model.predict(test_x)
        pred_y = np.zeros(len(pred_c))
        for c in range(k):
            pred_y[pred_c == c] = votes[c]
        em_res[i, 1] += accuracy_score(test_y, pred_y)/avg

    print(k)

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Article K-Means: Majority Vote Acc')
plt.xlabel('K')
plt.ylabel('Accuracy')
sns.lineplot(y=km_res[:, 0], x=k_range, label='Train')
sns.lineplot(y=km_res[:, 1], x=k_range, label='Test')
plt.savefig('Article p1 KM vote')

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Article EM: Majority Vote Acc')
plt.xlabel('# Clusters')
plt.ylabel('Accuracy')
sns.lineplot(y=em_res[:, 0], x=k_range, label='Train')
sns.lineplot(y=em_res[:, 1], x=k_range, label='Test')
plt.savefig('Article p1 EM vote')





model_pca = DC.PCA(n_components=6)
model_ica = DC.FastICA(n_components=6)
model_rp = RP.GaussianRandomProjection(n_components=6)
model_tsvd = DC.TruncatedSVD(n_components=6)

models = [model_pca, model_ica, model_rp, model_tsvd]
model_names = ['PCA', 'ICA', 'RP', 'TSVD']
train_xs = []
test_xs = []
for m in models:
    m.fit(train_x)
    tr_x = m.transform(train_x)
    te_x = m.transform(test_x)
    txt_mean = tr_x.mean(axis=0)
    txt_std = tr_x.std(axis=0)
    tr_x = (tr_x - txt_mean) / txt_std
    te_x = (te_x - txt_mean) / txt_std
    train_xs.append(tr_x)
    test_xs.append(te_x)

model_pca = DC.PCA(n_components=11)
model_ica = DC.FastICA(n_components=11)
model_rp = RP.GaussianRandomProjection(n_components=11)
model_tsvd = DC.TruncatedSVD(n_components=11)
models = [model_pca, model_ica, model_rp, model_tsvd]
train_xs_all = []
test_xs_all = []

for m in models:
    m.fit(train_x)
    tr_x = m.transform(train_x)
    te_x = m.transform(test_x)
    txt_mean = tr_x.mean(axis=0)
    txt_std = tr_x.std(axis=0)
    tr_x = (tr_x - txt_mean) / txt_std
    te_x = (te_x - txt_mean) / txt_std
    train_xs_all.append(tr_x)
    test_xs_all.append(te_x)

for c,t_x in enumerate(train_xs):
    k_range = list(range(2, 21))
    km_res = np.zeros((len(k_range), 2))
    em_res = np.zeros((len(k_range), 2))

    for i, k in enumerate(k_range):
        model = KMeans(n_clusters=k)
        model.fit(t_x)
        pred_c = model.predict(t_x)
        km_res[i, 0] = metrics.silhouette_score(t_x, pred_c)
        km_res[i, 1] = sum(np.min(cdist(t_x, model.cluster_centers_, 'euclidean'), axis=1)) / t_x.shape[0]

        model = GaussianMixture(n_components=k)
        model.fit(t_x)
        pred_c = model.predict(t_x)
        em_res[i, 0] = metrics.silhouette_score(t_x, pred_c)
        em_res[i, 1] = sum(np.min(cdist(t_x, model.means_, 'euclidean'), axis=1)) / t_x.shape[0]

        print(k)

    if True:  # p1 graphs
        sns.set()
        fig, ax = plt.subplots(figsize=(4, 2))
        plt.subplots_adjust(bottom=.26)
        plt.subplots_adjust(left=.16)
        plt.title('Article: ' + model_names[c]+' Silhouette')
        plt.xlabel('K')
        plt.ylabel('Metric Value')
        sns.lineplot(y=km_res[:, 0], x=k_range, label='K-Means')
        sns.lineplot(y=em_res[:, 0], x=k_range, label='EM')
        plt.savefig('Article p2 ' + model_names[c]+' silhouette')

        sns.set()
        fig, ax = plt.subplots(figsize=(4, 2))
        plt.subplots_adjust(bottom=.26)
        plt.subplots_adjust(left=.16)
        plt.title('Article: ' + model_names[c]+' Distortion')
        plt.xlabel('K')
        plt.ylabel('Metric Value')
        sns.lineplot(y=km_res[:, 1], x=k_range, label='K-Means')
        sns.lineplot(y=em_res[:, 1], x=k_range, label='EM')
        plt.savefig('Article p2 ' + model_names[c]+' distortion')


    scale = StandardScaler()
    scale.fit(t_x)


    km_res_2 = np.zeros((len(k_range), 2))
    em_res_2 = np.zeros((len(k_range), 2))
    for i, k in enumerate(k_range):
        model = KMeans(n_clusters=k)
        model.fit(t_x)
        pred_c = model.predict(t_x)
        km_res_2[i, 0] = metrics.adjusted_rand_score(pred_c, train_y)
        km_res_2[i, 1] = metrics.adjusted_mutual_info_score(pred_c, train_y)

        model = GaussianMixture(n_components=k)
        model.fit(t_x)
        pred_c = model.predict(t_x)
        em_res_2[i, 0] = metrics.adjusted_rand_score(pred_c, train_y)
        em_res_2[i, 1] = metrics.adjusted_mutual_info_score(pred_c, train_y)

        print(k)

    #p1b
    if True:  # p1b graphs
        sns.set()
        fig, ax = plt.subplots(figsize=(4, 2))
        plt.subplots_adjust(bottom=.26)
        plt.subplots_adjust(left=.16)
        plt.title('Article: ' + model_names[c]+' Adj Rand')
        plt.xlabel('K')
        plt.ylabel('Metric Value')
        sns.lineplot(y=km_res_2[:, 0], x=k_range, label='K-Means')
        sns.lineplot(y=em_res_2[:, 0], x=k_range, label='EM')
        plt.savefig('Article p2b ' + model_names[c]+' Rand')

        sns.set()
        fig, ax = plt.subplots(figsize=(4, 2))
        plt.subplots_adjust(bottom=.26)
        plt.subplots_adjust(left=.16)
        plt.title('Article: ' + model_names[c]+' Adj Mutual Info')
        plt.xlabel('K')
        plt.ylabel('Metric Value')
        sns.lineplot(y=km_res_2[:, 1], x=k_range, label='K-Means')
        sns.lineplot(y=em_res_2[:, 1], x=k_range, label='EM')
        plt.savefig('Article p2b ' + model_names[c]+' MI')

nn_base = MLPClassifier(hidden_layer_sizes=(256,),learning_rate='adaptive',learning_rate_init=0.001,max_iter=1000,batch_size=64,verbose=True, n_iter_no_change=1000)
nn_base.fit(train_x, train_y)
pred_y = nn_base.predict(test_x)
accuracy_score(test_y, pred_y)
test = nn_base.loss_curve_


if True:
    class Data(Dataset):
        def __init__(self, train_x, train_y):
            self.X = torch.tensor(train_x, dtype=torch.float32)
            self.y = torch.tensor(train_y, dtype=torch.long)
            self.len = train_x.shape[0]

        def __getitem__(self, index):
            return self.X[index], self.y[index]

        def __len__(self):
            return self.len

    class AverageMeter(object):
            """Computes and stores the average and current value"""

            def __init__(self):
                self.reset()

            def reset(self):
                self.val = 0
                self.avg = 0
                self.sum = 0
                self.count = 0

            def update(self, val, n=1):
                self.val = val
                self.sum += val * n
                self.count += n
                self.avg = self.sum / self.count
    def accuracy(output, target):
        """Computes the precision@k for the specified values of k"""
        batch_size = target.shape[0]

        _, pred = torch.max(output, dim=-1)

        correct = pred.eq(target).sum() * 1.0

        acc = correct / batch_size

        return acc

    # def train_cpu(epoch, data_tens_train, batch_size, model, optimizer, criterion):
    #
    #     acc = AverageMeter()
    #     splits = int(len(data_tens_train[0]) / batch_size)
    #     if splits < 2:
    #         splits = 2
    #     kf = KFold(n_splits=splits)
    #     for idx, (train_i, test_i) in enumerate(kf.split(data_tens_train[0])):
    #         data = data_tens_train[0][test_i]
    #         target = data_tens_train[1][test_i]
    #
    #         optimizer.zero_grad()
    #         out = model.forward(data)
    #         loss = criterion(out, target)
    #         loss.backward()
    #         optimizer.step()
    #
    #         batch_acc = accuracy(out, target)
    #
    #         acc.update(batch_acc, out.shape[0])
    #     # if epoch % 10 == 0:
    #     #     print(('Epoch: [{0}][{1}/{2}]\t'
    #     #            'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
    #     #           .format(epoch, idx, int(len(data_tens_train[0]) / batch_size),top1=acc))
    #     torch.cuda.empty_cache()
    #
    # def validate_cpu(epoch, data_tens, batch_size, model, criterion, num_class):
    #
    #     acc = AverageMeter()
    #     splits = int(len(data_tens[0]) / batch_size)
    #     if splits < 2:
    #         splits = 2
    #     kf = KFold(n_splits=splits)
    #     # evaluation loop
    #     for idx, (train_i, test_i) in enumerate(kf.split(data_tens[0])):
    #         data = data_tens[0][test_i]
    #         target = data_tens[1][test_i]
    #         # torch.cuda.empty_cache()
    #
    #         with torch.no_grad():
    #             out = model(data)
    #             loss = criterion(out, target)
    #
    #         batch_acc = accuracy(out, target)
    #         acc.update(batch_acc, out.shape[0])
    #     print("Epoch ",epoch ,"\t","* Prec @1: {top1.avg:.4f}".format(top1=acc))
    #     return acc.avg, 2

    def train_cpu(epoch, train_loader, batch_size, model, optimizer, criterion):

        acc = AverageMeter()

        for idx, (data, target) in enumerate(train_loader):

            optimizer.zero_grad()
            out = model.forward(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            batch_acc = accuracy(out, target)

            acc.update(batch_acc, out.shape[0])
        torch.cuda.empty_cache()

    def validate_cpu(epoch, val_loader, batch_size, model, criterion, num_class):

        acc = AverageMeter()

        # evaluation loop
        for idx, (data, target) in enumerate(val_loader):

            with torch.no_grad():
                out = model(data)
                loss = criterion(out, target)

            batch_acc = accuracy(out, target)
            acc.update(batch_acc, out.shape[0])
        if epoch%20 == 0:
            print("Epoch ",epoch ,"\t","* Prec @1: {top1.avg:.4f}".format(top1=acc))
        return acc.avg, 2

    class LayerNet_n_cpu(nn.Module):
        def __init__(self, input_dim, hidden_size, layers, num_classes):
            """
            :param input_dim: input feature dimension
            :param hidden_size: hidden dimension
            :param num_classes: total number of classes
            """
            super().__init__()
            #############################################################################
            # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
            #############################################################################
            self.layers = []
            self.l1 = nn.Linear(input_dim, hidden_size)
            for i in range(layers):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            # self.l1a = nn.Sigmoid(hidden_size, hidden_size)
            self.last = nn.Linear(hidden_size, num_classes)

        def forward(self, x):

            x = F.leaky_relu(self.l1(torch.flatten(x, 1)))
            for layer in self.layers:
                x = F.leaky_relu(layer(torch.flatten(x, 1)))

            return self.last(x)

    class LayerNet_cpu_1(nn.Module):
        def __init__(self, input_dim, hidden_size, num_classes):
            """
            :param input_dim: input feature dimension
            :param hidden_size: hidden dimension
            :param num_classes: total number of classes
            """
            super().__init__()
            #############################################################################
            # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
            #############################################################################

            self.l1 = nn.Linear(input_dim, hidden_size)
            self.last = nn.Linear(hidden_size, num_classes)

        def forward(self, x):

            x = torch.relu(self.l1(x))
            x = self.last(x)
            return x

    class LayerNet_cpu(nn.Module):
        def __init__(self, input_dim, hidden_size):
            """
            :param input_dim: input feature dimension
            :param hidden_size: hidden dimension
            :param num_classes: total number of classes
            """
            super().__init__()
            #############################################################################
            # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
            #############################################################################

            self.l1 = nn.Linear(input_dim, hidden_size)
            self.last = nn.Linear(hidden_size, 1)

        def forward(self, x):

            x = torch.relu(self.l1(x))
            x = self.last(x)
            return torch.flatten(x)


    learning_rate = 0.001
    momentum = 0.8
    reg = 0.0001
    batch_size = 64
    val_batch_size = 64

    tr_x, v_x, tr_y, v_y = train_test_split(train_x, train_y, test_size=0.2, random_state=100)

    train_loader = DataLoader(Data(tr_x, tr_y),batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Data(v_x, v_y),batch_size=len(v_x), shuffle=True)
    test_loader = DataLoader(Data(test_x, test_y), batch_size = len(test_y), shuffle=True)

    # tr_x_tens, v_x_tens, tr_y_tens, v_y_tens = train_test_split(train_x, train_y, test_size=0.2, random_state=100)
    # tr_x_tens = torch.tensor(tr_x_tens, dtype=torch.float32)
    # v_x_tens = torch.tensor(v_x_tens, dtype=torch.float32)
    # tr_y_tens = torch.tensor(tr_y_tens, dtype=torch.long)
    # v_y_tens = torch.tensor(v_y_tens, dtype=torch.long)


    def nn_results_torch(train_loader, val_loader, features=11, avg=10, lr_init = 0.01, batch_size = 64, epochs=200):

        momentum = 0.9
        reg = 0.0001
        val_batch_size = 64
        nn_accs = np.zeros((epochs, 3))
        nn_accs[:, 0] = range(epochs)
        criterion = nn.CrossEntropyLoss()
        for a in range(avg):
            learning_rate = lr_init
            nn_model = LayerNet_cpu_1(features, 512,2)
            # optimizer = torch.optim.SGD(nn_model.parameters(), lr_init, momentum=momentum, weight_decay=reg)
            optimizer = torch.optim.Adam(nn_model.parameters(), learning_rate, weight_decay=0.0001)

            best = 0.0
            counter = 0

            for epoch in range(epochs):

                if counter >= 20:

                    learning_rate /= 5
                    print('lr decrease', learning_rate)
                    # optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate, momentum=0.8, weight_decay=0.0001)
                    optimizer = torch.optim.Adam(nn_model.parameters(), learning_rate,  weight_decay=0.0001)

                    counter = 0

                train_cpu(epoch, train_loader, batch_size, nn_model, optimizer, criterion)

                tr_acc, tr_cm = validate_cpu(epoch, train_loader, val_batch_size, nn_model, criterion, 2)
                acc, cm = validate_cpu(epoch, val_loader, val_batch_size, nn_model, criterion, 2)


                nn_accs[epoch, 1] += tr_acc/avg
                nn_accs[epoch, 2] += acc/avg

                if acc > best:
                    best = acc
                    counter = 0
                else:
                    counter +=1
        return nn_accs

if False:
    iter_range=[1,2,3,4,5,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000]
    def get_nn_results(train_x, train_y, avg=10, iter_range=iter_range, hidden=(256,), lr='adaptive',lr_init = 0.001, v_frac = 0.2, batch_size = 64):
        res = np.zeros((len(iter_range),3))
        res[:,0] = iter_range
        t_x, v_x, t_y, v_y = train_test_split(train_x, train_y, test_size=v_frac, random_state=100)
        for i, iters in enumerate(iter_range):
            for c in range(avg):
                nn_model = MLPClassifier(hidden_layer_sizes=hidden, learning_rate=lr, learning_rate_init=lr_init, max_iter=iters, batch_size=batch_size, verbose=False, n_iter_no_change=iters)
                nn_model.fit(t_x, t_y)
                pred_t = nn_model.predict(t_x)
                pred_v = nn_model.predict(v_x)
                res[i,1] += accuracy_score(t_y, pred_t)/avg
                res[i,2] += accuracy_score(v_y, pred_v)/avg
                print(c, iters)
        return res

tr_x, v_x, tr_y, v_y = train_test_split(train_x, train_y, test_size=0.2, random_state=100)
train_xs_comb = []
test_xs_comb = []
for i in range(4):
    tr_x = np.hstack((train_x, train_xs[i]))
    te_x = np.hstack((test_x, test_xs[i]))
    txt_mean = tr_x.mean(axis=0)
    txt_std = tr_x.std(axis=0)
    tr_x = (tr_x - txt_mean) / txt_std
    te_x = (te_x - txt_mean) / txt_std
    train_xs_comb.append(tr_x)
    test_xs_comb.append(te_x)

train_xs_comb_all = []
test_xs_comb_all = []
for i in range(4):
    tr_x = np.hstack((train_x, train_xs_all[i]))
    te_x = np.hstack((test_x, test_xs_all[i]))
    txt_mean = tr_x.mean(axis=0)
    txt_std = tr_x.std(axis=0)
    tr_x = (tr_x - txt_mean) / txt_std
    te_x = (te_x - txt_mean) / txt_std
    train_xs_comb_all.append(tr_x)
    test_xs_comb_all.append(te_x)

train_loader_xs = []
train_loader_xs_all = []
train_loader_xs_comb = []
train_loader_xs_comb_all = []
test_loader_xs = []
test_loader_xs_all = []
test_loader_xs_comb = []
test_loader_xs_comb_all = []

for t in train_xs:
    train_loader_xs.append(DataLoader(Data(t,train_y),batch_size=batch_size, shuffle=True))
for t in train_xs_all:
    train_loader_xs_all.append(DataLoader(Data(t,train_y),batch_size=batch_size, shuffle=True))
for t in train_xs_comb:
    train_loader_xs_comb.append(DataLoader(Data(t,train_y),batch_size=batch_size, shuffle=True))
for t in train_xs_comb_all:
    train_loader_xs_comb_all.append(DataLoader(Data(t,train_y),batch_size=batch_size, shuffle=True))
for t in test_xs:
    test_loader_xs.append(DataLoader(Data(t,test_y),batch_size=len(test_y), shuffle=True))
for t in test_xs_all:
    test_loader_xs_all.append(DataLoader(Data(t,test_y),batch_size=len(test_y), shuffle=True))
for t in test_xs_comb:
    test_loader_xs_comb.append(DataLoader(Data(t,test_y),batch_size=len(test_y), shuffle=True))
for t in test_xs_comb_all:
    test_loader_xs_comb_all.append(DataLoader(Data(t,test_y),batch_size=len(test_y), shuffle=True))

train_loader = DataLoader(Data(train_x, train_y),batch_size=batch_size, shuffle=True)
test_loader = DataLoader(Data(test_x, test_y),batch_size=len(test_y), shuffle=True)

nn_res = np.zeros((33,2))

start_time = time.time()
nn_base_res = nn_results_torch(train_loader, test_loader, avg = 10, lr_init=0.01)
nn_res[0,0] = (time.time()-start_time)/10
nn_res[0,1] = np.max(nn_base_res[:,2])

#dimred
if True:
    #6 feat
    if True:
        start_time = time.time()
        nn_pca_res_0 = nn_results_torch(train_loader_xs[0], test_loader_xs[0],features=6, avg = 10, lr_init=0.01)
        nn_res[1, 0] = (time.time() - start_time) / 10
        nn_res[1, 1] = np.max(nn_pca_res_0[:, 2])

        start_time = time.time()
        nn_ica_res_0 = nn_results_torch(train_loader_xs[1], test_loader_xs[1],features=6, avg = 10, lr_init=0.01)
        nn_res[2, 0] = (time.time() - start_time) / 10
        nn_res[2, 1] = np.max(nn_ica_res_0[:, 2])

        start_time = time.time()
        nn_rp_res_0 = nn_results_torch(train_loader_xs[2], test_loader_xs[2],features=6, avg = 10, lr_init=0.01)
        nn_res[3, 0] = (time.time() - start_time) / 10
        nn_res[3, 1] = np.max(nn_rp_res_0[:, 2])

        start_time = time.time()
        nn_tsvd_res_0 = nn_results_torch(train_loader_xs[3], test_loader_xs[3],features=6, avg = 10, lr_init=0.01)
        nn_res[4, 0] = (time.time() - start_time) / 10
        nn_res[4, 1] = np.max(nn_tsvd_res_0[:, 2])

    # 17 feat
    if True:
        start_time = time.time()
        nn_pca_res_1 = nn_results_torch(train_loader_xs_comb[0], test_loader_xs_comb[0], features=17, avg=10, lr_init=0.01)
        nn_res[9, 0] = (time.time() - start_time) / 10
        nn_res[9, 1] = np.max(nn_pca_res_1[:, 2])

        start_time = time.time()
        nn_ica_res_1 = nn_results_torch(train_loader_xs_comb[1], test_loader_xs_comb[1], features=17, avg=10, lr_init=0.01)
        nn_res[10, 0] = (time.time() - start_time) / 10
        nn_res[10, 1] = np.max(nn_ica_res_1[:, 2])

        start_time = time.time()
        nn_rp_res_1 = nn_results_torch(train_loader_xs_comb[2], test_loader_xs_comb[2], features=17, avg=10, lr_init=0.01)
        nn_res[11, 0] = (time.time() - start_time) / 10
        nn_res[11, 1] = np.max(nn_rp_res_1[:, 2])

        start_time = time.time()
        nn_tsvd_res_1 = nn_results_torch(train_loader_xs_comb[3], test_loader_xs_comb[3], features=17, avg=10, lr_init=0.01)
        nn_res[12, 0] = (time.time() - start_time) / 10
        nn_res[12, 1] = np.max(nn_tsvd_res_1[:, 2])

    # 11 feat
    if True:
        start_time = time.time()
        nn_pca_res_2 = nn_results_torch(train_loader_xs_all[0], test_loader_xs_all[0], features=11, avg=10, lr_init=0.01)
        nn_res[5, 0] = (time.time() - start_time) / 10
        nn_res[5, 1] = np.max(nn_pca_res_2[:, 2])

        start_time = time.time()
        nn_ica_res_2 = nn_results_torch(train_loader_xs_all[1], test_loader_xs_all[1], features=11, avg=10, lr_init=0.01)
        nn_res[6, 0] = (time.time() - start_time) / 10
        nn_res[6, 1] = np.max(nn_ica_res_2[:, 2])

        start_time = time.time()
        nn_rp_res_2 = nn_results_torch(train_loader_xs_all[2], test_loader_xs_all[2], features=11, avg=10, lr_init=0.01)
        nn_res[7, 0] = (time.time() - start_time) / 10
        nn_res[7, 1] = np.max(nn_rp_res_2[:, 2])

        start_time = time.time()
        nn_tsvd_res_2 = nn_results_torch(train_loader_xs_all[3], test_loader_xs_all[3], features=11, avg=10, lr_init=0.01)
        nn_res[8, 0] = (time.time() - start_time) / 10
        nn_res[8, 1] = np.max(nn_tsvd_res_2[:, 2])

    # 22 feat
    if True:
        start_time = time.time()
        nn_pca_res_3 = nn_results_torch(train_loader_xs_comb_all[0], test_loader_xs_comb_all[0], features=22, avg=10, lr_init=0.01)
        nn_res[13, 0] = (time.time() - start_time) / 10
        nn_res[13, 1] = np.max(nn_pca_res_3[:, 2])

        start_time = time.time()
        nn_ica_res_3 = nn_results_torch(train_loader_xs_comb_all[1], test_loader_xs_comb_all[1], features=22, avg=10, lr_init=0.01)
        nn_res[14, 0] = (time.time() - start_time) / 10
        nn_res[14, 1] = np.max(nn_ica_res_3[:, 2])

        start_time = time.time()
        nn_rp_res_3 = nn_results_torch(train_loader_xs_comb_all[2], test_loader_xs_comb_all[2], features=22, avg=10, lr_init=0.01)
        nn_res[15, 0] = (time.time() - start_time) / 10
        nn_res[15, 1] = np.max(nn_rp_res_3[:, 2])

        start_time = time.time()
        nn_tsvd_res_3 = nn_results_torch(train_loader_xs_comb_all[3], test_loader_xs_comb_all[3], features=22, avg=10, lr_init=0.01)
        nn_res[16, 0] = (time.time() - start_time) / 10
        nn_res[16, 1] = np.max(nn_tsvd_res_3[:, 2])

# p3 graphs
if True:
    sns.set()
    fig, ax = plt.subplots(figsize=(4.5, 2.25))
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.16)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Iterations')
    plt.title('Article 6: NN Training')
    sns.lineplot(y=nn_base_res[:,1], x=range(200), label='Base (11 feat)', ax=ax)
    sns.lineplot(y=nn_pca_res_0[:,1], x=range(200), label='PCA (6 feat)', ax=ax)
    sns.lineplot(y=nn_ica_res_0[:,1], x=range(200), label='ICA (6 feat)', ax=ax)
    sns.lineplot(y=nn_rp_res_0[:,1], x=range(200), label='RP (6 feat)', ax=ax)
    sns.lineplot(y=nn_tsvd_res_0[:,1], x=range(200), label='TSVD (6 feat)', ax=ax)
    ax.legend(loc='lower center')
    plt.savefig('Article p3 nn train')

    sns.set()
    fig, ax = plt.subplots(figsize=(4.5, 2.25))
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.16)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Iterations')
    plt.title('Article 6: NN Validation')
    sns.lineplot(y=nn_base_res[:,2], x=range(200), label='Base (11 feat)', ax=ax)
    sns.lineplot(y=nn_pca_res_0[:,2], x=range(200), label='PCA (6 feat)', ax=ax)
    sns.lineplot(y=nn_ica_res_0[:,2], x=range(200), label='ICA (6 feat)', ax=ax)
    sns.lineplot(y=nn_rp_res_0[:,2], x=range(200), label='RP (6 feat)', ax=ax)
    sns.lineplot(y=nn_tsvd_res_0[:,2], x=range(200), label='TSVD (6 feat)', ax=ax)
    ax.legend(loc='lower center')
    plt.savefig('Article p3 nn val')



    sns.set()
    fig, ax = plt.subplots(figsize=(4.5, 2.25))
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.16)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Iterations')
    plt.title('Article 17: NN Training')
    sns.lineplot(y=nn_base_res[:,1], x=range(200), label='Base (11 feat)', ax=ax)
    sns.lineplot(y=nn_pca_res_1[:,1], x=range(200), label='Base+PCA (17 feat)', ax=ax)
    sns.lineplot(y=nn_ica_res_1[:,1], x=range(200), label='Base+ICA (17 feat)', ax=ax)
    sns.lineplot(y=nn_rp_res_1[:,1], x=range(200), label='Base+RP (17 feat)', ax=ax)
    sns.lineplot(y=nn_tsvd_res_1[:,1], x=range(200), label='Base+TSVD (17 feat)', ax=ax)
    ax.legend(loc='lower center')
    plt.savefig('Article p3b nn train')

    sns.set()
    fig, ax = plt.subplots(figsize=(4.5, 2.25))
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.16)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Iterations')
    plt.title('Article 17: NN Validation')
    sns.lineplot(y=nn_base_res[:,2], x=range(200), label='Base (11 feat)', ax=ax)
    sns.lineplot(y=nn_pca_res_1[:,2], x=range(200), label='Base+PCA (17 feat)', ax=ax)
    sns.lineplot(y=nn_ica_res_1[:,2], x=range(200), label='Base+ICA (17 feat)', ax=ax)
    sns.lineplot(y=nn_rp_res_1[:,2], x=range(200), label='Base+RP (17 feat)', ax=ax)
    sns.lineplot(y=nn_tsvd_res_1[:,2], x=range(200), label='Base+Trunc SVD (17 feat)', ax=ax)
    ax.legend(loc='lower center')
    plt.savefig('Article p3b nn val')



    sns.set()
    fig, ax = plt.subplots(figsize=(4.5, 2.25))
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.16)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Iterations')
    plt.title('Article 11: NN Training')
    sns.lineplot(y=nn_base_res[:,1], x=range(200), label='Base (11 feat)', ax=ax)
    sns.lineplot(y=nn_pca_res_2[:,1], x=range(200), label='PCA (11 feat)', ax=ax)
    sns.lineplot(y=nn_ica_res_2[:,1], x=range(200), label='ICA (11 feat)', ax=ax)
    sns.lineplot(y=nn_rp_res_2[:,1], x=range(200), label='RP (11 feat)', ax=ax)
    sns.lineplot(y=nn_tsvd_res_2[:,1], x=range(200), label='TSVD (11 feat)', ax=ax)
    ax.legend(loc='lower center')
    plt.savefig('Article p3c nn train')

    sns.set()
    fig, ax = plt.subplots(figsize=(4.5, 2.25))
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.16)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Iterations')
    plt.title('Article 11: NN Validation')
    sns.lineplot(y=nn_base_res[:,2], x=range(200), label='Base (11 feat)', ax=ax)
    sns.lineplot(y=nn_pca_res_2[:,2], x=range(200), label='PCA (11 feat)', ax=ax)
    sns.lineplot(y=nn_ica_res_2[:,2], x=range(200), label='ICA (11 feat)', ax=ax)
    sns.lineplot(y=nn_rp_res_2[:,2], x=range(200), label='RP (11 feat)', ax=ax)
    sns.lineplot(y=nn_tsvd_res_2[:,2], x=range(200), label='TSVD (11 feat)', ax=ax)
    ax.legend(loc='lower center')
    plt.savefig('Article p3c nn val')

    sns.set()
    fig, ax = plt.subplots(figsize=(4.5, 2.25))
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.16)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Iterations')
    plt.title('Article 22: NN Training')
    sns.lineplot(y=nn_base_res[:, 1], x=range(200), label='Base (11 feat)', ax=ax)
    sns.lineplot(y=nn_pca_res_3[:, 1], x=range(200), label='PCA (22 feat)', ax=ax)
    sns.lineplot(y=nn_ica_res_3[:, 1], x=range(200), label='ICA (22 feat)', ax=ax)
    sns.lineplot(y=nn_rp_res_3[:, 1], x=range(200), label='RP (22 feat)', ax=ax)
    sns.lineplot(y=nn_tsvd_res_3[:, 1], x=range(200), label='TSVD (22 feat)', ax=ax)
    ax.legend(loc='lower center')
    plt.savefig('Article p3d nn train')

    sns.set()
    fig, ax = plt.subplots(figsize=(4.5, 2.25))
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.16)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Iterations')
    plt.title('Article 22: NN Validation')
    sns.lineplot(y=nn_base_res[:, 2], x=range(200), label='Base (11 feat)', ax=ax)
    sns.lineplot(y=nn_pca_res_3[:, 2], x=range(200), label='PCA (22 feat)', ax=ax)
    sns.lineplot(y=nn_ica_res_3[:, 2], x=range(200), label='ICA (22 feat)', ax=ax)
    sns.lineplot(y=nn_rp_res_3[:, 2], x=range(200), label='RP (22 feat)', ax=ax)
    sns.lineplot(y=nn_tsvd_res_3[:, 2], x=range(200), label='TSVD (22 feat)', ax=ax)
    ax.legend(loc='lower center')
    plt.savefig('Article p3d nn val')


p3_names = ['Base (B) 11', 'PCA 6', 'ICA 6', 'RP 6', 'TSVD 6', 'PCA 11', 'ICA 11', 'RP 11', 'TSVD 11', 'B+PCA 17', 'B+ICA 17', 'B+RP 17', 'B+TSVD 17','B+PCA 21', 'B+ICA 21', 'B+RP 21', 'B+TSVD 21']
fig, ax = plt.subplots(figsize=(8, 2.75))
ax2 = ax.twinx()
ax2.grid(False)
ax.grid(False)
plt.subplots_adjust(bottom=.4)
plt.subplots_adjust(left=.08)
plt.title('Article NN: Clock Time and Accuracy')
plt.xlabel('Dataset')
ax.set_ylabel('Clock Time (Sec)')
ax2.set_ylabel('Validation Accuracy')
k = np.arange(17)
ax.bar(k, nn_res[:17,0], 0.3)
ax2.set_ylim(bottom=0,top=1.2)
ax.set_ylim([0,max(nn_res[:17,0])*2])
ax2.scatter(k[0],nn_res[0,1], color='red')
ax2.scatter(k[1:5],nn_res[1:5,1], color='blue')
ax2.scatter(k[5:9],nn_res[5:9,1], color='green')
ax2.scatter(k[9:13],nn_res[9:13,1], color='blue')
ax2.scatter(k[13:17],nn_res[13:17,1], color='green')
plt.xticks(k,p3_names)
fig.autofmt_xdate(rotation=45)
flipper = False
for xy in zip(k, nn_res[:17,1]):
    t = round(xy[1], 3)
    ax2.annotate('%s' % t, xy=xy, textcoords='data', fontsize=10, ha='center', verticalalignment='bottom', rotation=45)
    # if flipper:
    #     flipper = False
    #     ax2.annotate('%s' % t, xy=xy, textcoords='data', fontsize=10, ha='center', verticalalignment='top', rotation=45)
    # else:
    #     flipper = True
    #     ax2.annotate('%s' % t, xy=xy, textcoords='data', fontsize=10, ha='center', verticalalignment='bottom')
for xy in zip(k, nn_res[:17,0]):
    t = round(xy[1], 2)
    ax.annotate('%s' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
plt.subplots_adjust(bottom=.35)
plt.savefig('p3 wall clock')




train_xs_km = []
test_xs_km = []

for i in range(2, 31):
    model = KMeans(n_clusters=i)
    model.fit(train_x)
    centers = model.cluster_centers_
    dist_km_tr = np.zeros((len(train_x), i))
    for k in range(i):
        temp = np.sqrt(np.sum((train_x - centers[k, :]) ** 2, axis=1))
        dist_km_tr[:, k] = temp

    dist_km_te = np.zeros((len(test_x), i))
    for k in range(i):
        temp = np.sqrt(np.sum((test_x - centers[k, :]) ** 2, axis=1))
        dist_km_te[:, k] = temp

    tr_x = dist_km_tr
    te_x = dist_km_te

    txt_mean = tr_x.mean(axis=0)
    txt_std = tr_x.std(axis=0)
    tr_x = (tr_x - txt_mean) / txt_std
    te_x = (te_x - txt_mean) / txt_std
    train_xs_km.append(tr_x)
    test_xs_km.append(te_x)

train_loader_xs_km = []
test_loader_xs_km = []

for t in train_xs_km:
    train_loader_xs_km.append(DataLoader(Data(t, train_y), batch_size=batch_size, shuffle=True))
for t in test_xs_km:
    test_loader_xs_km.append(DataLoader(Data(t, test_y), batch_size=len(test_y), shuffle=True))

train_xs_em = []
test_xs_em = []

for i in range(2, 31):
    model = GaussianMixture(n_components=i)
    model.fit(train_x)
    tr_em_x = model.predict_proba(train_x)
    te_em_x = model.predict_proba(test_x)

    tr_x = tr_em_x
    te_x = te_em_x

    txt_mean = tr_x.mean(axis=0)
    txt_std = tr_x.std(axis=0)
    tr_x = (tr_x - txt_mean) / txt_std
    te_x = (te_x - txt_mean) / txt_std
    train_xs_em.append(tr_x)
    test_xs_em.append(te_x)

train_loader_xs_em = []
test_loader_xs_em = []

for t in train_xs_em:
    train_loader_xs_em.append(DataLoader(Data(t, train_y), batch_size=batch_size, shuffle=True))
for t in test_xs_em:
    test_loader_xs_em.append(DataLoader(Data(t, test_y), batch_size=len(test_y), shuffle=True))

train_loader = DataLoader(Data(train_x, train_y), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(Data(test_x, test_y), batch_size=len(test_y), shuffle=True)

nn_km_curves = []
nn_km_res = np.zeros((len(train_loader_xs_km), 3))
for i in range(len(train_loader_xs_km)):
    start_time = time.time()
    res = nn_results_torch(train_loader_xs_km[i], test_loader_xs_km[i], features=(i + 2), avg=5, lr_init=0.01)
    nn_km_curves.append(res)
    nn_km_res[i, 0] = (time.time() - start_time) / 5
    nn_km_res[i, 1] = np.max(res[:, 1])
    nn_km_res[i, 2] = np.max(res[:, 2])

nn_em_curves = []
nn_em_res = np.zeros((len(train_loader_xs_em), 3))
for i in range(len(train_loader_xs_em)):
    start_time = time.time()
    res = nn_results_torch(train_loader_xs_em[i], test_loader_xs_em[i], features=(i + 2), avg=5, lr_init=0.01)
    nn_em_curves.append(res)
    nn_em_res[i, 0] = (time.time() - start_time) / 5
    nn_em_res[i, 1] = np.max(res[:, 1])
    nn_em_res[i, 2] = np.max(res[:, 2])


train_xs_km_2 = []
test_xs_km_2 = []

for i in range(2,31):
    model = KMeans(n_clusters=i)
    model.fit(train_x)
    centers = model.cluster_centers_
    dist_km_tr = np.zeros((len(train_x),i))
    for k in range(i):
        temp = np.sqrt(np.sum((train_x - centers[k,:]) **2, axis=1))
        dist_km_tr[:,k] = temp
    
    dist_km_te = np.zeros((len(test_x), i))
    for k in range(i):
        temp = np.sqrt(np.sum((test_x - centers[k, :]) ** 2, axis=1))
        dist_km_te[:, k] = temp
    
    tr_x = np.hstack((train_x, dist_km_tr))
    te_x = np.hstack((test_x, dist_km_te))
    
    txt_mean = tr_x.mean(axis=0)
    txt_std = tr_x.std(axis=0)
    tr_x = (tr_x - txt_mean) / txt_std
    te_x = (te_x - txt_mean) / txt_std
    train_xs_km_2.append(tr_x)
    test_xs_km_2.append(te_x)


train_loader_xs_km_2 = []
test_loader_xs_km_2 = []

for t in train_xs_km_2:
    train_loader_xs_km_2.append(DataLoader(Data(t,train_y),batch_size=batch_size, shuffle=True))
for t in test_xs_km_2:
    test_loader_xs_km_2.append(DataLoader(Data(t,test_y),batch_size=len(test_y), shuffle=True))

train_xs_em_2 = []
test_xs_em_2 = []

for i in range(2, 31):

    model = GaussianMixture(n_components=i)
    model.fit(train_x)
    tr_em_x = model.predict_proba(train_x)
    te_em_x = model.predict_proba(test_x)

    tr_x = np.hstack((train_x, tr_em_x))
    te_x = np.hstack((test_x, te_em_x))

    txt_mean = tr_x.mean(axis=0)
    txt_std = tr_x.std(axis=0)
    tr_x = (tr_x - txt_mean) / txt_std
    te_x = (te_x - txt_mean) / txt_std
    train_xs_em_2.append(tr_x)
    test_xs_em_2.append(te_x)

train_loader_xs_em_2 = []
test_loader_xs_em_2 = []

for t in train_xs_em_2:
    train_loader_xs_em_2.append(DataLoader(Data(t, train_y), batch_size=batch_size, shuffle=True))
for t in test_xs_em_2:
    test_loader_xs_em_2.append(DataLoader(Data(t, test_y), batch_size=len(test_y), shuffle=True))

train_loader = DataLoader(Data(train_x, train_y),batch_size=batch_size, shuffle=True)
test_loader = DataLoader(Data(test_x, test_y),batch_size=len(test_y), shuffle=True)

nn_km_curves_2 = []
nn_km_res_2 = np.zeros((len(train_loader_xs_km_2),3))
i = 0
for i in range(len(train_loader_xs_km_2)):
    start_time = time.time()
    res = nn_results_torch(train_loader_xs_km_2[i], test_loader_xs_km_2[i], features=(i+2)+11, avg=5, lr_init=0.01)
    nn_km_curves_2.append(res)
    nn_km_res_2[i,0] = (time.time() - start_time) / 5
    nn_km_res_2[i, 1] = np.max(res[:, 1])
    nn_km_res_2[i, 2] = np.max(res[:, 2])

nn_em_curves_2 = []
nn_em_res_2 = np.zeros((len(train_loader_xs_em_2),3))

for i in range(len(train_loader_xs_em_2)):
    start_time = time.time()
    res = nn_results_torch(train_loader_xs_em_2[i], test_loader_xs_em_2[i], features=(i+2)+11, avg=5, lr_init=0.01)
    nn_em_curves_2.append(res)
    nn_em_res_2[i,0] = (time.time() - start_time) / 5
    nn_em_res_2[i, 1] = np.max(res[:, 1])
    nn_em_res_2[i, 2] = np.max(res[:, 2])


fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.27)
plt.subplots_adjust(left=.16)
plt.subplots_adjust(right=.98)
plt.title('Article NN: KM Clustering Accuracy')
ax.set_xlabel('# Clusters (0 is NN with original data)')
ax.set_ylabel('Accuracy')
sns.lineplot(y=nn_km_res[:, 1], x=range(2,31), label='Train', ax=ax, color = 'blue')
sns.lineplot(y=nn_km_res[:, 2], x=range(2,31), label='Test', ax=ax, color = 'red')
ax.scatter(0,np.max(nn_base_res[:,1]), color='blue')
ax.scatter(0,np.max(nn_base_res[:,2]), color='red')
ax.legend(loc = 'lower center')
plt.savefig('Article p4 KM')

fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.27)
plt.subplots_adjust(left=.16)
plt.subplots_adjust(right=.98)
plt.title('Article NN: EM Clustering Accuracy')
ax.set_xlabel('# Clusters (0 is NN with original data)')
ax.set_ylabel('Accuracy')
sns.lineplot(y=nn_em_res[:, 1], x=range(2,31), label='Train', ax=ax, color = 'blue')
sns.lineplot(y=nn_em_res[:, 2], x=range(2,31), label='Test', ax=ax, color = 'red')
ax.scatter(0,np.max(nn_base_res[:,1]), color='blue')
ax.scatter(0,np.max(nn_base_res[:,2]), color='red')
ax.legend(loc = 'upper center')
plt.savefig('Article p4 EM')

fig, ax = plt.subplots(figsize=(4, 2))
# ax.grid(False)
plt.subplots_adjust(bottom=.27)
plt.subplots_adjust(left=.16)
plt.subplots_adjust(right=.98)
plt.title('Article NN: KM Appended with Original Data')
ax.set_xlabel('# Clusters (0 is NN with original data)')
ax.set_ylabel('Accuracy')
sns.lineplot(y=nn_km_res_2[:, 1], x=range(2,31), label='Train', ax=ax, color = 'blue')
sns.lineplot(y=nn_km_res_2[:, 2], x=range(2,31), label='Test', ax=ax, color = 'red')
ax.scatter(0,np.max(nn_base_res[:,1]), color='blue')
ax.scatter(0,np.max(nn_base_res[:,2]), color='red')
ax.legend(loc = 'lower center')
plt.savefig('Article p4b KM')

fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.27)
plt.subplots_adjust(left=.16)
plt.subplots_adjust(right=.98)
plt.title('Article NN: EM Appended with Original Data')
ax.set_xlabel('# Clusters (0 is NN with original data)')
ax2.set_ylabel('Clock Time (Sec)')
ax.set_ylabel('Accuracy')
sns.lineplot(y=nn_em_res_2[:, 1], x=range(2,31), label='Train', ax=ax, color = 'blue')
sns.lineplot(y=nn_em_res_2[:, 2], x=range(2,31), label='Test', ax=ax, color = 'red')
ax.scatter(0,np.max(nn_base_res[:,1]), color='blue')
ax.scatter(0,np.max(nn_base_res[:,2]), color='red')
ax.legend(loc = 'lower center')
plt.savefig('Article p4b EM')

np.max(nn_km_curves[18][:, 2])
if True:
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.25)
    plt.subplots_adjust(left=.16)
    plt.subplots_adjust(right=.98)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Iterations')
    plt.title('Cluster NN Training Curve')
    sns.lineplot(y=nn_base_res[:,1], x=range(200), label='Base 11', ax=ax)
    sns.lineplot(y=nn_km_curves[8][:,1], x=range(200), label='KM 10', ax=ax)
    sns.lineplot(y=nn_km_curves[18][:, 1], x=range(200), label='KM 20', ax=ax)
    sns.lineplot(y=nn_km_curves[28][:, 1], x=range(200), label='KM 30', ax=ax)
    sns.lineplot(y=nn_em_curves[8][:, 1], x=range(200), label='EM 10', ax=ax)
    sns.lineplot(y=nn_em_curves[18][:, 1], x=range(200), label='EM 20', ax=ax)
    sns.lineplot(y=nn_em_curves[28][:, 1], x=range(200), label='EM 30', ax=ax)
    ax.legend(loc='lower center', fontsize=8)
    plt.savefig('Article p4b KMEM train')

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.25)
    plt.subplots_adjust(left=.16)
    plt.subplots_adjust(right=.98)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Iterations')
    plt.title('Cluster NN Testing Curve')
    sns.lineplot(y=nn_base_res[:, 2], x=range(200), label='Base 11', ax=ax)
    sns.lineplot(y=nn_km_curves[8][:, 2], x=range(200), label='KM 10', ax=ax)
    sns.lineplot(y=nn_km_curves[18][:, 2], x=range(200), label='KM 20', ax=ax)
    sns.lineplot(y=nn_km_curves[28][:, 2], x=range(200), label='KM 30', ax=ax)
    sns.lineplot(y=nn_em_curves[8][:, 2], x=range(200), label='EM 10', ax=ax)
    sns.lineplot(y=nn_em_curves[18][:, 2], x=range(200), label='EM 20', ax=ax)
    sns.lineplot(y=nn_em_curves[28][:, 2], x=range(200), label='EM 30', ax=ax)
    ax.legend(loc='lower center', fontsize=8)
    plt.savefig('Article p4b KMEM test')


if True:
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.25)
    plt.subplots_adjust(left=.16)
    plt.subplots_adjust(right=.98)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Iterations')
    plt.title('Cluster NN Training Curve')
    sns.lineplot(y=nn_base_res[:,1], x=range(200), label='Base 11', ax=ax)
    sns.lineplot(y=nn_km_curves_2[8][:,1], x=range(200), label='B+KM10 21', ax=ax)
    sns.lineplot(y=nn_km_curves_2[18][:, 1], x=range(200), label='B+KM20 31', ax=ax)
    sns.lineplot(y=nn_km_curves_2[28][:, 1], x=range(200), label='B+KM30 41', ax=ax)
    sns.lineplot(y=nn_em_curves_2[8][:, 1], x=range(200), label='B+EM10 21', ax=ax)
    sns.lineplot(y=nn_em_curves_2[18][:, 1], x=range(200), label='B+EM20 31', ax=ax)
    sns.lineplot(y=nn_em_curves_2[28][:, 1], x=range(200), label='B+EM30 41', ax=ax)
    ax.legend(loc='lower center', fontsize=8)
    plt.savefig('Article p4b KMEM train2')

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.25)
    plt.subplots_adjust(left=.16)
    plt.subplots_adjust(right=.98)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Iterations')
    plt.title('Cluster NN Testing Curve')
    sns.lineplot(y=nn_base_res[:, 2], x=range(200), label='Base 11', ax=ax)
    sns.lineplot(y=nn_km_curves_2[8][:, 2], x=range(200), label='B+KM10 21', ax=ax)
    sns.lineplot(y=nn_km_curves_2[18][:, 2], x=range(200), label='B+KM20 31', ax=ax)
    sns.lineplot(y=nn_km_curves_2[28][:, 2], x=range(200), label='B+KM30 41', ax=ax)
    sns.lineplot(y=nn_em_curves_2[8][:, 2], x=range(200), label='B+EM10 21', ax=ax)
    sns.lineplot(y=nn_em_curves_2[18][:, 2], x=range(200), label='B+EM20 31', ax=ax)
    sns.lineplot(y=nn_em_curves_2[28][:, 2], x=range(200), label='B+EM30 41', ax=ax)
    ax.legend(loc='lower center', fontsize=8)
    plt.savefig('Article p4b KMEM test2')


if __name__ == '__main__':
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
