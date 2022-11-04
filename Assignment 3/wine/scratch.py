k_range = list(range(2,51))
km_res_dr = np.zeros((len(k_range),6))
em_res_dr = np.zeros((len(k_range),6))
train_x_dr = ica_x_tr
test_x_dr = ica_x_te
avg = 10
for i, k in enumerate(k_range):
    for t in range(avg):
        model = KMeans(n_clusters=k, random_state=t)
        model.fit(train_x_dr)
        pred_c = model.predict(train_x_dr)
        test_2 = model.cluster_centers_
        km_res_dr[i,0] += metrics.silhouette_score(train_x_dr, pred_c, metric='euclidean')/5
        km_res_dr[i, 1] += metrics.silhouette_score(train_x_dr, pred_c, metric='manhattan')/5
        km_res_dr[i, 2] += metrics.silhouette_score(train_x_dr, pred_c, metric='cosine')/5
        km_res_dr[i,3] += (sum(np.min(cdist(train_x_dr, model.cluster_centers_,'euclidean'), axis=1)) / train_x_dr.shape[0])/5
        km_res_dr[i, 4] += (sum(np.min(cdist(train_x_dr, model.cluster_centers_, 'cityblock'), axis=1)) / train_x_dr.shape[0])/5
        km_res_dr[i, 5] += (sum(np.min(cdist(train_x_dr, model.cluster_centers_, 'cosine'), axis=1)) / train_x_dr.shape[0])/5
        # km_res_dr[i,1] = metrics.calinski_harabasz_score(train_x_dr, pred_c)
        # km_res_dr[i,2] = metrics.davies_bouldin_score(train_x_dr, pred_c)

        model = GaussianMixture(n_components=k, random_state=t)
        model.fit(train_x_dr)
        pred_c = model.predict(train_x_dr)
        test_1 = model.means_
        em_res_dr[i, 0] += metrics.silhouette_score(train_x_dr, pred_c, metric='euclidean')/5
        em_res_dr[i, 1] += metrics.silhouette_score(train_x_dr, pred_c, metric='manhattan')/5
        em_res_dr[i, 2] += metrics.silhouette_score(train_x_dr, pred_c, metric='cosine')/5
        em_res_dr[i, 3] += (sum(np.min(cdist(train_x_dr, model.means_,'euclidean'), axis=1)) / train_x_dr.shape[0])/5
        em_res_dr[i, 4] += (sum(np.min(cdist(train_x_dr, model.means_, 'cityblock'), axis=1)) / train_x_dr.shape[0])/5
        em_res_dr[i, 5] += (sum(np.min(cdist(train_x_dr, model.means_, 'cosine'), axis=1)) / train_x_dr.shape[0])/5
    # em_res_dr[i, 1] = metrics.calinski_harabasz_score(train_x_dr, pred_c)
    # em_res_dr[i, 2] = metrics.davies_bouldin_score(train_x_dr, pred_c)

    print(k)

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Wine ICA K-Means: Silhouette Score')
plt.xlabel('K')
plt.ylabel('Metric Value')
sns.lineplot(y=km_res_dr[:, 0], x=k_range, label='Euclidean')
sns.lineplot(y=km_res_dr[:, 1], x=k_range, label='Manhattan')
sns.lineplot(y=km_res_dr[:, 2], x=k_range, label='Cosine')
plt.savefig('wine p2 KM ICA silhouette')

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Wine ICA EM: Silhouette Score')
plt.xlabel('# Clusters')
plt.ylabel('Metric Value')
sns.lineplot(y=em_res_dr[:, 0], x=k_range, label='Euclidean')
sns.lineplot(y=em_res_dr[:, 1], x=k_range, label='Manhattan')
sns.lineplot(y=em_res_dr[:, 2], x=k_range, label='Cosine')
plt.savefig('wine p2 EM ICA silhouette')

k_range = list(range(2,101))
km_res_dr = np.zeros((len(k_range),2))
em_res_dr = np.zeros((len(k_range),2))

for i, k in enumerate(k_range):
    for t in range(avg):
        model = KMeans(n_clusters=k, random_state=t)
        model.fit(train_x_dr)
        pred_c = model.predict(train_x_dr)
        pred_y = np.zeros(len(pred_c))
        votes = np.zeros(k)
        for c in range(k):
            temp = train_y[pred_c==c]
            val = (np.mean(temp)>=0.5) * 1
            pred_y[pred_c==c] = val
            votes[c] = val
        km_res_dr[i,0] += accuracy_score(train_y, pred_y)/avg

        pred_c = model.predict(test_x_dr)
        pred_y = np.zeros(len(pred_c))
        for c in range(k):
            pred_y[pred_c == c] = votes[c]
        km_res_dr[i, 1] += accuracy_score(test_y, pred_y)/avg

        model = GaussianMixture(n_components=k, random_state=t)
        model.fit(train_x_dr)
        pred_c = model.predict(train_x_dr)
        pred_y = np.zeros(len(pred_c))
        votes = np.zeros(k)
        for c in range(k):
            temp = train_y[pred_c == c]
            val = (np.mean(temp) >= 0.5) * 1
            pred_y[pred_c == c] = val
            votes[c] = val
        em_res_dr[i, 0] += accuracy_score(train_y, pred_y)/avg

        pred_c = model.predict(test_x_dr)
        pred_y = np.zeros(len(pred_c))
        for c in range(k):
            pred_y[pred_c == c] = votes[c]
        em_res_dr[i, 1] += accuracy_score(test_y, pred_y)/avg

    print(k)

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Wine ICA K-Means: Majority Vote Acc')
plt.xlabel('K')
plt.ylabel('Accuracy')
sns.lineplot(y=km_res_dr[:, 0], x=k_range, label='Train')
sns.lineplot(y=km_res_dr[:, 1], x=k_range, label='Test')
plt.savefig('wine p2 KM ICA vote')

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Wine ICA EM: Majority Vote Acc')
plt.xlabel('# Clusters')
plt.ylabel('Accuracy')
sns.lineplot(y=em_res_dr[:, 0], x=k_range, label='Train')
sns.lineplot(y=em_res_dr[:, 1], x=k_range, label='Test')
plt.savefig('wine p2 EM ICA vote')


model_pca = DC.PCA(n_components=6)
model_ica = DC.FastICA(n_components=11, random_state = 0)
model_rp = RP.GaussianRandomProjection(n_components=6)
model_tsvd = DC.TruncatedSVD(n_components=6)
models = [model_pca, model_ica, model_rp, model_tsvd]
model_names = ['PCA', 'ICA', 'RP', 'TSVD']
train_xs = []
test_xs = []
for i,m in enumerate(models):
    if i == 1:
        m.fit(train_x)
        ica_x_tr = m.transform(train_x)
        txt_mean = ica_x_tr.mean(axis=0)
        txt_std = ica_x_tr.std(axis=0)
        ica_x_tr = (ica_x_tr - txt_mean) / txt_std
        ica_x_te = m.transform(test_x)
        ica_x_te = (ica_x_te - txt_mean) / txt_std
        ica_kurt = scipy.stats.kurtosis(ica_x_tr, axis=0)

        ica_x_tr = np.vstack((ica_x_tr[:, 0], ica_x_tr[:, 2], ica_x_tr[:, 3], ica_x_tr[:, 5], ica_x_tr[:, 6], ica_x_tr[:, 8])).T
        ica_x_te = np.vstack((ica_x_te[:, 0], ica_x_te[:, 2], ica_x_te[:, 3], ica_x_te[:, 5], ica_x_te[:, 6], ica_x_te[:, 8])).T
        train_xs.append(ica_x_tr)
        test_xs.append(ica_x_te)
    elif i == 3:
        m.fit(tsvd_train_x)
        tr_x = m.transform(tsvd_train_x)
        te_x = m.transform(tsvd_test_x)
        txt_mean = tr_x.mean(axis=0)
        txt_std = tr_x.std(axis=0)
        tr_x = (tr_x - txt_mean) / txt_std
        te_x = (te_x - txt_mean) / txt_std
        train_xs.append(tr_x)
        test_xs.append(te_x)
    else:
        m.fit(train_x)
        tr_x = m.transform(train_x)
        te_x = m.transform(test_x)
        txt_mean = tr_x.mean(axis=0)
        txt_std = tr_x.std(axis=0)
        tr_x = (tr_x - txt_mean) / txt_std
        te_x = (te_x - txt_mean) / txt_std
        train_xs.append(tr_x)
        test_xs.append(te_x)

silh_km = []
silh_em = []
vote_km = []
vote_em = []

for i,m in enumerate(models):
    k_range = list(range(2, 51))
    km_res_dr = np.zeros((len(k_range), 6))
    em_res_dr = np.zeros((len(k_range), 6))
    train_x_dr = train_xs[i]
    test_x_dr = test_xs[i]
    avg = 10
    for p, k in enumerate(k_range):
        for t in range(avg):
            model = KMeans(n_clusters=k, random_state=t)
            model.fit(train_x_dr)
            pred_c = model.predict(train_x_dr)
            test_2 = model.cluster_centers_
            km_res_dr[p, 0] += metrics.silhouette_score(train_x_dr, pred_c, metric='euclidean') / 5
            km_res_dr[p, 1] += metrics.silhouette_score(train_x_dr, pred_c, metric='manhattan') / 5
            km_res_dr[p, 2] += metrics.silhouette_score(train_x_dr, pred_c, metric='cosine') / 5
            km_res_dr[p, 3] += (sum(np.min(cdist(train_x_dr, model.cluster_centers_, 'euclidean'), axis=1)) / train_x_dr.shape[0]) / 5
            km_res_dr[p, 4] += (sum(np.min(cdist(train_x_dr, model.cluster_centers_, 'cityblock'), axis=1)) / train_x_dr.shape[0]) / 5
            km_res_dr[p, 5] += (sum(np.min(cdist(train_x_dr, model.cluster_centers_, 'cosine'), axis=1)) / train_x_dr.shape[0]) / 5
          
            model = GaussianMixture(n_components=k, random_state=t)
            model.fit(train_x_dr)
            pred_c = model.predict(train_x_dr)
            test_1 = model.means_
            em_res_dr[p, 0] += metrics.silhouette_score(train_x_dr, pred_c, metric='euclidean') / 5
            em_res_dr[p, 1] += metrics.silhouette_score(train_x_dr, pred_c, metric='manhattan') / 5
            em_res_dr[p, 2] += metrics.silhouette_score(train_x_dr, pred_c, metric='cosine') / 5
            em_res_dr[p, 3] += (sum(np.min(cdist(train_x_dr, model.means_, 'euclidean'), axis=1)) / train_x_dr.shape[0]) / 5
            em_res_dr[p, 4] += (sum(np.min(cdist(train_x_dr, model.means_, 'cityblock'), axis=1)) / train_x_dr.shape[0]) / 5
            em_res_dr[p, 5] += (sum(np.min(cdist(train_x_dr, model.means_, 'cosine'), axis=1)) / train_x_dr.shape[0]) / 5
        print(k)

    silh_km.append(km_res_dr)
    silh_em.append(em_res_dr)

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Wine '+model_names[i]+' K-Means: Silhouette Score')
    plt.xlabel('K')
    plt.ylabel('Metric Value')
    sns.lineplot(y=km_res_dr[:, 0], x=k_range, label='Euclidean')
    sns.lineplot(y=km_res_dr[:, 1], x=k_range, label='Manhattan')
    sns.lineplot(y=km_res_dr[:, 2], x=k_range, label='Cosine')
    plt.savefig('wine p2 KM '+model_names[i]+' silhouette')

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Wine '+model_names[i]+' EM: Silhouette Score')
    plt.xlabel('# Clusters')
    plt.ylabel('Metric Value')
    sns.lineplot(y=em_res_dr[:, 0], x=k_range, label='Euclidean')
    sns.lineplot(y=em_res_dr[:, 1], x=k_range, label='Manhattan')
    sns.lineplot(y=em_res_dr[:, 2], x=k_range, label='Cosine')
    plt.savefig('wine p2 EM '+model_names[i]+' silhouette')

    k_range = list(range(2, 101))
    km_res_dr = np.zeros((len(k_range), 2))
    em_res_dr = np.zeros((len(k_range), 2))

    for p, k in enumerate(k_range):
        for t in range(avg):
            model = KMeans(n_clusters=k, random_state=t)
            model.fit(train_x_dr)
            pred_c = model.predict(train_x_dr)
            pred_y = np.zeros(len(pred_c))
            votes = np.zeros(k)
            for c in range(k):
                temp = train_y[pred_c == c]
                val = (np.mean(temp) >= 0.5) * 1
                pred_y[pred_c == c] = val
                votes[c] = val
            km_res_dr[p, 0] += accuracy_score(train_y, pred_y) / avg

            pred_c = model.predict(test_x_dr)
            pred_y = np.zeros(len(pred_c))
            for c in range(k):
                pred_y[pred_c == c] = votes[c]
            km_res_dr[p, 1] += accuracy_score(test_y, pred_y) / avg

            model = GaussianMixture(n_components=k, random_state=t)
            model.fit(train_x_dr)
            pred_c = model.predict(train_x_dr)
            pred_y = np.zeros(len(pred_c))
            votes = np.zeros(k)
            for c in range(k):
                temp = train_y[pred_c == c]
                val = (np.mean(temp) >= 0.5) * 1
                pred_y[pred_c == c] = val
                votes[c] = val
            em_res_dr[p, 0] += accuracy_score(train_y, pred_y) / avg

            pred_c = model.predict(test_x_dr)
            pred_y = np.zeros(len(pred_c))
            for c in range(k):
                pred_y[pred_c == c] = votes[c]
            em_res_dr[p, 1] += accuracy_score(test_y, pred_y) / avg

        print(k)

    vote_km.append(km_res_dr)
    vote_em.append(em_res_dr)

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Wine ICA K-Means: Majority Vote Acc')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    sns.lineplot(y=km_res_dr[:, 0], x=k_range, label='Train')
    sns.lineplot(y=km_res_dr[:, 1], x=k_range, label='Test')
    plt.savefig('wine p2 KM '+model_names[i]+' vote')

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Wine ICA EM: Majority Vote Acc')
    plt.xlabel('# Clusters')
    plt.ylabel('Accuracy')
    sns.lineplot(y=em_res_dr[:, 0], x=k_range, label='Train')
    sns.lineplot(y=em_res_dr[:, 1], x=k_range, label='Test')
    plt.savefig('wine p2 EM '+model_names[i]+' vote')

sns.set()
fig, ax = plt.subplots(figsize=(4.5, 2.25))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Wine DimRed EM: Euclidean Silhouette')
plt.xlabel('# Clusters')
plt.ylabel('Metric Value')
k_range = range(2,51)
sns.lineplot(y=silh_km[0][:, 0], x=k_range, label='KM PCA6')
sns.lineplot(y=silh_km[1][:, 0], x=k_range, label='KM ICA6')
sns.lineplot(y=silh_km[2][:, 0], x=k_range, label='KM RP6')
sns.lineplot(y=silh_km[3][:, 0], x=k_range, label='KM TSVD6')
sns.lineplot(y=silh_em[0][:, 0], x=k_range, label='EM PCA6')
sns.lineplot(y=silh_em[1][:, 0], x=k_range, label='EM ICA6')
sns.lineplot(y=silh_em[2][:, 0], x=k_range, label='EM RP6')
sns.lineplot(y=silh_em[3][:, 0], x=k_range, label='EM TSVD6')
ax.legend(fontsize=8, loc='center')
plt.savefig('wine p2 allDIMRED euc silhouette')

sns.set()
fig, ax = plt.subplots(figsize=(4.5, 2.25))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Wine DimRed EM: Training Accuracy')
plt.xlabel('# Clusters')
plt.ylabel('Metric Value')
k_range = range(2,101)
sns.lineplot(y=vote_km[0][:, 0], x=k_range, label='KM PCA6')
sns.lineplot(y=vote_km[1][:, 0], x=k_range, label='KM ICA6')
sns.lineplot(y=vote_km[2][:, 0], x=k_range, label='KM RP6')
sns.lineplot(y=vote_km[3][:, 0], x=k_range, label='KM TSVD6')
sns.lineplot(y=vote_em[0][:, 0], x=k_range, label='EM PCA6')
sns.lineplot(y=vote_em[1][:, 0], x=k_range, label='EM ICA6')
sns.lineplot(y=vote_em[2][:, 0], x=k_range, label='EM RP6')
sns.lineplot(y=vote_em[3][:, 0], x=k_range, label='EM TSVD6')
ax.legend(fontsize=8, loc='center')
plt.savefig('wine p2 allDIMRED training')

vote_km[0][:, 0]
vote_km[3][:, 0]

sns.set()
fig, ax = plt.subplots(figsize=(4.5, 2.25))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Wine DimRed EM: Testing Accuracy')
plt.xlabel('# Clusters')
plt.ylabel('Metric Value')
k_range = range(2,101)
sns.lineplot(y=vote_km[0][:, 1], x=k_range, label='KM PCA6')
sns.lineplot(y=vote_km[1][:, 1], x=k_range, label='KM ICA6')
sns.lineplot(y=vote_km[2][:, 1], x=k_range, label='KM RP6')
sns.lineplot(y=vote_km[3][:, 1], x=k_range, label='KM TSVD6')
sns.lineplot(y=vote_em[0][:, 1], x=k_range, label='EM PCA6')
sns.lineplot(y=vote_em[1][:, 1], x=k_range, label='EM ICA6')
sns.lineplot(y=vote_em[2][:, 1], x=k_range, label='EM RP6')
sns.lineplot(y=vote_em[3][:, 1], x=k_range, label='EM TSVD6')
ax.legend(fontsize=8, loc='center')
plt.savefig('wine p2 allDIMRED testing')




dimred_time = np.zeros((len(list(range(1,11))),4))
scale = StandardScaler()
scale.fit(train_x)

pca_range = list(range(1,11))
ica_range = list(range(1,11))
rp_range = list(range(1,11))
tsvd_range=list(range(1,11))

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
plt.title('Wine: PCA')
sns.lineplot(y=pca_res[:, 1], x=pca_range, label='Expl Var', ax=ax, color='orange')
sns.lineplot(y=pca_res[:, 1], x=pca_range, label='Rec Err', ax=ax)
ax2 = ax.twinx()
ax2.grid(False)
ax2.set_ylabel('Explained Variance')
ax2.set_ylim([0,1.1])
sns.lineplot(y=pca_res[:, 0], x=pca_range, ax=ax2, color='orange')
ax.legend(loc='center right')
plt.savefig('Wine pca')



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
plt.title('Wine: RP')
sns.lineplot(y=rp_res, x=rp_range, label='Rec Err', ax=ax)
ax.legend(loc='center right')
plt.savefig('Wine rp')



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
plt.title('Wine: Reconstruction Error')
sns.lineplot(y=pca_res[:,1], x=tsvd_range, label='PCA=ICA', ax=ax)
sns.lineplot(y=rp_res, x=tsvd_range, label='RP', ax=ax)
sns.lineplot(y=tsvd_res, x=tsvd_range, label='TSVD', ax=ax)
ax.legend(loc='center')
plt.savefig('Wine p2 rec error')


sns.set()
fig, ax = plt.subplots(figsize=(5, 2.5))
plt.subplots_adjust(bottom=.3)
plt.subplots_adjust(left=.16)
plt.subplots_adjust(right=.86)
#ax.lege
ax.set_ylabel('Clock (ms)')
ax.set_xlabel('# Components')
plt.title('Wine: Dim Reduction Clock Times')
sns.lineplot(y=dimred_time[:,0]*1000, x=tsvd_range, label='PCA', ax=ax)
sns.lineplot(y=dimred_time[:,1]*1000, x=tsvd_range, label='ICA', ax=ax, color='purple')
sns.lineplot(y=dimred_time[:,2]*1000, x=tsvd_range, label='RP', ax=ax)
sns.lineplot(y=dimred_time[:,3]*1000, x=tsvd_range, label='TSVD', ax=ax)
ax.legend(loc='center')
ax2.legend(loc='upper center')
plt.savefig('Wine p2 clock')