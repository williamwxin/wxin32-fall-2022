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
plt.title('Article ICA K-Means: Silhouette Score')
plt.xlabel('K')
plt.ylabel('Metric Value')
sns.lineplot(y=km_res_dr[:, 0], x=k_range, label='Euclidean')
sns.lineplot(y=km_res_dr[:, 1], x=k_range, label='Manhattan')
sns.lineplot(y=km_res_dr[:, 2], x=k_range, label='Cosine')
plt.savefig('Article p2 KM ICA silhouette')

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Article ICA EM: Silhouette Score')
plt.xlabel('# Clusters')
plt.ylabel('Metric Value')
sns.lineplot(y=em_res_dr[:, 0], x=k_range, label='Euclidean')
sns.lineplot(y=em_res_dr[:, 1], x=k_range, label='Manhattan')
sns.lineplot(y=em_res_dr[:, 2], x=k_range, label='Cosine')
plt.savefig('Article p2 EM ICA silhouette')

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
plt.title('Article ICA K-Means: Majority Vote Acc')
plt.xlabel('K')
plt.ylabel('Accuracy')
sns.lineplot(y=km_res_dr[:, 0], x=k_range, label='Train')
sns.lineplot(y=km_res_dr[:, 1], x=k_range, label='Test')
plt.savefig('Article p2 KM ICA vote')

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Article ICA EM: Majority Vote Acc')
plt.xlabel('# Clusters')
plt.ylabel('Accuracy')
sns.lineplot(y=em_res_dr[:, 0], x=k_range, label='Train')
sns.lineplot(y=em_res_dr[:, 1], x=k_range, label='Test')
plt.savefig('Article p2 EM ICA vote')


model_pca = DC.PCA(n_components=20)
model_ica = DC.FastICA(n_components=30, random_state = 0)
model_rp = RP.GaussianRandomProjection(n_components=20)
model_tsvd = DC.TruncatedSVD(n_components=20)
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

        kurt_order = np.argsort(ica_kurt)
        ica_x_tr = ica_x_tr[:, kurt_order[20:]]
        ica_x_te = ica_x_te[:, kurt_order[20:]]
        train_xs.append(ica_x_tr)
        test_xs.append(ica_x_te)
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
    k_range = list(range(2, 21))+list(range(21, 61,3))
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
    plt.title('Article '+model_names[i]+' K-Means: Silhouette Score')
    plt.xlabel('K')
    plt.ylabel('Metric Value')
    sns.lineplot(y=km_res_dr[:, 0], x=k_range, label='Euclidean')
    sns.lineplot(y=km_res_dr[:, 1], x=k_range, label='Manhattan')
    sns.lineplot(y=km_res_dr[:, 2], x=k_range, label='Cosine')
    plt.savefig('Article p2 KM '+model_names[i]+' silhouette')

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Article '+model_names[i]+' EM: Silhouette Score')
    plt.xlabel('# Clusters')
    plt.ylabel('Metric Value')
    sns.lineplot(y=em_res_dr[:, 0], x=k_range, label='Euclidean')
    sns.lineplot(y=em_res_dr[:, 1], x=k_range, label='Manhattan')
    sns.lineplot(y=em_res_dr[:, 2], x=k_range, label='Cosine')
    plt.savefig('Article p2 EM '+model_names[i]+' silhouette')

    k_range = list(range(2, 101,4))
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
    plt.title('Article ICA K-Means: Majority Vote Acc')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    sns.lineplot(y=km_res_dr[:, 0], x=k_range, label='Train')
    sns.lineplot(y=km_res_dr[:, 1], x=k_range, label='Test')
    plt.savefig('Article p2 KM '+model_names[i]+' vote')

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Article ICA EM: Majority Vote Acc')
    plt.xlabel('# Clusters')
    plt.ylabel('Accuracy')
    sns.lineplot(y=em_res_dr[:, 0], x=k_range, label='Train')
    sns.lineplot(y=em_res_dr[:, 1], x=k_range, label='Test')
    plt.savefig('Article p2 EM '+model_names[i]+' vote')

sns.set()
fig, ax = plt.subplots(figsize=(4.5, 2.25))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Article DimRed EM: Euclidean Silhouette')
plt.xlabel('# Clusters')
plt.ylabel('Metric Value')
k_range= list(range(2, 21))+list(range(21, 61,3))
sns.lineplot(y=silh_km[0][:, 0], x=k_range, label='KM PCA20')
sns.lineplot(y=silh_km[1][:, 0], x=k_range, label='KM ICA30')
sns.lineplot(y=silh_km[2][:, 0], x=k_range, label='KM RP20')
sns.lineplot(y=silh_km[3][:, 0], x=k_range, label='KM TSVD20')
sns.lineplot(y=silh_em[0][:, 0], x=k_range, label='EM PCA20')
sns.lineplot(y=silh_em[1][:, 0], x=k_range, label='EM ICA30')
sns.lineplot(y=silh_em[2][:, 0], x=k_range, label='EM RP20')
sns.lineplot(y=silh_em[3][:, 0], x=k_range, label='EM TSVD20')
ax.legend(fontsize=8, loc='center')
plt.savefig('Article p2 allDIMRED euc silhouette')

sns.set()
fig, ax = plt.subplots(figsize=(4.5, 2.25))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Article DimRed EM: Training Accuracy')
plt.xlabel('# Clusters')
plt.ylabel('Metric Value')
k_range = range(2,101,4)
sns.lineplot(y=vote_km[0][:, 0], x=k_range, label='KM PCA20')
sns.lineplot(y=vote_km[1][:, 0], x=k_range, label='KM ICA30')
sns.lineplot(y=vote_km[2][:, 0], x=k_range, label='KM RP20')
sns.lineplot(y=vote_km[3][:, 0], x=k_range, label='KM TSVD20')
sns.lineplot(y=vote_em[0][:, 0], x=k_range, label='EM PCA20')
sns.lineplot(y=vote_em[1][:, 0], x=k_range, label='EM ICA30')
sns.lineplot(y=vote_em[2][:, 0], x=k_range, label='EM RP20')
sns.lineplot(y=vote_em[3][:, 0], x=k_range, label='EM TSVD20')
ax.legend(fontsize=8, loc='center')
plt.savefig('Article p2 allDIMRED training')

sns.set()
fig, ax = plt.subplots(figsize=(4.5, 2.25))
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('Article DimRed EM: Testing Accuracy')
plt.xlabel('# Clusters')
plt.ylabel('Metric Value')
k_range = range(2,101,4)
sns.lineplot(y=vote_km[0][:, 1], x=k_range, label='KM PCA20')
sns.lineplot(y=vote_km[1][:, 1], x=k_range, label='KM ICA30')
sns.lineplot(y=vote_km[2][:, 1], x=k_range, label='KM RP20')
sns.lineplot(y=vote_km[3][:, 1], x=k_range, label='KM TSVD20')
sns.lineplot(y=vote_em[0][:, 1], x=k_range, label='EM PCA20')
sns.lineplot(y=vote_em[1][:, 1], x=k_range, label='EM ICA30')
sns.lineplot(y=vote_em[2][:, 1], x=k_range, label='EM RP20')
sns.lineplot(y=vote_em[3][:, 1], x=k_range, label='EM TSVD20')
ax.legend(fontsize=8, loc='center')
plt.savefig('Article p2 allDIMRED testing')


silh_km = []
silh_em = []
for i, m in enumerate(models):
    k_range = list(range(2, 21)) + list(range(21, 61, 3))
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
    plt.title('Article ' + model_names[i] + ' K-Means: Silhouette Score')
    plt.xlabel('K')
    plt.ylabel('Metric Value')
    sns.lineplot(y=km_res_dr[:, 0], x=k_range, label='Euclidean')
    sns.lineplot(y=km_res_dr[:, 1], x=k_range, label='Manhattan')
    sns.lineplot(y=km_res_dr[:, 2], x=k_range, label='Cosine')
    plt.savefig('Article p2 KM ' + model_names[i] + ' silhouette')

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Article ' + model_names[i] + ' EM: Silhouette Score')
    plt.xlabel('# Clusters')
    plt.ylabel('Metric Value')
    sns.lineplot(y=em_res_dr[:, 0], x=k_range, label='Euclidean')
    sns.lineplot(y=em_res_dr[:, 1], x=k_range, label='Manhattan')
    sns.lineplot(y=em_res_dr[:, 2], x=k_range, label='Cosine')
    plt.savefig('Article p2 EM ' + model_names[i] + ' silhouette')



