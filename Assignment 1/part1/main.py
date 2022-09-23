# pruning: https://medium.com/analytics-vidhya/cost-complexity-pruning-in-decision-trees-f82b14a7fe91#:~:text=DecisionTree%20in%20sklearn%20has%20a%20function%20called%20cost_complexity_pruning_path%2C,prune%20our%20decision%20tree-%20Cost%20Complexity%20Pruning%20Path


import sklearn.tree
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import loadtxt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import time
import copy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import os

if __name__ == '__main__':
    os.getcwd()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = loadtxt(open('OnlineNewsPopularity.csv'), delimiter=",", dtype=object)
    # data = loadtxt(open('housing.csv'),delimiter = ",", dtype=object)
    data = data[:, 1:]
    cols = data[0, :]
    data = data[1:, :].astype(float)
    data_p1 = data[data[:,-1]>=5000]
    data_p2 = data[data[:,-1]<5000]
    data_p1[:,-1] = 1
    data_p2[:,-1] = 0

    p2_idx = np.random.choice(len(data_p2),len(data_p1),replace=False)
    data_p2 = data_p2[p2_idx]

    data_b = np.vstack((data_p1, data_p2))

    data_x = data_b[:, :-1]
    data_x = np.array(data_x, dtype=float)
    data_y = data_b[:, -1]

    data_y_class = data_y.copy()
    # data_y_class[np.array(data_y, dtype=float)>=23]=1
    data_y_class = np.array(data_y_class, dtype=int)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y_class, test_size=0.20, random_state=0)

    cv = 5
    kf = KFold(n_splits=cv, shuffle=True, random_state=100)


    def tt_errors(model, train_x, test_x, train_y, test_y):
        pred_train = model.predict(train_x)
        pred_test = model.predict(test_x)
        train_acc = sum(pred_train == train_y) / len(pred_train)
        test_acc = sum(pred_test == test_y) / len(pred_test)
        return train_acc, test_acc


    tree_depth = np.zeros((10, 3))
    tree_depth[:, 0] = (range(11))[1:]

    for i in range(len(tree_depth)):
        for train_i, test_i in kf.split(train_x):
            tree_x = DecisionTreeClassifier(random_state=0, max_depth=int(tree_depth[i, 0]), min_samples_leaf=192).fit(train_x[train_i],
                                                                                                 train_y[train_i])

            train_acc, test_acc = tt_errors(tree_x, train_x[train_i], train_x[test_i], train_y[train_i], train_y[test_i])
            tree_depth[i, 1] += train_acc
            tree_depth[i, 2] += test_acc
        tree_depth[i, 1] /= cv
        tree_depth[i, 2] /= cv

    # leaf_size = np.zeros((100,3))
    # leaf_size[:,0] = (range(101))[1:]
    leaf_size = np.zeros((24, 3))
    leaf_size[:, 0] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1024 + 512, 2048,
                       2048 + 1024, 4096]
    for i in range(len(leaf_size)):
        for train_i, test_i in kf.split(train_x):
            tree_x = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_leaf=int(leaf_size[i, 0])).fit(train_x[train_i],
                                                                                                       train_y[train_i])

            train_acc, test_acc = tt_errors(tree_x, train_x[train_i], train_x[test_i], train_y[train_i], train_y[test_i])
            leaf_size[i, 1] += train_acc
            leaf_size[i, 2] += test_acc
        leaf_size[i, 1] /= cv
        leaf_size[i, 2] /= cv

    # min_split = np.zeros((26, 3))
    # min_split[:, 0] = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1024 + 512, 2048,
    #                    2048 + 1024, 4096, 4096 + 2048, 8192, 8921 + 4096]
    # for i in range(len(leaf_size)):
    #     for train_i, test_i in kf.split(train_x):
    #         tree_x = DecisionTreeClassifier(random_state=0, min_samples_split=int(min_split[i, 0])).fit(train_x[train_i],
    #                                                                                                     train_y[train_i])
    #
    #         train_acc, test_acc = tt_errors(tree_x, train_x[train_i], train_x[test_i], train_y[train_i], train_y[test_i])
    #         min_split[i, 1] += train_acc
    #         min_split[i, 2] += test_acc
    #     min_split[i, 1] /= 4
    #     min_split[i, 2] /= 4

    # tree_0a = DecisionTreeClassifier(random_state=0, max_depth=8).fit(train_x, train_y)
    # tree_0b = DecisionTreeClassifier(random_state=0, min_samples_leaf=256).fit(train_x, train_y)
    # tree_0c = DecisionTreeClassifier(random_state=0, min_samples_split=768).fit(train_x, train_y)
    #
    # test_acc_0a = tt_errors(tree_0a, train_x, test_x, train_y, test_y)[1]
    # test_acc_0b = tt_errors(tree_0b, train_x, test_x, train_y, test_y)[1]
    # test_acc_0c = tt_errors(tree_0c, train_x, test_x, train_y, test_y)[1]

    # training data fits well but testing doesn't. Suggests that model isn't generalizing well

    parameters = {'max_depth': range(1, 21),
                  'min_samples_leaf': [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024,
                                       1024 + 512, 2048, 2048 + 1024, 4096]}

    # parameters = {'max_depth': range(1, 21),
    #               'min_samples_leaf': [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64],}

    clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=8)
    clf.fit(X=train_x, y=train_y)
    best_tree = clf.best_estimator_
    best_train_acc, best_test_acc = tt_errors(best_tree, train_x, test_x, train_y, test_y)

    best_tree = DecisionTreeClassifier(random_state=0, max_depth=int(tree_depth[i, 0]), min_samples_leaf=192)

    path = best_tree.cost_complexity_pruning_path(train_x, train_y)
    alphas = path['ccp_alphas']

    alphas_res = np.zeros((len(alphas)-1, 3))
    alphas_res[:, 0] = alphas[:-1]
    for i, a in enumerate(alphas_res[:, 0]):
        for train_i, test_i in kf.split(train_x):
            tree_x = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_leaf=192,
                                            ccp_alpha=a).fit(train_x[train_i],
                                                             train_y[train_i])

            train_acc, test_acc = tt_errors(tree_x, train_x[train_i], train_x[test_i], train_y[train_i], train_y[test_i])
            alphas_res[i, 1] += train_acc
            alphas_res[i, 2] += test_acc
        alphas_res[i, 1] /= cv
        alphas_res[i, 2] /= cv



    no_prune_tree = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_leaf=192).fit(train_x, train_y)
    prune_tree = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_leaf=192, ccp_alpha=0.0008).fit(train_x, train_y)
    prune_train_acc, prune_test_acc = tt_errors(prune_tree, train_x, test_x, train_y, test_y)





    sns.set()


    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Articles: Accuracy vs alpha')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    sns.lineplot(y=alphas_res[:, 1], x=alphas_res[:,0], label='Train Fold Accuracy')
    sns.lineplot(y=alphas_res[:, 2], x=alphas_res[:,0], label='Val Fold Accuracy')
    plt.savefig('Articles Accuracy vs alpha')


    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Articles: Accuracy vs Tree Depth')
    plt.xlabel('Max Tree Depth')
    plt.ylabel('Accuracy')
    sns.lineplot(y=tree_depth[:, 1], x=tree_depth[:, 0], label='Train Fold Accuracy')
    sns.lineplot(y=tree_depth[:, 2], x=tree_depth[:, 0], label='Val Fold Accuracy')
    plt.savefig('Articles Accuracy vs Tree Depth')

    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Articles: Accuracy vs Leaf Size')
    plt.xlabel('Min Leaf Size')
    plt.ylabel('Accuracy')
    sns.lineplot(y=leaf_size[:, 1], x=leaf_size[:, 0], label='Train Fold Accuracy')
    sns.lineplot(y=leaf_size[:, 2], x=leaf_size[:, 0], label='Val Fold Accuracy')
    plt.savefig('Articles Accuracy vs Leaf Size.png')


    _, ax = plt.subplots(figsize=(5, 2.5), dpi=75)
    plt.suptitle('Articles: Pruned Tree')
    sklearn.tree.plot_tree(prune_tree, impurity=False, fontsize=0, ax=ax)
    plt.savefig('Articles pruned tree')

    _, ax = plt.subplots(figsize=(5, 2.5), dpi=75)
    plt.suptitle('Articles: Un-pruned Tree')
    sklearn.tree.plot_tree(no_prune_tree, impurity=False, fontsize=0)
    plt.savefig('Articles unpruned tree')



    device = "cuda" if torch.cuda.is_available() else "cpu"

    for train_i, test_i in kf.split(train_x):
        train_x_tens = torch.tensor(train_x[train_i], dtype=torch.float32, device=device)
        val_x_tens = torch.tensor(train_x[test_i], dtype=torch.float32, device=device)
        train_y_tens = torch.tensor(train_y[train_i], dtype=torch.long, device=device)
        val_y_tens = torch.tensor(train_y[test_i], dtype=torch.long, device=device)
        break


    test_x_tens = torch.tensor(test_x, dtype=torch.float32, device=device)
    test_y_tens = torch.tensor(test_y, dtype=torch.long, device=device)

    txt_mean = train_x_tens.mean(axis=0)
    txt_std = train_x_tens.std(axis=0)
    train_x_tens = (train_x_tens - txt_mean) / txt_std
    val_x_tens = (val_x_tens - txt_mean) / txt_std
    test_x_tens = (test_x_tens - txt_mean) / txt_std
    samples, features = train_x_tens.shape



    class LayerNet_n(nn.Module):
        def __init__(self, input_dim, hidden_size, layers, num_classes):
            """
            :param input_dim: input feature dimension
            :param hidden_size: hidden dimension
            :param num_classes: total number of classes
            """
            super(LayerNet_n, self).__init__()
            #############################################################################
            # TODO: Initialize the TwoLayerNet, use sigmoid activation between layers   #
            #############################################################################
            self.layers = []
            self.l1 = nn.Linear(input_dim, hidden_size)
            for i in range(layers):
                self.layers.append(nn.Linear(hidden_size, hidden_size))
            for layers in self.layers:
                layers = layers.cuda()
            # self.l1a = nn.Sigmoid(hidden_size, hidden_size)
            self.last = nn.Linear(hidden_size, num_classes)

            #############################################################################
            #                              END OF YOUR CODE                             #
            #############################################################################

        def forward(self, x):
            # out = None
            #############################################################################
            # TODO: Implement forward pass of the network                               #
            #############################################################################
            # x_flat = x.reshape((x.shape[0], np.product(x.shape[1:])))
            # x1 = self.l1(x_flat)
            x = F.leaky_relu(self.l1(torch.flatten(x, 1)))
            for layer in self.layers:
                x = F.leaky_relu(layer(torch.flatten(x, 1)))
            # x1a = nn.Sigmoid()(x1)
            # x1a = F.sigmoid(x1)
            # out = self.l2(x1)
            return self.last(x)

            #############################################################################
            #                              END OF YOUR CODE                             #
            #############################################################################
            # return out

    def accuracy(output, target):
        """Computes the precision@k for the specified values of k"""
        batch_size = target.shape[0]

        _, pred = torch.max(output, dim=-1)

        correct = pred.eq(target).sum() * 1.0

        acc = correct / batch_size

        return acc

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

    print("Device : {}".format(device))

    def train(epoch, data_tens_train, batch_size, model, optimizer, criterion):
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()
        kf = KFold(n_splits=int(len(data_tens_train[0]) / batch_size))
        for idx, (train_i, test_i) in enumerate(kf.split(data_tens_train[0])):
            data = data_tens_train[0][test_i]
            target = data_tens_train[1][test_i]
            start = time.time()

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()

            optimizer.zero_grad()
            out = model.forward(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            batch_acc = accuracy(out, target)

            losses.update(loss, out.shape[0])
            acc.update(batch_acc, out.shape[0])

            iter_time.update(time.time() - start)
            if idx % 10 == 0:
                print(('Epoch: [{0}][{1}/{2}]\t'
                       'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t'
                       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                       'Prec @1 {top1.val:.4f} ({top1.avg:.4f})\t')
                      .format(epoch, idx, int(len(data_tens_train[0]) / batch_size), iter_time=iter_time, loss=losses,
                              top1=acc))
        torch.cuda.empty_cache()

    def validate(epoch, data_tens, batch_size, model, criterion, num_class):
        iter_time = AverageMeter()
        losses = AverageMeter()
        acc = AverageMeter()

        cm = torch.zeros(num_class, num_class)
        splits = int(len(data_tens[0]) / batch_size)
        if splits < 2:
            splits = 2
        kf = KFold(n_splits=splits)
        # evaluation loop
        for idx, (train_i, test_i) in enumerate(kf.split(data_tens[0])):
            data = data_tens[0][test_i]
            target = data_tens[1][test_i]
            # torch.cuda.empty_cache()
            start = time.time()

            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            with torch.no_grad():
                out = model(data)
                loss = criterion(out, target)

            batch_acc = accuracy(out, target)

            # update confusion matrix
            _, preds = torch.max(out, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

            losses.update(loss, out.shape[0])
            acc.update(batch_acc, out.shape[0])

            iter_time.update(time.time() - start)
            if (idx + 1) % 10 == 0:
                print(('Epoch: [{0}][{1}/{2}]\t'
                       'Time {iter_time.val:.3f} ({iter_time.avg:.3f})\t')
                      .format(epoch, idx, int(len(data_tens[0]) / batch_size), iter_time=iter_time, loss=losses, top1=acc))
        cm = cm / cm.sum(1)
        per_cls_acc = cm.diag().detach().numpy().tolist()
        for i, acc_i in enumerate(per_cls_acc):
            print("Accuracy of Class {}: {:.4f}".format(i, acc_i))

        print("* Prec @1: {top1.avg:.4f}".format(top1=acc))
        return acc.avg, cm


    batch_size = 32
    val_batch_size = 512
    learning_rate = 0.01
    momentum = 0.9
    reg = 0.0001
    epochs = 20

    # train_loader = DataLoader(ArticleDatasetTrain(), batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # test_loader = DataLoader(ArticleDatasetTest(), batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False)

    def bestLayerNet(train_x, train_y, test_x, test_y, batch_size=32, val_batch_size=512, learning_rate = 0.01, momentum = 0.9, reg = 0.0001, epochs = 20):

        nn_model = LayerNet_n(features, 256, 2, 2)
        nn_model = nn_model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate, momentum=momentum, weight_decay=reg)

        best = 0.0
        best_cm = None
        best_model = None

        train_x_tens = torch.tensor(train_x, dtype=torch.float32, device=device)
        val_x_tens = torch.tensor(test_x, dtype=torch.float32, device=device)
        train_y_tens = torch.tensor(train_y, dtype=torch.long, device=device)
        val_y_tens = torch.tensor(test_y, dtype=torch.long, device=device)
        txt_mean = train_x_tens.mean(axis=0)
        txt_std = train_x_tens.std(axis=0)
        train_x_tens = (train_x_tens - txt_mean) / txt_std
        val_x_tens = (val_x_tens - txt_mean) / txt_std

        for epoch in range(epochs):
            data_tens_train = (train_x_tens, train_y_tens)
            data_tens_val = (val_x_tens, val_y_tens)
            if epoch == 10:
                optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate / 10,
                                            momentum=momentum,
                                            weight_decay=reg)

            train(epoch, data_tens_train, batch_size, nn_model, optimizer, criterion)

            acc, cm = validate(epoch, data_tens_val, val_batch_size, nn_model, criterion, 2)
            # tr_acc, tr_cm = validate(epoch, data_tens_train, val_batch_size, nn_model, criterion, 2)

            if acc > best:
                best = acc
                best_cm = cm
                best_model = copy.deepcopy(nn_model)
            if (epoch > 4) & (acc < 0.5):
                break
        acc, cm = validate(0, data_tens_val, val_batch_size, best_model, criterion, 2)
        tr_acc, tr_cm = validate(0, data_tens_train, val_batch_size, best_model, criterion, 2)
        return (tr_acc, acc)



    if True:
        nn_model = LayerNet_n(features, 256, 2, 2)
        nn_model = nn_model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate, momentum=momentum, weight_decay=reg)

        best = 0.0
        best_cm = None
        best_model = None

        nn_accs = np.zeros((epochs, 3))
        nn_accs[:, 0] = range(epochs)

        for epoch in range(epochs):
            data_tens_train = (train_x_tens, train_y_tens)
            data_tens_val = (val_x_tens, val_y_tens)
            if epoch == 10:
                optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate / 10,
                                            momentum=momentum,
                                            weight_decay=reg)
            if epoch == 10:
                optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate / 100,
                                            momentum=momentum,
                                            weight_decay=reg)
            train(epoch, data_tens_train, batch_size, nn_model, optimizer, criterion)

            acc, cm = validate(epoch, data_tens_val, val_batch_size, nn_model, criterion, 2)
            tr_acc, tr_cm = validate(epoch, data_tens_train, val_batch_size, nn_model, criterion, 2)

            nn_accs[epoch,1] = tr_acc
            nn_accs[epoch,2] = acc

            if acc > best:
                best = acc
                best_cm = cm
                best_model = copy.deepcopy(nn_model)

        sns.set()
        fig, ax = plt.subplots(figsize=(5, 2.5))
        plt.subplots_adjust(bottom=.26)
        plt.subplots_adjust(left=.16)
        plt.title('Articles: Neural Net Accuracy vs Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        sns.lineplot(y=nn_accs[:, 1], x=nn_accs[:, 0], label='Train Fold Accuracy')
        sns.lineplot(y=nn_accs[:, 2], x=nn_accs[:, 0], label='Val Fold Accuracy')
        plt.savefig('Articles NN Accuracy vs Epoch')


    nn_layers = np.zeros((10, 3))
    nn_layers[:, 0] = range(11)[1:]

    for i in range(len(nn_layers)):
        nn_model = LayerNet_n(features, 256, i + 1, 2)
        nn_model = nn_model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate, momentum=momentum, weight_decay=reg)

        best = 0.0
        best_cm = None
        best_model = None

        if False:
            for epoch in range(epochs):
                for train_i, test_i in KFold(n_splits=4).split(train_x_tens):
                    data_tens_train = (train_x_tens[train_i], train_y_tens[train_i])
                    data_tens_test = (train_x_tens[test_i], train_y_tens[test_i])
                    if epoch == 2:
                        optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate / 10,
                                                    momentum=momentum,
                                                    weight_decay=reg)
                    elif epoch == 4:
                        optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate / 50,
                                                    momentum=momentum,
                                                    weight_decay=reg)
                    elif epoch == 7:
                        optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate / 100,
                                                    momentum=momentum,
                                                    weight_decay=reg)
                    # elif epoch == 40:
                    #     optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate / 100,
                    #                                 momentum=momentum,
                    #                                 weight_decay=reg)
                    # elif epoch == 60:
                    #     optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate / 500,
                    #                                 momentum=momentum,
                    #                                 weight_decay=reg)
                    # adjust_learning_rate(optimizer, epoch, args)

                    train(epoch, data_tens_train, batch_size, nn_model, optimizer, criterion)

                    acc, cm = validate(epoch, data_tens_test, batch_size, nn_model, criterion, 2)

                    if acc > best:
                        best = acc
                        best_cm = cm
                        best_model = copy.deepcopy(nn_model)

        for epoch in range(epochs):
            data_tens_train = (train_x_tens, train_y_tens)
            data_tens_test = (val_x_tens, val_y_tens)
            if epoch == 10:
                optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate / 10,
                                            momentum=momentum,
                                            weight_decay=reg)

            train(epoch, data_tens_train, batch_size, nn_model, optimizer, criterion)

            acc, cm = validate(epoch, data_tens_test, val_batch_size, nn_model, criterion, 2)
            # tr_acc, tr_cm = validate(epoch, data_tens_train, batch_size, nn_model, criterion, 2)
            if acc > best:
                best = acc
                best_cm = cm
                best_model = copy.deepcopy(nn_model)

        acc, cm = validate(0, (train_x_tens, train_y_tens), val_batch_size, nn_model, criterion, 2)
        nn_layers[i, 1] = acc
        acc, cm = validate(0, (val_x_tens, val_y_tens), val_batch_size, nn_model, criterion, 2)
        nn_layers[i, 2] = acc

    nn_layer_size = np.zeros((10, 3))
    nn_layer_size[:, 0] = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    for i in range(len(nn_layer_size)):
        nn_model = LayerNet_n(features, int(nn_layer_size[i, 0]), 2, 2)
        nn_model = nn_model.cuda()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate, momentum=momentum, weight_decay=reg)

        best = 0.0
        best_cm = None
        best_model = None

        for epoch in range(epochs):
            data_tens_train = (train_x_tens, train_y_tens)
            data_tens_test = (val_x_tens, val_y_tens)
            if epoch == 5:
                optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate / 10,
                                            momentum=momentum,
                                            weight_decay=reg)

            train(epoch, data_tens_train, batch_size, nn_model, optimizer, criterion)

            acc, cm = validate(epoch, data_tens_test, val_batch_size, nn_model, criterion, 2)

            if acc > best:
                best = acc
                best_cm = cm
                best_model = copy.deepcopy(nn_model)

        acc, cm = validate(0, (train_x_tens, train_y_tens), val_batch_size, nn_model, criterion, 2)
        nn_layer_size[i, 1] = acc
        acc, cm = validate(0, (val_x_tens, val_y_tens), val_batch_size, nn_model, criterion, 2)
        nn_layer_size[i, 2] = acc

    sns.set()
    fig, ax = plt.subplots(figsize=(5, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Articles: Neural Net Accuracy vs Number of Hidden Layers')
    plt.xlabel('Hidden Layers')
    plt.ylabel('Accuracy')
    sns.lineplot(y=nn_layers[:, 1], x=nn_layers[:, 0], label='Train Fold Accuracy')
    sns.lineplot(y=nn_layers[:, 2], x=nn_layers[:, 0], label='Val Fold Accuracy')
    plt.savefig('Articles NN Accuracy vs Num Layers')

    sns.set()
    fig, ax = plt.subplots(figsize=(5, 2))
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.16)
    plt.title('Articles: Neural Net Accuracy vs Size of Hidden Layer')
    plt.xlabel('Hidden Layer Size')
    plt.ylabel('Accuracy')
    sns.lineplot(y=nn_layer_size[:, 1], x=nn_layer_size[:, 0], label='Train Fold Accuracy')
    sns.lineplot(y=nn_layer_size[:, 2], x=nn_layer_size[:, 0], label='Val Fold Accuracy')
    ax.set_xscale('log', basex=2)
    plt.savefig('Articles NN Accuracy vs Layersize')



    class TreeBoost(AdaBoostClassifier):
        def __init__(self, n_estimators=64, learning_rate=1.0, max_depth=5, ccp_alpha=0, random_state=0):
            super().__init__(base_estimator=DecisionTreeClassifier(max_depth=max_depth, ccp_alpha=ccp_alpha),
                                            n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
            self.max_depth = max_depth
            self.n_estimators = n_estimators
            self.learning_rate = learning_rate
        # def learner(self):
        #     return self.boost

    parameters = {'max_depth': range(1, 10),
                  'n_estimators': [4,8,16,32,64,128,256,512]}

    boost_grid = GridSearchCV(TreeBoost(), parameters, n_jobs=16)
    boost_grid_search = boost_grid.fit(train_x, train_y)
    print(boost_grid_search.best_params_)
    print(boost_grid_search.best_score_)


    boo_depth = np.zeros((5, 3))
    boo_depth[:, 0] = range(1,6)
    cv = 4
    kf = KFold(n_splits=cv, shuffle=True, random_state=100)

    for i in range(len(boo_depth)):
        for train_i, test_i in kf.split(train_x):
            knn_x = TreeBoost(n_estimators=128, max_depth=i+1).fit(train_x[train_i], train_y[train_i])

            train_acc, test_acc = tt_errors(knn_x, train_x[train_i], train_x[test_i], train_y[train_i], train_y[test_i])
            boo_depth[i, 1] += train_acc
            boo_depth[i, 2] += test_acc
        boo_depth[i, 1] /= cv
        boo_depth[i, 2] /= cv


    boo_n = np.zeros((10, 3))
    boo_n[:, 0] = [2,4,8,16,32,64,128,256,512, 1024]
    cv = 4
    kf = KFold(n_splits=cv, shuffle=True, random_state=10)

    for i in range(len(boo_n)):
        for train_i, test_i in kf.split(train_x):
            knn_x = TreeBoost(n_estimators=int(boo_n[i,0]), max_depth=1).fit(train_x[train_i], train_y[train_i])

            train_acc, test_acc = tt_errors(knn_x, train_x[train_i], train_x[test_i], train_y[train_i], train_y[test_i])
            boo_n[i, 1] += train_acc
            boo_n[i, 2] += test_acc
        boo_n[i, 1] /= cv
        boo_n[i, 2] /= cv
        print(i)

    boo_c = np.zeros((10, 3))
    boo_c[:, 0] = [2,4,8,16,32,64,128,256,512, 1024]
    cv = 4
    kf = KFold(n_splits=cv, shuffle=True, random_state=10)

    for i in range(len(boo_n)):
        for train_i, test_i in kf.split(train_x):
            knn_x = TreeBoost(n_estimators=int(boo_n[i,0]), max_depth=1).fit(train_x[train_i], train_y[train_i])

            train_acc, test_acc = tt_errors(knn_x, train_x[train_i], train_x[test_i], train_y[train_i], train_y[test_i])
            boo_n[i, 1] += train_acc
            boo_n[i, 2] += test_acc
        boo_n[i, 1] /= cv
        boo_n[i, 2] /= cv
        print(i)


    boo_frac = np.zeros((13, 3))
    boo_frac[:,0] = dataset_fractions

    for f in range(len(dataset_fractions)):
        avg = 8
        for i in range(avg):
            frac = dataset_fractions[f]
            if frac == 1:
                train_x_frac, train_y_frac = train_x, train_y
            else:
                train_x_frac, _, train_y_frac, _ = train_test_split(train_x, train_y, test_size=1-dataset_fractions[f])
            tree_x = TreeBoost(n_estimators=16, max_depth=1).fit(train_x_frac, train_y_frac)
            train_acc, test_acc = tt_errors(tree_x, train_x_frac, test_x, train_y_frac, test_y)
            train_acc, test_acc = tt_errors(tree_x, train_x_frac, test_x, train_y_frac, test_y)

            boo_frac[f, 1] += train_acc
            boo_frac[f, 2] += test_acc
        boo_frac[f, 1] /= avg
        boo_frac[f, 2] /= avg

    knn_x = TreeBoost(n_estimators=128, max_depth=1).fit(train_x, train_y)
    print(tt_errors(knn_x, train_x, test_x, train_y, test_y))
    knn_x = TreeBoost(n_estimators=128, max_depth=1).fit(train_x_norm_sub, train_y)
    print(tt_errors(knn_x, train_x_norm_sub, test_x_norm_sub, train_y, test_y))


    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.16)
    plt.title('Articles: Boost Accuracy vs Number of Estimators')
    plt.xlabel('Estimators')
    plt.ylabel('Accuracy')
    sns.lineplot(y=boo_n[:, 1], x=boo_n[:, 0], label='Train Fold Accuracy')
    sns.lineplot(y=boo_n[:, 2], x=boo_n[:, 0], label='Val Fold Accuracy')
    ax.set_xscale('log', basex=2)
    # for xy in zip(boo_n[6:7, 0], boo_n[6:7, 2]):
    #     t = (xy[0], round(xy[1],3))
    #     ax.annotate('Val: (2^7 = %s, %s)' % t, xy=t, textcoords='data')
    plt.savefig('Articles Boost Accuracy vs Num Estimators')

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Articles: Boost Accuracy vs Learner Max Depth')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    sns.lineplot(y=boo_depth[:, 1], x=boo_depth[:, 0], label='Train Fold Accuracy')
    sns.lineplot(y=boo_depth[:, 2], x=boo_depth[:, 0], label='Val Fold Accuracy')
    plt.savefig('Articles Boost Accuracy vs depth')


    sns.set()
    fig, ax = plt.subplots(figsize=(5, 3))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Articles: Boosting Accuracy vs Fraction of Training Data Used')
    plt.xlabel('Fraction of Training Data')
    plt.ylabel('Accuracy')
    sns.lineplot(y=boo_frac[:, 1], x=boo_frac[:, 0], label='Training Accuracy')
    sns.lineplot(y=boo_frac[:, 2], x=boo_frac[:, 0], label='Testing Accuracy')
    for xy in zip(boo_frac[:1, 0], boo_frac[:1, 1]):
        t = (xy[0], round(xy[1],3))
        ax.annotate('Train: (%s, %s)' % t, xy=t, textcoords='data')
    for xy in zip(boo_frac[:1, 0], boo_frac[:1, 2]):
        t = (xy[0], round(xy[1],3))
        ax.annotate('Val: (%s, %s)' % t, xy=t, textcoords='data')
    # ax.set_xscale('log', basex=2)
    plt.savefig('Articles Boost Accuracy vs Sample Frac')



    train_x_norm = (train_x - txt_mean) / txt_std
    test_x_norm = (test_x - txt_mean) / txt_std
    # trxn, texn, trxny, texny = train_test_split(train_x_norm, train_y, test_size=0.2, random_state=1)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_x, train_y)
    knn_test_pred = knn.predict(test_x)
    print(accuracy_score(test_y, knn_test_pred))

    if True:
        knn_n = np.zeros((19, 3))
        knn_n[:, 0] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]
        cv = 5
        kf = KFold(n_splits=cv, shuffle=True, random_state=100)
        for train_i, test_i in kf.split(train_x):
            test_i = test_i
            break
        for i in range(len(knn_n)):
            for train_i, test_i in kf.split(train_x):
                knn_x = KNeighborsClassifier(n_neighbors=int(knn_n[i, 0])).fit(train_x_norm[train_i], train_y[train_i])

                train_acc, test_acc = tt_errors(knn_x, train_x_norm[train_i], train_x_norm[test_i], train_y[train_i],
                                                train_y[test_i])
                knn_n[i, 1] += train_acc
                knn_n[i, 2] += test_acc
            knn_n[i, 1] /= cv
            knn_n[i, 2] /= cv

    knn_iterative = np.zeros((features, 4))
    knn_iterative[:, 0] = range(features + 1)[1:]
    knn_rank = []
    knn_rank_b = []

    for i in range(features):
        iter = list(range(features))
        for k in knn_rank:
            iter.remove(k)
        knn_iter = np.zeros((len(iter), 3))
        knn_iter[:, 0] = iter
        idx = np.random.choice(len(train_y), 2000, replace=False)

        for (x, k) in enumerate(iter):
            sub = copy.copy(knn_rank)
            sub.append(k)
            x_tr = train_x_norm[idx, :][:,sub]
            y_tr = train_y[idx]

            cv = 5
            kf = KFold(n_splits=cv, shuffle=True, random_state=100)

            knn_res = np.zeros(3)
            knn_ns = np.array([9, 81, 200])

            for train_i, test_i in kf.split(x_tr):
                knn_l = (KNeighborsClassifier(n_neighbors=9).fit(x_tr[train_i], y_tr[train_i]),
                         KNeighborsClassifier(n_neighbors=81).fit(x_tr[train_i], y_tr[train_i]),
                         KNeighborsClassifier(n_neighbors=200).fit(x_tr[train_i], y_tr[train_i]))
                knn_res_x = []
                for knn_x in knn_l:
                    knn_res_x.append(tt_errors(knn_x, x_tr[train_i], x_tr[test_i], y_tr[train_i], y_tr[test_i])[1])
                knn_res += np.array(knn_res_x)
            knn_res /= cv

            knn_iter[x, 1] = knn_ns[np.argmax(knn_res)]
            knn_iter[x, 2] = np.max(knn_res)
        knn_max = knn_iter[np.argmax(knn_iter[:, 2]), :]
        knn_rank.append(int(knn_max[0]))
        knn_iterative[i, 1:] = knn_max
        print(i)

    knn_n = np.zeros((21, 3))
    knn_n[:, 0] = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1280, 1280+768]
    cv = 5
    kf = KFold(n_splits=cv, shuffle=True, random_state=10)

    train_x_norm_sub = train_x_norm[:, knn_rank[:11]]
    test_x_norm_sub = test_x_norm[:, knn_rank[:11]]
    for i in range(len(knn_n)):
        for train_i, test_i in kf.split(train_x):
            knn_x = KNeighborsClassifier(n_neighbors=int(knn_n[i, 0])).fit(train_x_norm_sub[train_i], train_y[train_i])

            train_acc, test_acc = tt_errors(knn_x, train_x_norm_sub[train_i], train_x_norm_sub[test_i], train_y[train_i],
                                            train_y[test_i])

            knn_n[i, 1] += train_acc
            knn_n[i, 2] += test_acc
        knn_n[i, 1] /= cv
        knn_n[i, 2] /= cv

    knn_features = np.zeros((features, 3))
    knn_features[:, 0] = range(features + 1)[1:]
    cv = 5
    kf = KFold(n_splits=cv, shuffle=True, random_state=100)

    for i in range(len(knn_features)):
        train_x_norm_sub = train_x_norm[:, knn_rank[:i + 1]]
        for train_i, test_i in kf.split(train_x):
            knn_x = KNeighborsClassifier(n_neighbors=64).fit(train_x_norm_sub[train_i], train_y[train_i])

            train_acc, test_acc = tt_errors(knn_x, train_x_norm_sub[train_i], train_x_norm_sub[test_i], train_y[train_i],
                                            train_y[test_i])

            knn_features[i, 1] += train_acc
            knn_features[i, 2] += test_acc
        knn_features[i, 1] /= cv
        knn_features[i, 2] /= cv

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.16)
    plt.title('Articles: KNN Accuracy vs Num Neighbors')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    sns.lineplot(y=knn_n[:, 1], x=knn_n[:, 0], label='Train Fold Accuracy')
    sns.lineplot(y=knn_n[:, 2], x=knn_n[:, 0], label='Val Fold Accuracy')
    ax.set_xscale('log', basex=2)
    plt.savefig('Articles KNN Accuracy vs Num Neighbors')

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Articles: KNN Accuracy vs Num Features Used')
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy')
    sns.lineplot(y=knn_features[:, 1], x=knn_features[:, 0], label='Train Fold Accuracy')
    sns.lineplot(y=knn_features[:, 2], x=knn_features[:, 0], label='Val Fold Accuracy')
    plt.savefig('Articles KNN Accuracy vs Num Features')


    svm_kernel = np.zeros((4, 3)).astype(object)
    svm_kernel[:, 0] = ['linear','poly','rbf','sigmoid']
    txt_mean = train_x.mean(axis=0)
    txt_std = train_x.std(axis=0)
    train_x_norm = (train_x - txt_mean) / txt_std

    for i in range(len(svm_kernel)):
        for train_i, test_i in kf.split(train_x):
            tree_x = SVC(kernel=svm_kernel[i,0]).fit(train_x_norm[train_i],train_y[train_i])

            train_acc, test_acc = tt_errors(tree_x, train_x_norm[train_i], train_x_norm[test_i], train_y[train_i], train_y[test_i])
            svm_kernel[i, 1] += train_acc
            svm_kernel[i, 2] += test_acc
        svm_kernel[i, 1] /= cv
        svm_kernel[i, 2] /= cv

    svm_rbf = np.zeros((15, 3))
    svm_rbf[:, 0] = [.001, 0.002, 0.003, 0.005, 0.0075, .01, .02, 0.03, 0.04, .05, .1, .2, .3,  .4, .5]

    txt_mean = train_x.mean(axis=0)
    txt_std = train_x.std(axis=0)
    train_x_norm = (train_x - txt_mean) / txt_std
    test_x_norm = (test_x - txt_mean) / txt_std

    for i in range(len(svm_rbf)):
        for train_i, test_i in kf.split(train_x):
            tree_x = SVC(kernel='rbf', gamma=svm_rbf[i, 0]).fit(train_x_norm[train_i],train_y[train_i])

            train_acc, test_acc = tt_errors(tree_x, train_x_norm[train_i], train_x_norm[test_i], train_y[train_i], train_y[test_i])
            svm_rbf[i, 1] += train_acc
            svm_rbf[i, 2] += test_acc
        svm_rbf[i, 1] /= cv
        svm_rbf[i, 2] /= cv

    if True:

        svm_rbf_b = np.zeros((11, 3))
        svm_rbf_b[:, 0] = [.001, 0.005, .01, .025, .05, .1, .25, .5,.75, 1, 2]
        txt_mean = train_x.mean(axis=0)
        txt_std = train_x.std(axis=0)
        train_x_norm = (train_x - txt_mean) / txt_std

        train_x_norm_sub_svm = train_x_norm[:, knn_rank[:11]]

        for i in range(len(svm_rbf_b)):
            for train_i, test_i in kf.split(train_x):
                tree_x = SVC(kernel='rbf', gamma=svm_rbf_b[i, 0]).fit(train_x_norm_sub_svm[train_i], train_y[train_i])

                train_acc, test_acc = tt_errors(tree_x, train_x_norm_sub_svm[train_i], train_x_norm_sub_svm[test_i], train_y[train_i],
                                                train_y[test_i])
                svm_rbf_b[i, 1] += train_acc
                svm_rbf_b[i, 2] += test_acc
            svm_rbf_b[i, 1] /= cv
            svm_rbf_b[i, 2] /= cv

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Article: SVM Accuracy vs Kernel Type')
    plt.xlabel('Kernel')
    plt.ylabel('Accuracy')
    sns.lineplot(y=svm_kernel[:, 1], x=svm_kernel[:, 0], label='Train Fold Accuracy')
    sns.lineplot(y=svm_kernel[:, 2], x=svm_kernel[:, 0], label='Val Fold Accuracy')
    plt.savefig('Article SVM Accuracy vs kernel')

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.16)
    plt.title('Article: RBF SVM Accuracy vs Gamma')
    plt.xlabel('Gamma')
    plt.ylabel('Accuracy')
    sns.lineplot(y=svm_rbf[:, 1], x=svm_rbf[:, 0], label='Train Fold Accuracy')
    sns.lineplot(y=svm_rbf[:, 2], x=svm_rbf[:, 0], label='Val Fold Accuracy')
    ax.set_xscale('log', basex=10)
    for xy in zip(svm_rbf[5:6, 0], svm_rbf[5:6, 2]):
        t = (xy[0], round(xy[1],3))
        ax.annotate('(%s, %s)' % t, xy=t, textcoords='data', ha='center')
    plt.savefig('Article SVM Accuracy vs gamma')

    sns.set()
    fig, ax = plt.subplots(figsize=(5, 2.5))
    plt.subplots_adjust(bottom=.3)
    plt.subplots_adjust(left=.16)
    plt.title('Article (11 features): RBF SVM Accuracy vs Gamma')
    plt.xlabel('Gamma')
    plt.ylabel('Accuracy')
    sns.lineplot(y=svm_rbf_b[:, 1], x=svm_rbf_b[:, 0], label='Train Fold Accuracy')
    sns.lineplot(y=svm_rbf_b[:, 2], x=svm_rbf_b[:, 0], label='Val Fold Accuracy')
    ax.set_xscale('log', basex=10)
    for xy in zip(svm_rbf_b[4:5, 0], svm_rbf_b[4:5, 2]):
        t = (xy[0], round(xy[1],3))
        ax.annotate('(%s, %s)' % t, xy=t, textcoords='data', ha='center')
    plt.savefig('Article SVM_B Accuracy vs gamma')


    def train_cpu(epoch, data_tens_train, batch_size, model, optimizer, criterion):

        acc = AverageMeter()
        splits = int(len(data_tens_train[0]) / batch_size)
        if splits < 2:
            splits = 2
        kf = KFold(n_splits=splits)
        for idx, (train_i, test_i) in enumerate(kf.split(data_tens_train[0])):
            data = data_tens_train[0][test_i]
            target = data_tens_train[1][test_i]

            optimizer.zero_grad()
            out = model.forward(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            batch_acc = accuracy(out, target)

            acc.update(batch_acc, out.shape[0])

        torch.cuda.empty_cache()

    def validate_cpu(epoch, data_tens, batch_size, model, criterion, num_class):

        acc = AverageMeter()

        cm = torch.zeros(num_class, num_class)
        splits = int(len(data_tens[0]) / batch_size)
        if splits < 2:
            splits = 2
        kf = KFold(n_splits=splits)
        # evaluation loop
        for idx, (train_i, test_i) in enumerate(kf.split(data_tens[0])):
            data = data_tens[0][test_i]
            target = data_tens[1][test_i]
            # torch.cuda.empty_cache()

            with torch.no_grad():
                out = model(data)
                loss = criterion(out, target)

            batch_acc = accuracy(out, target)

            # update confusion matrix
            _, preds = torch.max(out, 1)
            for t, p in zip(target.view(-1), preds.view(-1)):
                cm[t.long(), p.long()] += 1

            acc.update(batch_acc, out.shape[0])

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
    def bestLayerNetReturnModel(train_x, train_y, test_x, test_y, batch_size=32, val_batch_size=32, learning_rate = 0.01, momentum = 0.9, reg = 0.0001, epochs = 10):

        nn_model = LayerNet_n_cpu(features, 256, 2, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate, momentum=momentum, weight_decay=reg)

        best = 0.0
        best_cm = None
        best_model = None

        train_x_tens = torch.tensor(train_x, dtype=torch.float32).cpu()
        val_x_tens = torch.tensor(test_x, dtype=torch.float32).cpu()
        train_y_tens = torch.tensor(train_y, dtype=torch.long).cpu()
        val_y_tens = torch.tensor(test_y, dtype=torch.long).cpu()
        txt_mean = train_x_tens.mean(axis=0)
        txt_std = train_x_tens.std(axis=0)
        train_x_tens = (train_x_tens - txt_mean) / txt_std
        val_x_tens = (val_x_tens - txt_mean) / txt_std

        for epoch in range(epochs):
            data_tens_train = (train_x_tens, train_y_tens)
            data_tens_val = (val_x_tens, val_y_tens)
            if epoch == 10:
                optimizer = torch.optim.SGD(nn_model.parameters(), learning_rate / 10,
                                            momentum=momentum,
                                            weight_decay=reg)

            train_cpu(epoch, data_tens_train, batch_size, nn_model, optimizer, criterion)

            acc, cm = validate_cpu(epoch, data_tens_val, val_batch_size, nn_model, criterion, 2)
            # tr_acc, tr_cm = validate(epoch, data_tens_train, val_batch_size, nn_model, criterion, 2)

            if acc > best:
                best = acc
                best_cm = cm
                best_model = copy.deepcopy(nn_model)


        return best_model

    def test_errors(model, test_x, test_y):
        pred_test = model.predict(test_x)
        test_acc = sum(pred_test == test_y) / len(pred_test)
        return test_acc

    def nn_acc(model, data_tens_x, data_tens_y ):
        nn_x = data_tens_x
        target = data_tens_y
        with torch.no_grad():
            out = model(nn_x)
        return accuracy(out, target).cpu().numpy().reshape(1)[0]

    train_x_norm_sub = train_x_norm[:,knn_rank[:11]]
    test_x_norm_sub = test_x_norm[:,knn_rank[:11]]
    train_x_tens_cpu = train_x_tens.cpu()
    test_x_tens_cpu = test_x_tens.cpu()
    train_y_tens_cpu = train_y_tens.cpu()
    test_y_tens_cpu = test_y_tens.cpu()

    final_dt = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_leaf=192,ccp_alpha=0.0008).fit(train_x, train_y)
    final_knn = KNeighborsClassifier(n_neighbors=200).fit(train_x_norm_sub, train_y)
    final_nn = bestLayerNetReturnModel(train_x, train_y, test_x, test_y)
    final_svm = SVC(kernel='rbf', gamma=0.01).fit(train_x_norm,train_y)
    final_boost = TreeBoost(n_estimators=128, max_depth=1).fit(train_x, train_y)



    final_accuracies = np.zeros((5,3)).astype(object)
    final_accuracies[:,0] = ['Decision Tree', 'Neural Network', 'Boosting', 'SVM', 'KNN']
    final_accuracies[:,1] = [test_errors(final_dt, train_x, train_y), nn_acc(final_nn, train_x_tens_cpu, train_y_tens_cpu),
                             test_errors(final_boost, train_x, train_y), test_errors(final_svm, train_x_norm, train_y),
                             test_errors(final_knn, train_x_norm_sub, train_y)]
    final_accuracies[:,2] = [test_errors(final_dt, test_x, test_y), nn_acc(final_nn, test_x_tens_cpu, test_y_tens_cpu),
                             test_errors(final_boost, test_x, test_y), test_errors(final_svm, test_x_norm, test_y),
                             test_errors(final_knn, test_x_norm_sub, test_y)]

    final_models = np.zeros((5,6)).astype(object)
    final_models[:,0] = [final_dt, final_nn, final_boost, final_svm, final_knn]
    final_models[:,1] = [test_errors, nn_acc, test_errors, test_errors, test_errors]
    final_models[:,2] = [train_x, train_x_tens_cpu, train_x, train_x_norm, train_x_norm_sub]
    final_models[:,3] = [train_y, train_y_tens_cpu, train_y, train_y, train_y]
    final_models[:,4] = [test_x, test_x_tens_cpu, test_x, test_x_norm, test_x_norm_sub]
    final_models[:,5] = [test_y, test_y_tens_cpu, test_y, test_y, test_y]


    train_time = np.zeros((5,2)).astype(object)
    train_time[:,0] = ['DTree', 'NN', 'Boosting', 'SVM', 'KNN']

    if True:
        start_time = time.time()
        for i in range(10):
            DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_leaf=192,ccp_alpha=0.0008).fit(train_x, train_y)
        end_time = time.time()
        train_time[0,1] = (end_time - start_time) / 10 * 1000

        start_time = time.time()
        for i in range(10):
            bestLayerNetReturnModel(train_x, train_y, test_x, test_y, epochs=10)
        end_time = time.time()
        train_time[1, 1] = (end_time - start_time) / 10 * 1000

        start_time = time.time()
        for i in range(10):
            TreeBoost(n_estimators=128, max_depth=1).fit(train_x, train_y)
        end_time = time.time()
        train_time[2, 1] = (end_time - start_time) / 10 * 1000

        start_time = time.time()
        for i in range(10):
            SVC(kernel='rbf', gamma=0.01).fit(train_x_norm,train_y)
        end_time = time.time()
        train_time[3, 1] = (end_time - start_time) / 10 * 1000

        start_time = time.time()
        for i in range(10):
            KNeighborsClassifier(n_neighbors=200).fit(train_x_norm_sub, train_y)
        end_time = time.time()
        train_time[4, 1] = (end_time - start_time) / 10 * 1000

    query_time = np.zeros((5,2)).astype(object)
    query_time[:,0] = ['DTree', 'NN', 'Boosting', 'SVM', 'KNN']

    for i in range(5):

        start_time = time.time()
        for k in range(10):
            _ = final_models[i,1](final_models[i,0], final_models[i,4], final_models[i,5])
        end_time = time.time()
        query_time[i, 1] = (end_time - start_time) / 10 * 1000

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.19)
    plt.title('Article: Algorithm Train Times')
    plt.xlabel('Algorithm')
    plt.ylabel('Milliseconds')
    k = range(5)
    ax.bar(k, train_time[:, 1])
    plt.xticks(k,train_time[:, 0])
    # sns.barplot(y=train_time[:, 1], x=train_time[:, 0])
    for xy in zip(k, train_time[:, 1]):
        t = round(xy[1],1)
        ax.annotate('%s ms' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
    plt.savefig('Article train times')

    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.19)
    plt.title('Article: Algorithm Query Times')
    plt.xlabel('Algorithm')
    plt.ylabel('Milliseconds')
    k = range(5)
    ax.bar(k, query_time[:, 1])
    plt.xticks(k,query_time[:, 0])
    # sns.barplot(y=query_time[:, 1], x=query_time[:, 0])
    for xy in zip(k, query_time[:, 1]):
        t = round(xy[1],1)
        ax.annotate('%s ms' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
    plt.savefig('Article query times')



    dataset_fractions = [.01, .02, 0.05, 0.1, .15, 0.2, .25, 0.3, .35, 0.4, .45, 0.5, .55, 0.6,.65, 0.7,.75, 0.8,.85, 0.9,.95, 1]
    dataset_sample_fractions = np.array(dataset_fractions) * len(train_x)
    frac_train = np.zeros((22,6))
    frac_train[:,0] = dataset_fractions
    frac_val = np.zeros((22,6))
    frac_val[:,0] = dataset_fractions
    avgs = 10

    for i,f in enumerate(dataset_fractions):

        tr_x_list = []
        tr_y_list = []
        if f == 1:
            f = .999999

        tr_x, tr_y = [],[]
        f_dt = []
        for x in range(avgs):
            tr_xi, _, tr_yi, _ = train_test_split(final_models[0, 2], final_models[0, 3], test_size=1 - f, random_state=x*10)
            tr_x.append(tr_xi)
            tr_y.append(tr_yi)
            f_dt.append(DecisionTreeClassifier(random_state=x, max_depth=5, min_samples_leaf=192,ccp_alpha=0.0008).fit(tr_xi, tr_yi))
        tr_x_list.append(tr_x)
        tr_y_list.append(tr_y)

        tr_x, tr_y = [],[]
        f_nn = []
        for x in range(avgs):
            tr_xi, _, tr_yi, _ = train_test_split(final_models[1, 2], final_models[1, 3], test_size=1 - f, random_state=x*10)
            tr_x.append(tr_xi)
            tr_y.append(tr_yi)
            f_nn.append(bestLayerNetReturnModel(tr_xi, tr_yi, test_x_tens_cpu, test_y_tens_cpu))
        tr_x_list.append(tr_x)
        tr_y_list.append(tr_y)

        tr_x, tr_y = [],[]
        f_boo = []
        for x in range(avgs):
            tr_xi, _, tr_yi, _ = train_test_split(final_models[2, 2], final_models[2, 3], test_size=1 - f, random_state=x*10)
            tr_x.append(tr_xi)
            tr_y.append(tr_yi)
            f_boo.append(TreeBoost(n_estimators=128, max_depth=1).fit(tr_xi, tr_yi))
        tr_x_list.append(tr_x)
        tr_y_list.append(tr_y)

        tr_x, tr_y = [],[]
        f_svm = []
        for x in range(avgs):
            tr_xi, _, tr_yi, _ = train_test_split(final_models[3, 2], final_models[3, 3], test_size=1 - f, random_state=x*10)
            tr_x.append(tr_xi)
            tr_y.append(tr_yi)
            f_svm.append(SVC(kernel='rbf', gamma=0.01).fit(tr_xi, tr_yi))
        tr_x_list.append(tr_x)
        tr_y_list.append(tr_y)

        tr_x, tr_y = [],[]
        f_knn = []
        for x in range(avgs):
            tr_xi, _, tr_yi, _ = train_test_split(final_models[4, 2], final_models[4, 3], test_size=1 - f, random_state=x*10)
            tr_x.append(tr_xi)
            tr_y.append(tr_yi)
            f_knn.append(KNeighborsClassifier(n_neighbors=min(200,len(tr_yi))).fit(tr_xi, tr_yi))
        tr_x_list.append(tr_x)
        tr_y_list.append(tr_y)


        model_list = [f_dt, f_nn, f_boo, f_svm, f_knn]

        for k in range(5):
            for c,modelx in enumerate(model_list[k]):
                frac_train[i, k + 1] += final_models[k,1](modelx, tr_x_list[k][c], tr_y_list[k][c])
                frac_val[i, k + 1] += final_models[k,1](modelx, final_models[k,4], final_models[k,5])
            frac_train[i, k + 1] /= avgs
            frac_val[i, k + 1] /= avgs
        print(i)


    fig, ax = plt.subplots(figsize=(4.5, 2.25))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.15)
    plt.title('Articles: Train Accuracy vs Training Samples Used')
    plt.xlabel('Training Samples (min = 83)')
    plt.ylabel('Accuracy')
    k = range(5)
    sns.lineplot(y=frac_train[:, 1], x=dataset_sample_fractions, label='DT')
    sns.lineplot(y=frac_train[:, 2], x=dataset_sample_fractions, label='NN')
    sns.lineplot(y=frac_train[:, 3], x=dataset_sample_fractions, label='Boosting')
    sns.lineplot(y=frac_train[:, 4], x=dataset_sample_fractions, label='SVM')
    sns.lineplot(y=frac_train[:, 5], x=dataset_sample_fractions, label='KNN')
    # sns.barplot(y=query_time[:, 1], x=query_time[:, 0])
    plt.legend(loc='lower center')
    plt.savefig('Article train frac')

    fig, ax = plt.subplots(figsize=(4.5, 2.25))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.15)
    plt.title('Articles: Test Accuracy vs Training Samples Used')
    plt.xlabel('Training Samples (min = 83)')
    plt.ylabel('Accuracy')
    k = range(5)
    sns.lineplot(y=frac_val[:, 1], x=dataset_sample_fractions, label='DT')
    sns.lineplot(y=frac_val[:, 2], x=dataset_sample_fractions, label='NN')
    sns.lineplot(y=frac_val[:, 3], x=dataset_sample_fractions, label='Boosting')
    sns.lineplot(y=frac_val[:, 4], x=dataset_sample_fractions, label='SVM')
    sns.lineplot(y=frac_val[:, 5], x=dataset_sample_fractions, label='KNN')
    plt.legend(loc='lower center')
    # sns.barplot(y=query_time[:, 1], x=query_time[:, 0])
    plt.savefig('Article test frac')





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
