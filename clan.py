from model import FF
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score

def clan(communities, theta, features, device):
    writer = SummaryWriter()
    train_nodes, train_labels, test_nodes, test_labels = [], [], [], []

    # create a train dataset using node attributes as features
    for c in communities:
        if len(communities[c]) > theta:
            train_nodes.append(communities[c])
            train_labels.append(c)
        else:
            test_nodes.append(communities[c])
            test_labels.append(c)
    

    # create a model, optimizer et cetera
    num_input = features.shape[1]
    num_clusters = len(communities) - 1
    model = FF(num_input, num_clusters, device)
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()

    # train model on train set
    n_iter = 0
    for epoch in range(15):
        running_loss = 0
        for i, (nodes, label) in enumerate(zip(train_nodes, train_labels)):
            label = torch.tensor(label - 1, dtype=torch.int64)
            y = torch.nn.functional.one_hot(label, num_classes=num_clusters)
            y = y.expand(len(nodes), len(y)).to(device)
            X = features[nodes].to(device)
        
            optim.zero_grad()
            pred = model(X)
            loss = criterion(pred, y.float())
            loss.backward()
            optim.step()

            running_loss += loss
            if i % 10 == 0:
                writer.add_scalar('Train_Loss', loss, n_iter)
                # print(f"Iteration {n_iter}/{len(features[1])}: {loss}")
                running_loss = 0
            n_iter += 1
    new_communities = communities.copy()

    # classify the "outliers"
    for i, (nodes, label) in enumerate(zip(test_nodes, test_labels)):
        X = features[nodes].to(device)
        prediction = model(X)
        community = prediction.argmax() 
        community = community % prediction.shape[1]

        index = community.item()
        if index in new_communities:
            new_communities[index].extend(new_communities[label]) 
        else:
            new_communities.update({index: new_communities[label]}) 
        new_communities[label] = []

    return new_communities


def compute_f1(true_communities, true_labels, pred_communities, pred_labels):
    f1, f2 = [], []
    for _, community in true_communities.items():
        # use mapping to compare backwards
        pred = pred_labels[community].tolist()
        truth = true_labels[community].tolist()
        score = f1_score(truth, pred, average='micro')
        f1.append(np.mean(score))

    for _, community in pred_communities.items():
        if len(community) == 0:
            continue
        pred = pred_labels[community].tolist()
        truth = true_labels[community].tolist()
        score = f1_score(truth, pred, average='micro')
        f2.append(np.mean(score))

    f1_1 = np.mean(f1)
    f1_2 = np.mean(f2)

    return (f1_1 + f1_2)/2