import networkx as nx
from dgl.data import CoraFullDataset
import community as community_louvain
import numpy as np
import torch
from sklearn.metrics import classification_report, jaccard_score

from clan import clan
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

# Load the dataset and extract ground truth communities
dataset = CoraFullDataset()
graph = dataset[0]
features = graph.ndata['feat']
labels = graph.ndata['label'] 

true_communities = dict()
for i, label in enumerate(labels):
    label = label.item()
    if label in true_communities:
        true_communities[label].append(i)
    else:
        true_communities[label] = [i]

# Perform the Louvain method and find its communities
G = graph.to_networkx().to_undirected()
partition = community_louvain.best_partition(G, randomize=False)

louvain_communities = dict()
for k, v in partition.items():
    if v in louvain_communities:
        louvain_communities[v].append(k)
    else:
        louvain_communities[v] = [k]

# Map the communities from Louvain to Ground truth communities
# By finding the biggest intersection
map_louvain_gt = dict()
for k_louvain, v_louvain in louvain_communities.items():
    intersection_length = []
    for _, v_truth in true_communities.items():
        intersection = [value for value in v_truth if value in v_louvain]
        intersection_length.append(len(intersection))
    i = np.argmax(intersection_length)
    map_louvain_gt.update({k_louvain: i})

# Calculating the F1 and Jaccard Score for Louvain
lovain_pred_labels = [map_louvain_gt[x] for _, x in partition.items()]

print("Louvain report")
print(classification_report(labels.tolist(), lovain_pred_labels, zero_division=True))
print(jaccard_score(labels.tolist(), lovain_pred_labels, zero_division=True))

# Perform CLAN
clan_communities = clan(louvain_communities, theta=20, features=features, device=device)

# Map the communities from Louvain to Ground truth communities
# By finding the biggest intersection
map_clan_gt = dict()
for k_clan, v_clan in clan_communities.items():
    intersection_length = []
    for _, v_truth in true_communities.items():
        intersection = [value for value in v_truth if value in v_clan]
        intersection_length.append(len(intersection))
    i = np.argmax(intersection_length)
    map_clan_gt.update({k_clan: i})

# Calculating the F1 and Jaccard Score for CLAN
clan_pred_labels = torch.zeros(labels.shape[0])
for k, values in clan_communities.items():
    for v in values:
        clan_pred_labels[v] = map_clan_gt[k]

print("CLAN report")
print(classification_report(labels.tolist(), clan_pred_labels, zero_division=True))
print(jaccard_score(labels.tolist(), clan_pred_labels, zero_division=True))



