import torch
import numpy as np
import networkx as nx
import community as community_louvain
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from dgl.data import CoraFullDataset, RedditDataset
from clan import clan, compute_f1


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)

dataset = CoraFullDataset()
graph = dataset[0]
features = graph.ndata['feat']
labels = graph.ndata['label'] 

# ground truth: dict(communities, nodes)
true_communities = dict()
for i, label in enumerate(labels):
    label = label.item()
    if label in true_communities:
        true_communities[label].append(i)
    else:
        true_communities[label] = [i]

#first compute the best partition
graph = graph.to_networkx().to_undirected()
# partition = nx.algorithms.community.louvain_communities(graph)
partition = community_louvain.best_partition(graph)

# Louvain: dict(community, nodes)
louvain_communities = dict()
for k, v in partition.items():
    if v in louvain_communities:
        louvain_communities[v].append(k)
    else:
        louvain_communities[v] = [k]

# second perform CLAN
clan_communities = clan(louvain_communities, theta=4, features=features, num_classes=50, device=device)

# Community mapping: dict(louvain, ground truth)
map_louvain_gt = dict()
for k_louvain, v_louvain in louvain_communities.items():
    intersection_length = []
    for _, v_truth in true_communities.items():
        intersection = [value for value in v_truth if value in v_louvain]
        intersection_length.append(len(intersection))
    i = np.argmax(intersection_length)
    map_louvain_gt.update({k_louvain: i})

# Community mapping: dict(clan, ground truth)
map_clan_gt = dict()
for k_clan, v_clan in clan_communities.items():
    intersection_length = []
    for _, v_truth in true_communities.items():
        intersection = [value for value in v_truth if value in v_clan]
        intersection_length.append(len(intersection))
    i = np.argmax(intersection_length)
    map_clan_gt.update({k_clan: i})


# calculate similarity with F1
predicted_nodes = torch.zeros(labels.shape[0])
for k, values in clan_communities.items():
    for v in values:
        predicted_nodes[v] = map_clan_gt[k]

louvain_nodes = torch.zeros(labels.shape[0])
for k, values in louvain_communities.items():
    for v in values:
        louvain_nodes[v] = map_louvain_gt[k]

f1_clan_gt = compute_f1(true_communities, labels, clan_communities, predicted_nodes)
f1_louvain_gt = compute_f1(true_communities, labels, louvain_communities, louvain_nodes)
f1_clan_louvain = compute_f1(louvain_communities, louvain_nodes, clan_communities, predicted_nodes)

print(f"F1 Score CLAN: {f1_clan_gt}")
print(f"F1 Score Louvain: {f1_louvain_gt}")
print(f"F1 Score CLAN-Louvain: {f1_clan_louvain}")


# draw the graph
pos = nx.spring_layout(graph)
# color the nodes according to their partition
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_nodes(graph, pos, partition.keys(), node_size=40,
                       cmap=cmap, node_color=list(partition.values()))
nx.draw_networkx_edges(graph, pos, alpha=0.5)
plt.show()