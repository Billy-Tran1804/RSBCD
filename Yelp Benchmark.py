!pip install karateclub

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import networkx as nx
from karateclub import LabelPropagation
import time
import matplotlib as plt
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import random
import torch
# from torch_geometric.datasets import TUDataset
# from torch_geometric.utils import to_networkx
import community

# graph = nx.newman_watts_strogatz_graph(100, 20, 0.05)

# model = LabelPropagation()
# start_time = time.time()
# model.fit(graph)
# cluster_membership = model.get_memberships()
# end_time = time.time()
# time_complexity = end_time - start_time
# time_complexity

# G = nx.Graph()

# # Add nodes with cluster as the node attribute
# for node, cluster in cluster_membership.items():
#     G.add_node(node, cluster=cluster)

# # Add edges between nodes in the same cluster
# for edge in edges:
#     node1 = edge[0]
#     node2 = edge[1]
#     if cluster_membership[node1] != cluster_membership[node2]:
#         graph.remove_edge(node1, node2)

# # Draw the graph
# # plt.figure(figsize=(10, 8))
# # pos = nx.spring_layout(G, seed=42)  # Positions for all nodes
# # nx.draw(graph, pos, node_color=[cluster_membership[node] for node in G.nodes()], with_labels=True, cmap='viridis', node_size=300, edge_color='gray', width=1.0)
# # plt.title('Cluster Visualization with Nodes and Edges')
# # # plt.colorbar(label='Cluster')
# # plt.show()

# communities = {}
# for node, community in cluster_membership.items():
#     if community not in communities:
#         communities[community] = [node]
#     else:
#         communities[community].append(node)

# # Draw the graph with nodes colored by community
# plt.figure(figsize=(10, 8))
# pos = nx.spring_layout(G)
# for i, (community, nodes) in enumerate(communities.items()):
#     nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=plt.cm.tab20(i), label=f'Community {community}')
# nx.draw_networkx_edges(G, pos, alpha=0.5)
# plt.title('Graph with Communities')
# plt.legend()
# plt.show()

#Yelp dataset
import json
import networkx as nx
G = nx.Graph()
# Extract nodes (users and businesses)
nodes = set()
edges = []

# Reading Review data in chunks because of memory restrictions

for chunk in pd.read_json('/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json', chunksize=50000, lines=True):
    for review in chunk.itertuples():
        nodes.add(review[2])
        nodes.add(review[3])
        edges.append((review[2], review[3]))

G.add_nodes_from(nodes)
G.add_edges_from(edges)

mapping = {node: i for i, node in enumerate(G.nodes())}
nx.relabel_nodes(G, mapping, copy=False)

# Label Propagation

# model = LabelPropagation(seed=23, iterations=200)
# start_time = time.time()
# model.fit(G)
# cluster_membership = model.get_memberships()
# end_time = time.time()
# time_complexity = end_time - start_time
# print("Time complexity: ", time_complexity/60)
# f = open("/kaggle/working/log.txt", "a")
# f.write("Yelp: f{time_complexity/60}")
# f.close()

# Girvan
# !conda install -y -c rapidsai -c nvidia -c conda-forge cugraph=VERSION python=PYTHON_VERSION cudatoolkit=CUDA_VERSION
# import cugraph as cnx
# start_time = time.time()

# communities = list(nx.community.girvan_newman(G))

# time_complexity = time.time() - start_time
# print("--- %s minutes ---" % time_complexity/60)

# f = open("/kaggle/working/log.txt", "a")
# f.write("Girvan: f{time_complexity/60}")
# f.close()

# # Modularity -> measures the strength of division of a network into modules
# df = pd.DataFrame(
#     [
#         [k + 1, nx.community.modularity(G, communities[k])]
#         for k in range(len(communities))
#     ],
#     columns=["k", "modularity"],
# )
# max_id = df['modularity'].idxmax()
# k_max = df.loc[max_id, 'k']
# print(k_max)
# # G_girvan = communities[k_max - 1]
# # # print(G_girvan)

# Clique Percolation
start_time = time.time()
c = list(nx.community.k_clique_communities(G,100))
time_complexity = time.time() - start_time
print(time_complexity/60)

f = open("/kaggle/working/log.txt", "a")
f.write("Overlapping: f{time_complexity/60}")
f.close()