import torch

def format_pytorch_version(version):

  return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)

!pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
!pip install torch-geometric

!pip install networkx==3.3
!sudo apt install libcairo2-dev pkg-config python3-dev
!pip install manimlib
!pip install pycairo
!pip install cairocffi
!pip install karateclub

!pip install karateclub

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import community
from karateclub import LabelPropagation
import pandas as pd
import time
from google.colab import drive
drive.mount('/content/drive')

def create_community_node_colors(graph, communities):
    number_of_colors = len(communities[0])
    colors = ["#D4FCB1", "#CDC5FC", "#FFC2C4", "#F2D140", "#BCC6C8"][:number_of_colors]
    node_colors = []
    for node in graph:
        current_community_index = 0
        for community in communities:
            if node in community:
                node_colors.append(colors[current_community_index])
                break
            current_community_index += 1
    return node_colors
def visualize_communities(graph, communities, i):
    node_colors = create_community_node_colors(graph, communities)
    modularity = round(nx.community.modularity(graph, communities), 6)
    title = f"Community Visualization of {len(communities)} communities with modularity of {modularity}"
    pos = nx.spring_layout(graph, seed=1)
    plt.subplot(3, 1, i)
    plt.title(title)
    nx.draw(
        graph,
        pos=pos,
        node_size=200,
        node_color=node_colors,
        with_labels=True,
        font_size=5,
        font_color="black",
    )

data = pd.read_table('/content/drive/MyDrive/NCKH/train.txt',header=None)
G = nx.Graph()
for rows in data.iterrows():
    row = rows[1][0].split()
    for i in range(len(row[1:])):
        G.add_edge(int(row[0]),int(row[i+1]))
print(f"NetworkX graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
start = time.time()
k_clique_comms = list(nx.community.k_clique_communities(G,3))
end = time.time()
print(end-start)
k_clique_df = pd.DataFrame(k_clique_comms)
k_clique_df.to_csv('k_clique_df.csv')
start = time.time()
communities = list(nx.community.girvan_newman(G))
end = time.time()
print(end-start)
print(communities)
model = LabelPropagation(seed=23, iterations=200)
start_time = time.time()
model.fit(G)
cluster_membership = model.get_memberships()
end_time = time.time()
label_propagation_complexity = end_time - start_time
label_propagation_complexity
