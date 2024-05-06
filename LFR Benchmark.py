# Add this in a Google Colab cell to install the correct version of Pytorch Geometric.
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

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
import community

# function to create node colour list
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


# function to plot graph with node colouring based on communities
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

    dataset = TUDataset("/content/TUDataset", "ENZYMES")
    data = dataset[0]

    # Convert to NetworkX graph
    G = to_networkx(data)
    communities = list(nx.community.girvan_newman(G))

    # Modularity -> measures the strength of division of a network into modules
    modularity_df = pd.DataFrame(
        [
            [k + 1, nx.community.modularity(G, communities[k])]
            for k in range(len(communities))
        ],
        columns=["k", "modularity"],
    )

    fig, ax = plt.subplots(3, figsize=(15, 20))

    # Plot graph with colouring based on communities
    visualize_communities(G, communities[0], 1)
    visualize_communities(G, communities[1], 2)
    # visualize_communities(G, communities[2], 2)

    # Plot change in modularity as the important edges are removed
    modularity_df.plot.bar(
        x="k",
        ax=ax[2],
        color="#F2D140",
        title="Modularity Trend for Girvan-Newman Community Detection",
    )
    plt.show()

    import networkx as nx
    import matplotlib.pyplot as plt
    from networkx.generators.community import LFR_benchmark_graph

    # Generate LFR benchmark graph
    n = 100  # Number of nodes
    tau1 = 3  # Power law exponent for the degree distribution of the created graph
    tau2 = 2  # Power law exponent for the community size distribution in the created graph
    mu = 0.1  # Fraction of intra-community edges incident to each node
    seed = 1
    G = LFR_benchmark_graph(n, tau1, tau2, mu,
                            average_degree=5,
                            # min_degree = 2,
                            min_community=10,
                            max_community=50,
                            seed=seed)
    G.remove_edges_from(nx.selfloop_edges(G))

    # Assigning community attribute to each node
    communities = {frozenset(G.nodes[v]['community']): i for i, v in enumerate(G.nodes)}
    nx.set_node_attributes(G, communities, 'community')

    # Convert communities to integers for coloring
    node_color = [G.nodes[v]['community'] for v in G]
    community_colors = [node_color.index(i) for i in node_color]

    # Visualizing the graph with communities
    # pos = nx.spring_layout(G, seed=seed)
    # nx.draw(G, pos, node_color=community_colors, node_size=20, edge_color='gray', with_labels=False, cmap=plt.cm.tab10)
    # plt.show()

    # # Visualize the graph
    # pos = nx.spring_layout(G, seed=seed)  # Position nodes using Fruchterman-Reingold force-directed algorithm
    # nx.draw(G, pos, with_labels=False, node_size=20, node_color='blue', edge_color='gray', linewidths=0.5)
    # plt.title('LFR Community Network')
    # plt.show()
    ground_truth = {frozenset(G.nodes[v]['community']): i for i, v in enumerate(G.nodes)}
    ground_truth = tuple({s for s in d}for d in ground_truth)
    print(ground_truth)
    import sklearn
    from sklearn.metrics import normalized_mutual_info_score
    import time
    start_time = time.time()

    communities = list(nx.community.girvan_newman(G))
    # print("--- %s seconds ---" % (time.time() - start_time))
    # Modularity -> measures the strength of division of a network into modules
    df = pd.DataFrame(
        [
            [k + 1, nx.community.modularity(G, communities[k])]
            for k in range(len(communities))
        ],
        columns=["k", "modularity"],
    )
    max_id = df['modularity'].idxmax()
    k_max = df.loc[max_id, 'k']
    print(k_max)
    G_girvan = communities[k_max - 1]
    print(G_girvan)

    print("--- %s seconds ---" % (time.time() - start_time))

    # node_color = [G_girvan[v] for v in G_girvan]
    # community_colors = [node_color.index(i) for i in node_color]

    # # Visualizing the graph with communities
    # pos = nx.spring_layout(G_girvan, seed=seed)
    # nx.draw(G_girvan, pos, node_color=community_colors, node_size=20, edge_color='gray', with_labels=False, cmap=plt.cm.tab10)
    # plt.show()
    labels1 = [i for i, comm in enumerate(ground_truth) for _ in comm]
    labels2 = [i for i, comm in enumerate(G_girvan) for _ in comm]

    # Calculate NMI
    nmi = normalized_mutual_info_score(labels1, labels2)
    print("NMI = ", nmi)
    !pip
    install
    karateclub
    from karateclub import LabelPropagation
    model = LabelPropagation()
    model.fit(G)
    cluster_membership = model.get_memberships()

    communities = {}
    for node, community in cluster_membership.items():
        if community not in communities:
            communities[community] = [node]
        else:
            communities[community].append(node)

    # Draw the graph with nodes colored by community
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    for i, (community, nodes) in enumerate(communities.items()):
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=plt.cm.tab20(i), label=f'Community {community}')
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    plt.title('Graph with Communities')
    plt.legend()
    plt.show()
    import time
    start_time = time.time()

    # communities = list(nx.community.girvan_newman(G))
    model = LabelPropagation()
    model.fit(G)
    cluster_membership = model.get_memberships()
    communities = {}
    for node, community in cluster_membership.items():
        if community not in communities:
            communities[community] = [node]
        else:
            communities[community].append(node)
    # Modularity -> measures the strength of division of a network into modules
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
    # G_girvan = communities[k_max - 1]
    # print(G_girvan)

    print("--- %s seconds ---" % (time.time() - start_time))

    print(communities)
    print(ground_truth)
    # node_color = [G_girvan[v] for v in G_girvan]
    # community_colors = [node_color.index(i) for i in node_color]

    # # Visualizing the graph with communities
    # pos = nx.spring_layout(G_girvan, seed=seed)
    # nx.draw(G_girvan, pos, node_color=community_colors, node_size=20, edge_color='gray', with_labels=False, cmap=plt.cm.tab10)
    # plt.show()
    labels1 = [i for i, comm in enumerate(ground_truth) for _ in comm]

    # Convert dictionary to list of tuples
    converted_tuples = tuple([set(i) for i in communities.values()])

    print(converted_tuples)
    labels2 = [i for i, comm in enumerate(converted_tuples) for _ in comm]
    # Calculate NMI
    nmi = normalized_mutual_info_score(labels1, labels2)
    print("NMI = ", nmi)

    !pip install networkx
    !pip install python - igraph
    !sudo apt install libcairo2 - dev pkg - config python3 - dev
    !pip  install   manimlib
    !pip  install   pycairo
    !pip  install   cairocffi

    import time
    start_time = time.time()
    K5 = nx.convert_node_labels_to_integers(G, first_label=2)
    G.add_edges_from(K5.edges())
    c = list(nx.community.k_clique_communities(G, 5))
    print(list(c))

    print("--- %s seconds ---" % (time.time() - start_time))