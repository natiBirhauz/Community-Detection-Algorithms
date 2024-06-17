import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import matplotlib.patches as mpatches
from cdlib import algorithms

# Read connections from a text file
connections = []
with open("sorted_network_by_source.txt", "r") as file:
    for line in file:
        source, destination, weight = map(int, line.strip().split(","))
        connections.append((source, destination, weight))

# Create a directed graph
G = nx.DiGraph()

# Add edges with weights
for source, destination, weight in connections:
    G.add_edge(source, destination, weight=weight)

# Convert directed graph to undirected graph for Clique Percolation Method
G_undirected = G.to_undirected()

# Define the size of the cliques (k) for the Clique Percolation Method
k = 4  # Using 3-cliques as an example

# Detect communities using Clique Percolation Method
communities = algorithms.kclique(G_undirected, k)

# Print detected communities
print("Detected communities:")
for i, community in enumerate(communities.communities):
    print(f"Community {i}: {sorted(community)}")

# Check if any nodes are missing from the communities
all_nodes_in_communities = set().union(*communities.communities)
missing_nodes = set(G.nodes) - all_nodes_in_communities
if missing_nodes:
    print(f"Nodes not in any community: {sorted(missing_nodes)}")

# Create a color map for nodes based on communities
color_map = {}
community_colors = []
community_labels = []
colors = plt.cm.rainbow(np.linspace(0, 1, len(communities.communities)))
for idx, community in enumerate(communities.communities):
    color = colors[idx]
    community_colors.append(color)
    community_labels.append(f"Community {idx}: {sorted(community)}")
    for node in community:
        color_map[node] = color

# If there are nodes not in any community, assign them a default color
default_color = "#000000"  # Black for nodes not in any community
for node in missing_nodes:
    color_map[node] = default_color

# Draw the graph with node colors based on communities
pos = nx.spring_layout(G_undirected)  # Layout for the graph
plt.figure(figsize=(10, 7))

# Draw nodes with colors based on their community
nx.draw_networkx_nodes(G_undirected, pos, node_color=[color_map.get(node, "blue") for node in G_undirected.nodes()], node_size=300)

# Draw edges with weights
nx.draw_networkx_edges(G_undirected, pos, edge_color='gray', alpha=0.5)
nx.draw_networkx_labels(G_undirected, pos, font_size=12, font_color='black')

# Draw edge weights
edge_labels = {(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)}
#nx.draw_networkx_edge_labels(G_undirected, pos, edge_labels=edge_labels, font_size=8, font_color='red')

# Create legend
legend_patches = [mpatches.Patch(color=community_colors[i], label=community_labels[i]) for i in range(len(communities.communities))]
if missing_nodes:
    legend_patches.append(mpatches.Patch(color=default_color, label=f"Not in any community: {sorted(missing_nodes)}"))
plt.legend(handles=legend_patches, loc='upper left', fontsize='x-small', title="Communities")

# Display the plot
plt.title(f"Clique Percolation Method (k={k})")
plt.show()
