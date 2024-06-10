import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from cdlib import algorithms
import matplotlib.patches as mpatches

# Read connections from a text file
connections = []
with open("social_network100.txt", "r") as file:
    for line in file:
        source, destination, weight = map(int, line.strip().split(","))
        connections.append((source, destination, weight))

# Create a graph
G = nx.Graph()

# Add edges with weights
for source, destination, weight in connections:
    G.add_edge(source, destination, weight=weight)

# Apply Louvain Algorithm
communities = algorithms.louvain(G)

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

# Draw the graph with node colors based on communities
pos = nx.spring_layout(G)  # Layout for the graph
plt.figure(figsize=(10, 7))

# Draw nodes with colors based on their community
nx.draw_networkx_nodes(G, pos, node_color=[color_map.get(node, "blue") for node in G.nodes()], node_size=300)

# Draw edges with weights
nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# Create legend
legend_patches = [mpatches.Patch(color=community_colors[i], label=community_labels[i]) for i in range(len(communities.communities))]
plt.legend(handles=legend_patches, loc='upper left', fontsize='x-small', title="Detected communities")

# Display the plot
plt.title("Louvain Algorithm")
plt.show()
