import csv
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def sample_ppm(memb, p, q):
    n = len(memb)  # Number of vertices
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    for i in range(n):
        for j in range(i+1, n):
            if memb[i] == memb[j]:
                # Nodes belong to the same community
                if np.random.rand() < p:
                    G.add_edge(i, j)
            else:
                # Nodes belong to different communities
                if np.random.rand() < q:
                    G.add_edge(i, j)
    return G

# Example usage:
memb = [i % 4 for i in range(50)]  # 50 vertices, 4 communities
p = 1/3  # Probability of edge within the same community
q = 0.01  # Probability of edge between different communities
G = sample_ppm(memb, p, q)

# Create a colormap for communities
cmap = plt.get_cmap("tab10", max(memb) + 1)

# Visualize the graph
pos = nx.spring_layout(G)

# Draw nodes with community colors
nodes = nx.draw_networkx_nodes(G, pos, node_color=memb, cmap=cmap)

# Draw edges with gray color and alpha 0.5
edges = nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)

# Draw node labels
nx.draw_networkx_labels(G, pos, font_size=10, font_color='black', font_family='sans-serif')

# Create a legend for the communities
community_labels = list(set(memb))  # Unique community labels
community_labels.sort()
legend_points = [plt.Line2D([0], [0], marker='o', color='w', markersize=10, markerfacecolor=cmap(label)) for label in community_labels]
plt.legend(legend_points, community_labels, title='Communities', loc='best')

plt.show()

# Export graph edges and community information to CSV
edges = G.edges()
with open('graph_edges_with_community.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['node1', 'node2', 'community_of_node1'])  # Write header
    for edge in edges:
        node1, node2 = edge
        community_node1 = memb[node1]
        writer.writerow([node1, node2, community_node1])