import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
# Fruchterman-Reingold algorithm
# Read connections from a text file
connections = []
with open("social_network100.txt", "r") as file:
    for line in file:
        source, destination, weight = map(int, line.strip().split(","))
        connections.append((source, destination, weight))

# Create a directed graph
G = nx.DiGraph()

# Add edges with weights
for source, destination, weight in connections:
    G.add_edge(source, destination, weight=weight)

# Define communities
communities = [{8, 18, 13}, {9, 12, 5}, {1, 2, 20, 6}, {3, 4, 7, 10, 11, 14, 15, 16, 17, 19}]

# Create a color map for nodes based on communities
color_map = {}
community_colors = []
community_labels = []

def get_random_color():
    return "#{:06x}".format(random.randint(0, 0xFFFFFF))

for i, community in enumerate(communities):
    color = get_random_color()  # Assigning a random color to each community
    community_colors.append(color)
    community_nodes = ", ".join(map(str, community))
    community_labels.append(f"Community {i}: {community_nodes}")
    for node in community:
        color_map[node] = color

# Draw the graph
pos = nx.spring_layout(G)  # Positions for all nodes
edge_labels = {(source, destination): weight['weight'] for source, destination, weight in G.edges(data=True)}
nx.draw(G, pos, with_labels=True, node_size=200, node_color=[color_map.get(node, "blue") for node in G.nodes()], font_size=7, width=0.3)

# Create legend
legend_patches = [mpatches.Patch(color=community_colors[i], label=community_labels[i]) for i in range(len(communities))]
plt.legend(handles=legend_patches, loc='upper left', fontsize='x-small', title="Communities")



plt.title("Connection Web Visualization")
plt.show()
