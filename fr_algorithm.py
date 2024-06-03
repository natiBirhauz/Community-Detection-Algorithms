import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Read connections from a text file
connections = []
with open("graph.txt", "r") as file:
    for line in file:
        source, destination, weight = map(int, line.strip().split(","))
        connections.append((source, destination, weight))

# Create a directed graph
G = nx.DiGraph()

# Add edges with weights
for source, destination, weight in connections:
    G.add_edge(source, destination, weight=weight)

# Define communities
communities = [{0}, {1, 65, 56, 68, 66, 45, 17, 54, 22, 24, 57, 26, 30, 31},
               {14}, {9, 11, 21, 25, 28, 29, 32, 37, 41, 43, 44, 51, 53, 55, 58, 59, 61, 63, 70, 72},
               {48, 50}, {52}, {2}, {3}, {34, 4, 5}, {19, 7}, {16}, {18}, {20}, {27}, {35}, {39}, {42}, 
               {47}, {49}, {62}, {67}, {69}, {71}, {73}, {6}, {12}, {13}, {15}, {33}, {38}, {40}, {60}, 
               {64}, {8}, {10}, {23}, {36}, {46}, {74}]

# Create a color map for nodes based on communities
color_map = {}
community_colors = []
community_labels = []

for i, community in enumerate(communities):
    color = plt.cm.jet(i / len(communities))  # Assigning a unique color to each community
    community_colors.append(color)
    community_labels.append(f"Community {i}")
    for node in community:
        color_map[node] = color

# Draw the graph
pos = nx.spring_layout(G)  # Positions for all nodes
edge_labels = {(source, destination): weight['weight'] for source, destination, weight in G.edges(data=True)}
nx.draw(G, pos, with_labels=True, node_size=300, node_color=[color_map[node] for node in G.nodes()], font_size=7, width=0.3)

# Create legend
legend_patches = [mpatches.Patch(color=community_colors[i], label=community_labels[i]) for i in range(len(communities))]
plt.legend(handles=legend_patches, loc='upper left')

plt.title("Connection Web Visualization")
plt.show()
