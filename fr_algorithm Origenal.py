import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random

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


# Draw the graph
pos = nx.spring_layout(G)  # Positions for all nodes
edge_labels = {(source, destination): weight['weight'] for source, destination, weight in G.edges(data=True)}
nx.draw(G, pos, with_labels=True, node_size=200, font_size=7, width=0.3)




plt.title("Connection Web Visualization")
plt.show()
