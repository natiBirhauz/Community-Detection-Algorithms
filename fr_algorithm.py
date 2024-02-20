import networkx as nx
import matplotlib.pyplot as plt
#Position nodes using Fruchterman-Reingold force-directed algorithm.
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


# Draw the graph
pos = nx.spring_layout(G)  # Positions for all nodes
edge_labels = {(source, destination): weight['weight'] for source, destination, weight in G.edges(data=True)}
nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_weight="bold")

plt.title("Connection Web Visualization")
plt.show()
