import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load nodes
nodes_file_path = 'fb-pages-public-figure/fb-pages-public-figure.nodes'
nodes_df = pd.read_csv(nodes_file_path, delimiter=",", skiprows=1, header=None)
nodes_df.columns = ['Node ID', 'Node Name', 'Attribute']

# Convert 'Node ID' to string to ensure matching with graph nodes
nodes_df['Node ID'] = nodes_df['Node ID'].astype(str)

# Load edges
edges_file_path = 'fb-pages-public-figure/fb-pages-public-figure.edges'
edges_df = pd.read_csv(edges_file_path, delimiter=",", skiprows=1, header=None)
edges_df.columns = ['Source', 'Target']

# Convert 'Source' and 'Target' to string to ensure matching with graph nodes
edges_df['Source'] = edges_df['Source'].astype(str)
edges_df['Target'] = edges_df['Target'].astype(str)

# Create a graph
G = nx.Graph()

# Add nodes
for index, row in nodes_df.iterrows():
    G.add_node(row['Node ID'], name=row['Node Name'], attribute=row['Attribute'])

# Add edges
for index, row in edges_df.iterrows():
    G.add_edge(row['Source'], row['Target'])

# Filter nodes with high degree
degree_threshold = 50  # Increase threshold to reduce node count
high_degree_nodes = [n for n, d in G.degree() if d > degree_threshold]
H = G.subgraph(high_degree_nodes)

# Choose layout
pos = nx.kamada_kawai_layout(H)  # Try Kamada-Kawai layout for better separation

# Draw the graph with node size fixed
plt.figure(figsize=(12, 8), facecolor='black')  # Change the background color to black
nodes = nx.draw_networkx_nodes(H, pos, node_size=100,  # Set fixed node size
                               node_color=list(dict(H.degree()).values()), cmap=plt.cm.viridis, alpha=0.9)
edges = nx.draw_networkx_edges(H, pos, alpha=0.5, edge_color='white', width=1.0)  # Increase edge width

# Adding labels to central nodes
central_nodes = [node for node, degree in dict(H.degree()).items() if degree > 100]
labels = {}
for node in central_nodes:
    if not nodes_df[nodes_df['Node ID'] == node].empty:
        labels[node] = nodes_df[nodes_df['Node ID'] == node]['Node Name'].values[0]

nx.draw_networkx_labels(H, pos, labels, font_size=8, font_color='white')

plt.title("Filtered Facebook Pages Public Figure Network", color='white')  # Change title color to white
plt.colorbar(nodes)
plt.show()

