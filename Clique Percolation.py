import os
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import get_named_colors_mapping
import itertools
import timeit
from cdlib import algorithms

# Step 1: Read the graph from the text file and filter nodes
def read_graph_from_file(file_path):
    connections = []
    with open(file_path, "r") as file:
        for line in file:
            source, destination, weight = map(int, line.strip().split(","))
            connections.append((source, destination, weight))

    # Create a graph
    G = nx.Graph()
    # Add edges with weights
    for source, destination, weight in connections:
        G.add_edge(source, destination, weight=weight)

    # Remove isolated nodes
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    return G

data_folder = "network data"
file_path = os.path.join(data_folder, "CommunityNetwork_200N4566E3C.txt")

# Step 2: Create the graph using NetworkX
G = read_graph_from_file(file_path)
k = 3

# Print graph information for debugging
print(f'Number of nodes after filtering: {G.number_of_nodes()}')
print(f'Number of edges after filtering: {G.number_of_edges()}')

# Step 3: Run the Clique Percolation algorithm
def run_clique_percolation(G):
    return algorithms.kclique(G, k)

# Measure the average time taken to run the Clique Percolation algorithm
number_of_runs = 1
avg_time = timeit.timeit(lambda: run_clique_percolation(G), number=number_of_runs) / number_of_runs
print(f'Average time over {number_of_runs} runs: {avg_time:.8f} seconds')

# Step 4: Perform community detection
if len(G.edges()) > 0:
    communities = run_clique_percolation(G)

    num_communities = len(communities.communities)
    print(f'Number of communities: {num_communities}')

    # Step 5: Display the graph with the communities
    def draw_communities(G, communities):
        pos = nx.spring_layout(G)
        # Generate a unique color for each community
        colors = itertools.cycle(get_named_colors_mapping().values())
        color_map = {}
        node_color_map = {}

        for i, community in enumerate(communities.communities):
            color = next(colors)
            for node in community:
                node_color_map[node] = color
            nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=color, node_size=200, label=f'Community {i+1}')
        
        # Nodes that do not belong to any community
        non_community_nodes = set(G.nodes()) - set(node_color_map.keys())
        if non_community_nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=list(non_community_nodes), node_color='gray', node_size=200, label='No Community', alpha=0.3)
        
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
        nx.draw_networkx_labels(G, pos)

        # Custom legend with rectangles
        legend_handles = []
        for i, color in enumerate(set(node_color_map.values())):
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color))  # Create a rectangle patch
        if non_community_nodes:
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc='gray'))
        legend_labels = [f'Community {i+1}' for i in range(num_communities)]
        if non_community_nodes:
            legend_labels.append('No Community')
        plt.legend(handles=legend_handles, labels=legend_labels, loc='upper left', fontsize='x-small', title="Detected communities")
        plt.title(f"Clique Percolation Algorithm k={k}")
        plt.show()

        # Print community information
        for i, community in enumerate(communities.communities):
            print(f'Community {i+1}: {community}')
        if non_community_nodes:
            print(f'No Community: {list(non_community_nodes)}')

    draw_communities(G, communities)
else:
    print('The graph has no edges. Unable to compute communities.')
