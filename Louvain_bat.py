import os
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import loadmat
from networkx.algorithms.community.quality import modularity
from matplotlib.colors import get_named_colors_mapping
import itertools
import timeit
from cdlib import algorithms

# Step 1: Read the graph from the .mat file and filter nodes
def read_and_filter_graph(filename, max_nodes, folder_path='facebook100'):
    file_path = os.path.join(folder_path, filename)
    data = loadmat(file_path)
    
    if 'A' in data:
        A_matrix = data['A']
        # Include only the first 'max_nodes' rows and columns
        A_matrix = A_matrix[:max_nodes, :max_nodes]
        
        # Create graph from adjacency matrix
        G = nx.Graph()
        
        # Add nodes and edges
        for i in range(max_nodes):
            G.add_node(i)  # Add node i
            for j in range(i + 1, max_nodes):
                if A_matrix[i, j] != 0:
                    G.add_edge(i, j, weight=A_matrix[i, j])  # Add edge between i and j if weight is non-zero
        
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        return G
    else:
        raise ValueError(f"Matrix 'A' not found in {filename}")

# Step 2: Create the graph using NetworkX
filename = 'NYU9.mat'
G = read_and_filter_graph(filename, max_nodes=2200)

# Print graph information for debugging
print(f'Number of nodes after filtering: {G.number_of_nodes()}')
print(f'Number of edges after filtering: {G.number_of_edges()}')

# Step 3: Run the Louvain algorithm
def run_louvain(G):
    return algorithms.louvain(G)

# Measure the average time taken to run the Louvain algorithm
number_of_runs = 1
avg_time = timeit.timeit(lambda: run_louvain(G), number=number_of_runs) / number_of_runs
print(f'Average time over {number_of_runs} runs: {avg_time:.8f} seconds')

# Step 4: Perform community detection and compute modularity
if len(G.edges()) > 0:
    communities = run_louvain(G)

    # Compute modularity
    mod = modularity(G, communities.communities)
    num_communities = len(communities.communities)
    print(f'Modularity: {mod}')
    print(f'Number of communities: {num_communities}')

    # Step 5: Display the graph with the communities
    def draw_communities(G, communities):
        pos = nx.spring_layout(G)
        # Generate a unique color for each community
        colors = itertools.cycle(get_named_colors_mapping().values())
        color_map = {}

        for i, community in enumerate(communities.communities):
            color = next(colors)
            color_map[i] = color
            nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=color, node_size=200, label=f'Community {i+1}')
        
        nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
        nx.draw_networkx_labels(G, pos)

        # Custom legend with rectangles
        legend_handles = []
        for i, color in color_map.items():
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=color))  # Create a rectangle patch
        plt.legend(handles=legend_handles, labels=[f'Community {i+1}' for i in color_map.keys()], loc='upper left', fontsize='x-small', title="Detected communities")
        plt.title("Louvain Algorithm")
        plt.show()

        # Print community information
        for i, community in enumerate(communities.communities):
            print(f'Community {i+1}: {community}')

    draw_communities(G, communities)
else:
    print('The graph has no edges. Unable to compute modularity or communities.')
