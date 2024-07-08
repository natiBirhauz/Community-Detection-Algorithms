import os
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community.quality import modularity
from matplotlib.colors import get_named_colors_mapping
import itertools
import timeit

# Step 1: Read the graph from the input file with comma-separated values
def read_graph(filename):
    G = nx.Graph()
    data_folder = "network data"
    file_path = os.path.join(data_folder, filename)
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 3:
                u, v, w = parts
                G.add_edge(u, v, weight=float(w))
    return G

# Step 2: Create the graph using NetworkX
filename = "randomNetwork_200N4430E.txt"
G = read_graph(filename)

# Print graph information for debugging
print(f'Number of nodes: {G.number_of_nodes()}')
print(f'Number of edges: {G.number_of_edges()}')

# Step 3: Run the Girvan-Newman algorithm
def run_girvan_newman(G):
    communities_generator = girvan_newman(G)
    top_level_communities = next(communities_generator)
    try:
        next_level_communities = next(communities_generator)
    except StopIteration:
        next_level_communities = top_level_communities
    return sorted(map(sorted, next_level_communities))

# Measure the average time taken to run the Girvan-Newman algorithm
number_of_runs = 1000
avg_time = timeit.timeit(lambda: run_girvan_newman(G), number=number_of_runs) / number_of_runs
print(f'Average time over {number_of_runs} runs: {avg_time:.8f} seconds')

if len(G.edges()) > 0:
    communities = run_girvan_newman(G)

    # Step 4: Compute the modularity and number of communities
    mod = modularity(G, communities)
    num_communities = len(communities)
    print(f'Modularity: {mod}')
    print(f'Number of communities: {num_communities}')

    # Step 5: Display the graph with the communities
    def draw_communities(G, communities):
        pos = nx.spring_layout(G)
        # Generate a unique color for each community
        colors = itertools.cycle(get_named_colors_mapping().values())
        color_map = {}

        for i, community in enumerate(communities):
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
        plt.title(" Girvan-Newman algorithm")
        plt.show()

        # Print community information
        for i, community in enumerate(communities):
            print(f'Community {i+1}: {community}')

    draw_communities(G, communities)
else:
    print('The graph has no edges. Unable to compute modularity or communities.')
