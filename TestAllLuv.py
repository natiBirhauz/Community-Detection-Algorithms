#runs luv on all the networks
import timeit
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from cdlib import algorithms
import matplotlib.patches as mpatches
from networkx.algorithms.community.quality import modularity
import os

# Function to read connections from a text file
def read_connections(file_path):
    connections = []
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            for line in file:
                source, destination, weight = map(int, line.strip().split(","))
                connections.append((source, destination, weight))
    except ValueError as e:
        print(f"Skipping file {file_path} due to parsing error: {e}")
    return connections

# Function to create a graph from connections
def create_graph(connections):
    G = nx.Graph()
    for source, destination, weight in connections:
        G.add_edge(source, destination, weight=weight)
    return G

# Function to run the Louvain algorithm and calculate metrics
def run_louvain_algorithm(G):
    start_time = timeit.default_timer()
    communities = algorithms.louvain(G)
    elapsed_time = timeit.default_timer() - start_time

    mod_value = modularity(G, communities.communities)
    num_communities = len(communities.communities)
    community_sizes = [len(c) for c in communities.communities]
    community_edges = [G.subgraph(c).number_of_edges() for c in communities.communities]
    community_nodes = [list(c) for c in communities.communities]

    return mod_value, num_communities, community_sizes, community_edges, community_nodes, elapsed_time

# Function to plot the graph with community colors
def plot_graph(G, communities, filename):
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

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_color=[color_map.get(node) for node in G.nodes()], node_size=300)
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

    legend_patches = [mpatches.Patch(color=community_colors[i], label=community_labels[i]) for i in range(len(communities.communities))]
    plt.legend(handles=legend_patches, loc='upper left', fontsize='x-small', title="Detected communities")
    plt.title(f"Louvain Algorithm for {filename}")
    plt.show()

# Main function to process all files in the folder
def process_all_files(data_folder):
    comparison_results = []

    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):  # Only process text files
            file_path = os.path.join(data_folder, filename)
            connections = read_connections(file_path)
            if not connections:  # Skip files that could not be parsed
                continue

            G = create_graph(connections)

            # Run the Louvain algorithm 100 times and calculate the average time
            number_of_runs = 100
            elapsed_times = []
            for _ in range(number_of_runs):
                _, _, _, _, _, elapsed_time = run_louvain_algorithm(G)
                elapsed_times.append(elapsed_time)
            average_time = sum(elapsed_times) / number_of_runs

            # Run the Louvain algorithm once more to get the metrics
            mod_value, num_communities, community_sizes, community_edges, community_nodes, _ = run_louvain_algorithm(G)

            comparison_results.append({
                "file": filename,
                "modularity": mod_value,
                "num_communities": num_communities,
                "community_sizes": community_sizes,
                "community_edges": community_edges,
                "community_nodes": community_nodes,
                "average_time": average_time
            })

            print(f"File: {filename}")
            print(f"Modularity: {mod_value}")
            print(f"Number of communities: {num_communities}")
            print(f"Community sizes: {community_sizes}")
            print(f"Community edges: {community_edges}")
            for idx, nodes in enumerate(community_nodes):
                print(f"Community {idx} nodes: {nodes}")
            print(f"Average time over {number_of_runs} runs: {average_time:.8f} seconds\n")

            plot_graph(G, algorithms.louvain(G), filename)

# Run the processing for all files in the "network data" folder
data_folder = "network data"
process_all_files(data_folder)
