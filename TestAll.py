
import itertools
import timeit
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from cdlib import algorithms, evaluation
import matplotlib.patches as mpatches
from networkx.algorithms.community.quality import modularity
import os
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
from matplotlib.colors import get_named_colors_mapping

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

# Function to run Louvain algorithm and calculate metrics
def run_louvain(G):
    start_time = timeit.default_timer()
    communities = algorithms.louvain(G)
    elapsed_time = timeit.default_timer() - start_time

    if not communities.communities:
        raise ValueError("No communities detected, invalid partition")

    mod_value = modularity(G, communities.communities)
    num_communities = len(communities.communities)
    community_sizes = [len(c) for c in communities.communities]
    community_edges = [G.subgraph(c).number_of_edges() for c in communities.communities]
    community_nodes = [list(c) for c in communities.communities]

    return mod_value, num_communities, community_sizes, community_edges, community_nodes, elapsed_time

# Function to convert overlapping communities to non-overlapping for modularity calculation
def convert_to_non_overlapping(communities):
    unique_nodes = set(itertools.chain(*communities))
    non_overlapping_communities = []
    node_to_community = {node: None for node in unique_nodes}
    
    for idx, community in enumerate(communities):
        for node in community:
            if node_to_community[node] is None:
                node_to_community[node] = idx
            else:
                node_to_community[node] = -1  # Mark as overlapping
    
    for idx, community in enumerate(communities):
        non_overlapping_community = [node for node in community if node_to_community[node] == idx]
        if non_overlapping_community:
            non_overlapping_communities.append(non_overlapping_community)
    
    return non_overlapping_communities

# Step 3: Run the Clique Percolation algorithm
def run_clique_percolation(G, k):
    start_time = timeit.default_timer()
    communities = algorithms.kclique(G, k)
    elapsed_time = timeit.default_timer() - start_time

    # Step 4: Perform community detection
    non_overlapping_communities = convert_to_non_overlapping(communities.communities)
    mod_value = modularity(G, non_overlapping_communities)
    num_communities = len(non_overlapping_communities)
    community_sizes = [len(c) for c in non_overlapping_communities]
    community_edges = [G.subgraph(c).number_of_edges() for c in non_overlapping_communities]
    community_nodes = [list(c) for c in non_overlapping_communities]

    return mod_value, num_communities, community_sizes, community_edges, community_nodes, elapsed_time

# Function to run Girvan-Newman algorithm and calculate metrics
def run_girvan_newman(G, level):
    start_time = timeit.default_timer()
    communities_generator = nx.algorithms.community.centrality.girvan_newman(G)
    result = None
    for i in range(level):
        try:
            result = next(communities_generator)
        except StopIteration:
            break
    elapsed_time = timeit.default_timer() - start_time

    if result is None:
        raise ValueError("No communities detected, invalid partition")

    community_list = [list(c) for c in result]
    mod_value = modularity(G, community_list)
    num_communities = len(community_list)
    community_sizes = [len(c) for c in community_list]
    community_edges = [G.subgraph(c).number_of_edges() for c in community_list]
    community_nodes = community_list

    return mod_value, num_communities, community_sizes, community_edges, community_nodes, elapsed_time

# Function to calculate conductance for each community
def calculate_conductance(G, community):
    cut_size = nx.cut_size(G, community)
    volume = sum(d for n, d in G.degree(community))
    return cut_size / volume

# Function to measure variance in community structures across multiple runs
def measure_stability(community_nodes_runs):
    stability_score = 0
    num_runs = len(community_nodes_runs)
    for i in range(num_runs):
        for j in range(i + 1, num_runs):
            nmi = normalized_mutual_info_score(
                [item for sublist in community_nodes_runs[i] for item in sublist],
                [item for sublist in community_nodes_runs[j] for item in sublist]
            )
            stability_score += nmi
    stability_score /= (num_runs * (num_runs - 1) / 2)
    return stability_score

# Function to compare detected communities with ground truth using NMI
def compare_with_ground_truth(detected_communities, ground_truth):
    flattened_detected = [item for sublist in detected_communities for item in sublist]
    flattened_ground_truth = [item for sublist in ground_truth for item in sublist]
    return normalized_mutual_info_score(flattened_ground_truth, flattened_detected)

# Function to plot the graph with community colors
def plot_graph(G, community_nodes, filename, title, save_path):
    color_map = {}
    community_colors = []
    community_labels = []
    colors = plt.cm.rainbow(np.linspace(0, 1, len(community_nodes)))
    
    for idx, community in enumerate(community_nodes):
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

    legend_patches = [mpatches.Patch(color=community_colors[i], label=community_labels[i]) for i in range(len(community_nodes))]
    plt.legend(handles=legend_patches, loc='upper left', fontsize='x-small', title="Detected communities")

    # Nodes that do not belong to any community
    non_community_nodes = set(G.nodes()) - set(color_map.keys())
    if non_community_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=list(non_community_nodes), node_color='gray', node_size=200, label='No Community', alpha=0.3)
        legend_patches.append(mpatches.Patch(color='gray', label='No Community'))

    plt.title(f"{title} for {filename}")
    plt.savefig(os.path.join(save_path, f"{filename}_{title.replace(' ', '_')}.png"))
    plt.close()

# Function to process all files in the folder and compare algorithms
def process_all_files(data_folder, num_runs, ground_truth=None):
    comparison_results = []

    # Create folder for saving results if it doesn't exist
    results_folder = "TestAllResults"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for filename in os.listdir(data_folder):
        if filename.endswith('.txt'):  # Only process text files
            file_path = os.path.join(data_folder, filename)
            connections = read_connections(file_path)
            if not connections:  # Skip files that could not be parsed
                continue

            G = create_graph(connections)
            
            # Remove isolated nodes
            G.remove_nodes_from(list(nx.isolates(G)))

            # Print graph information for debugging
            print(f'Number of nodes after filtering: {G.number_of_nodes()}')
            print(f'Number of edges after filtering: {G.number_of_edges()}')

            # Run each algorithm and store results
            algorithms_info = [
                ("Louvain", run_louvain),
                ("Girvan-Newman", run_girvan_newman, 2)  # Example with level=2
            ]

            clique_percolation_ks = [2, 3, 4, 5]
            for k in clique_percolation_ks:
                algorithms_info.append((f"Clique Percolation (k={k})", run_clique_percolation, k))

            for algo_info in algorithms_info:
                algo_name = algo_info[0]
                algo_func = algo_info[1]
                algo_args = algo_info[2:] if len(algo_info) > 2 else []

                avg_time = 0
                community_nodes_runs = []
                mod_value = None
                for _ in range(num_runs):
                    try:
                        mod_value, num_communities, community_sizes, community_edges, community_nodes, elapsed_time = algo_func(G, *algo_args)
                        avg_time += elapsed_time
                        community_nodes_runs.append(community_nodes)
                    except ValueError as e:
                        print(f"Error in {algo_name} for file {filename}: {e}")
                        break
                    except TypeError as e:
                        print(f"TypeError in {algo_name} for file {filename}: {e}")
                        break
                    except nx.exception.NetworkXError as e:
                        print(f"NetworkXError in {algo_name} for file {filename}: {e}")
                        break
                
                if mod_value is not None:
                    avg_time /= num_runs

                    # Calculate conductance for each community
                    conductance_values = [calculate_conductance(G, community) for community in community_nodes]

                    # Measure stability across multiple runs
                    stability_score = measure_stability(community_nodes_runs)

                    # Compare with ground truth if provided
                    nmi_score = None
                    if ground_truth:
                        nmi_score = compare_with_ground_truth(community_nodes, ground_truth)

                    comparison_results.append({
                        "file": filename,
                        "algorithm": algo_name,
                        "modularity": mod_value,
                        "num_communities": num_communities,
                        "community_sizes": community_sizes,
                        "community_edges": community_edges,
                        "community_nodes": community_nodes,
                        "average_time": avg_time,
                        "conductance": conductance_values,
                        "stability": stability_score,
                        "nmi_with_ground_truth": nmi_score
                    })

                    print(f"File: {filename}, Algorithm: {algo_name}")
                    print(f"Modularity: {mod_value}")
                    print(f"Number of communities: {num_communities}")
                    print(f"Community sizes: {community_sizes}")
                    print(f"Community edges: {community_edges}")
                    for idx, nodes in enumerate(community_nodes):
                        print(f"Community {idx} nodes: {nodes}")
                    print(f"Average time over {num_runs} runs: {avg_time:.8f} seconds")
                    print(f"Conductance values: {conductance_values}")
                    print(f"Stability score: {stability_score}")
                    if nmi_score is not None:
                        print(f"NMI with ground truth: {nmi_score}")
                    print("\n")

                    plot_graph(G, community_nodes=community_nodes, filename=filename, title=algo_name, save_path=results_folder)

    # Export results to an Excel file
    df = pd.DataFrame(comparison_results)
    df.to_excel('comparison_results5.xlsx', index=False)
    print("Results have been exported to 'comparison_results.xlsx'")

# Run the processing for all files in the "network data" folder with a specified number of runs
data_folder = "network data"
num_runs = 20  # You can change this to the desired number of runs
ground_truth = None  # Replace with actual ground truth data if available
process_all_files(data_folder, num_runs, ground_truth)
print("finished!'")

