#runs QP on all the networks
import os
import timeit
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import get_named_colors_mapping
import itertools
import pandas as pd
from cdlib import algorithms, evaluation

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

# Step 2: Draw and save community visualization
def draw_communities(G, communities, k, file_name, output_folder):
    pos = nx.spring_layout(G)
    colors = itertools.cycle(get_named_colors_mapping().values())
    node_color_map = {}

    for i, community in enumerate(communities.communities):
        color = next(colors)
        for node in community:
            node_color_map[node] = color
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=color, node_size=200, label=f'Community {i+1}')
    
    non_community_nodes = set(G.nodes()) - set(node_color_map.keys())
    if non_community_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=list(non_community_nodes), node_color='gray', node_size=200, label='No Community', alpha=0.3)
    
    nx.draw_networkx_edges(G, pos, edge_color='gray', alpha=0.3)
    nx.draw_networkx_labels(G, pos)

    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=color) for color in set(node_color_map.values())]
    if non_community_nodes:
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc='gray'))
    legend_labels = [f'Community {i+1}' for i in range(len(communities.communities))]
    if non_community_nodes:
        legend_labels.append('No Community')
    plt.legend(handles=legend_handles, labels=legend_labels, loc='upper left', fontsize='x-small', title="Detected communities")
    plt.title(f"Clique Percolation Algorithm k={k} on {file_name}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, f"{file_name}_k{k}.png"))
    plt.clf()  # Clear the plot for the next graph

# Step 3: Run the Clique Percolation algorithm and measure performance
def run_clique_percolation(G, k):
    return algorithms.kclique(G, k)

# Step 4: Main script to process all files and export results
data_folder = "network data"
output_folder = "results"
result_list = []

for file_name in os.listdir(data_folder):
    if file_name.endswith(".txt"):
        file_path = os.path.join(data_folder, file_name)
        G = read_graph_from_file(file_path)
        
        if len(G.edges()) == 0:
            print(f'The graph in {file_name} has no edges. Skipping.')
            continue
        
        for k in range(2, 6):
            start_time = timeit.default_timer()
            communities = run_clique_percolation(G, k)
            run_time = timeit.default_timer() - start_time
            
            num_communities = len(communities.communities)
            community_sizes = [len(c) for c in communities.communities]
            community_edges = [G.subgraph(c).number_of_edges() for c in communities.communities]
            community_nodes = [G.subgraph(c).number_of_nodes() for c in communities.communities]
            modularity = evaluation.newman_girvan_modularity(G, communities).score
            
            result_list.append({
                "file_name": file_name,
                "clique_percolation_k": f'clique_percolation_{k}',
                "modularity": modularity,
                "num_communities": num_communities,
                "community_sizes": community_sizes,
                "community_edges": community_edges,
                "community_nodes": community_nodes,
                "average_run_time": run_time
            })
            
            draw_communities(G, communities, k, file_name, output_folder)

# Step 5: Export results to XLSX
df = pd.DataFrame(result_list)
df.to_excel(os.path.join(output_folder, "clique_percolation_results.xlsx"), index=False)

print("Processing complete. Results are saved to the 'results' folder.")
