#test ALL algs on the mat files
import os
import timeit
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.colors import get_named_colors_mapping
import itertools
import pandas as pd
from cdlib import algorithms, evaluation
from sklearn.metrics import normalized_mutual_info_score

# Step 1: Read the graph from the .mat file and filter nodes
def read_and_filter_graph(filename, max_nodes, folder_path='facebook100'):
    file_path = os.path.join(folder_path, filename)
    data = loadmat(file_path)
    
    if 'A' in data and 'local_info' in data:
        A_matrix = data['A']
        local_info = data['local_info']
        
        # Include only the first 'max_nodes' rows and columns
        A_matrix = A_matrix[:max_nodes, :max_nodes]
        local_info = local_info[:max_nodes]
        
        # Create graph from adjacency matrix
        G = nx.Graph()
        
        # Add nodes and edges
        for i in range(max_nodes):
            G.add_node(i, info=local_info[i])  # Add node i with its info
            for j in range(i + 1, max_nodes):
                try:
                    if A_matrix[i, j] != 0:
                        G.add_edge(i, j, weight=A_matrix[i, j])  # Add edge between i and j if weight is non-zero
                except IndexError:
                    print(f"Skipping problematic node/index: ({i}, {j})")
                    continue
        
        # Remove isolated nodes
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        return G, local_info
    else:
        raise ValueError(f"Matrix 'A' or 'local_info' not found in {filename}")

# Step 2: Draw and save community visualization
def draw_communities(G, communities, method, file_name, output_folder):
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
    plt.title(f"{method} on {file_name}")
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(os.path.join(output_folder, f"{file_name}_{method}.png"))
    plt.clf()  # Clear the plot for the next graph

# Step 3: Run the community detection algorithms and measure performance
def run_clique_percolation(G, k):
    return algorithms.kclique(G, k)

def run_louvain(G):
    return algorithms.louvain(G)

def run_girvan_newman(G):
    comp = algorithms.girvan_newman(G,2)
    return comp

# Step 4: Calculate NMI
def calculate_nmi(communities, ground_truth):
    # Create a list of node labels for detected communities
    node_community_map = {}
    for idx, community in enumerate(communities.communities):
        for node in community:
            node_community_map[node] = idx
    
    detected_labels = [node_community_map.get(node, -1) for node in range(len(ground_truth))]
    ground_truth_labels = [info[0] for info in ground_truth]  # Select the first attribute (e.g., student/faculty status)
    
    return normalized_mutual_info_score(ground_truth_labels, detected_labels)

# Step 5: Main script to process all files and export results
data_folder = "facebook100"
output_folder = "results_200nodesFB2"
max_nodes = 200  # Adjust the number of nodes to include
result_list = []

for file_name in os.listdir(data_folder):
    if file_name.endswith(".mat"):
        try:
            G, local_info = read_and_filter_graph(file_name, max_nodes=max_nodes, folder_path=data_folder)
        except ValueError as e:
            print(e)
            continue
        
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
            nmi = calculate_nmi(communities, local_info)
            
            result_list.append({
                "file_name": file_name,
                "algorithm": f"Clique Percolation- {k}",
                "modularity": modularity,
                "num_communities": num_communities,
                "community_sizes": community_sizes,
                "community_edges": community_edges,
                "community_nodes": community_nodes,
                "average_run_time": run_time,
                "nmi": nmi
            })
            
            draw_communities(G, communities, f"Clique_Percolation_k{k}", file_name, output_folder)
        
        # Louvain algorithm
        start_time = timeit.default_timer()
        communities = run_louvain(G)
        run_time = timeit.default_timer() - start_time
        
        num_communities = len(communities.communities)
        community_sizes = [len(c) for c in communities.communities]
        community_edges = [G.subgraph(c).number_of_edges() for c in communities.communities]
        community_nodes = [G.subgraph(c).number_of_nodes() for c in communities.communities]
        modularity = evaluation.newman_girvan_modularity(G, communities).score
        nmi = calculate_nmi(communities, local_info)
        
        result_list.append({
            "file_name": file_name,
            "algorithm": "Louvain",
            "modularity": modularity,
            "num_communities": num_communities,
            "community_sizes": community_sizes,
            "community_edges": community_edges,
            "community_nodes": community_nodes,
            "average_run_time": run_time,
            "nmi": nmi
        })
        
        draw_communities(G, communities, "Louvain", file_name, output_folder)

        # Girvan-Newman algorithm
        start_time = timeit.default_timer()
        communities = run_girvan_newman(G)
        run_time = timeit.default_timer() - start_time
        
        num_communities = len(communities.communities)
        community_sizes = [len(c) for c in communities.communities]
        community_edges = [G.subgraph(c).number_of_edges() for c in communities.communities]
        community_nodes = [G.subgraph(c).number_of_nodes() for c in communities.communities]
        modularity = evaluation.newman_girvan_modularity(G, communities).score
        nmi = calculate_nmi(communities, local_info)
        
        result_list.append({
            "file_name": file_name,
            "algorithm": "Girvan-Newman",
            "modularity": modularity,
            "num_communities": num_communities,
            "community_sizes": community_sizes,
            "community_edges": community_edges,
            "community_nodes": community_nodes,
            "average_run_time": run_time,
            "nmi": nmi
        })
        
        draw_communities(G, communities, "Girvan_Newman", file_name, output_folder)

# Step 6: Export results to XLSX
df = pd.DataFrame(result_list)
df.to_excel(os.path.join(output_folder, "community_detection_results.xlsx"), index=False)

print("Processing complete. Results are saved to the 'results' folder.")
