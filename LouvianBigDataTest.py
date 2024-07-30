import os
import networkx as nx
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import normalized_mutual_info_score
from collections import  defaultdict
from cdlib import algorithms, evaluation, NodeClustering

###--------------------------------------------- Part A: Load and Clean Data---------------------------------------- ###
# Load the data from the provided file path
file_path = 'C:/Users/gavri/Desktop/FP/facebook100/facebook100/NYU9.mat'
data = sio.loadmat(file_path)
A = data['A']
local_info = data['local_info']

# Clean the local_info data
clean_local_info = np.where(local_info == 0, np.nan, local_info)
non_missing_indices = ~np.isnan(clean_local_info).all(axis=1)  # remove students with all missing value
clean_local_info = clean_local_info[non_missing_indices]
clean_local_info[:, 5] = np.where(np.isnan(clean_local_info[:, 5]), 0, clean_local_info[:, 5]).astype(int)

# Convert to DataFrame for easier manipulation
columns = ['student_faculty', 'gender', 'major', 'second_major_minor', 'dorm', 'year', 'high_school']
df = pd.DataFrame(clean_local_info, columns=columns)
df.index = np.arange(df.shape[0])

# Filter the DataFrame
df = df[~df['year'].isin([0.0])]
grouped_by_year = df.groupby('year').filter(lambda x: len(x) >= 900).groupby('year')

# Display summary
print("\nData grouped by year:")
print(grouped_by_year.size().reset_index(name='count'))

# Create output directory if it doesn't exist
output_dir = 'C:/Users/gavri/Desktop/FP/facebook100/output'
os.makedirs(output_dir, exist_ok=True)

# Export each year group to CSV
for year, group in grouped_by_year:
    group.to_csv(f'{output_dir}/NYU_{year}.csv', index=False)

# Create and clean the NetworkX graph
G = nx.from_scipy_sparse_array(A)
for i in range(len(G.nodes)):
    if i < len(clean_local_info):
        G.nodes[i].update({
            'student_faculty': clean_local_info[i, 0],
            'gender': clean_local_info[i, 1],
            'major': clean_local_info[i, 2],
            'second_major_minor': clean_local_info[i, 3],
            'dorm': clean_local_info[i, 4],
            'year': clean_local_info[i, 5],
            'high_school': clean_local_info[i, 6]
        })

# Remove nodes with 'year' equal to 0.0 and export the graph
nodes_to_remove = [n for n, d in G.nodes(data=True) if d.get('year') == 0.0]
G.remove_nodes_from(nodes_to_remove)
nx.write_gexf(G, f'{output_dir}/NYU_graph.gexf')
edges_per_year = []
for year in grouped_by_year.groups.keys():
    nodes_in_year = grouped_by_year.get_group(year).index.tolist()
    subgraph = G.subgraph(nodes_in_year)
    edges_per_year.append((year, len(subgraph.edges)))

# Display summary
print("\nData grouped by year:")
year_counts = grouped_by_year.size().reset_index(name='count')
for year, count in zip(year_counts['year'], year_counts['count']):
    edges = next(edge_count for y, edge_count in edges_per_year if y == year)
    print(f"Year: {year}, Nodes: {count}, Edges: {edges}")

# Verify the graph structure after filtering
print(f"Total nodes in the graph after filtering: {len(G.nodes)}")
print(f"Nodes removed with year 0.0: {len(nodes_to_remove)}")
print("Data loading and cleaning complete.")

###------------------------------------------------------------- Part B: Centrality Metrics and Statistical Analysis -----------------------------------------------------###
def calculate_modularity(subgraph, partition, merged_community_id):
    print("Calculating modularity...")
    # If there is a merged community, remove it from the partition
    if (merged_community_id is not None) and (merged_community_id in partition.values()):
        partition = {node: comm for node, comm in partition.items() if comm != merged_community_id}

    # Convert the partition to the format required by cdlib
    communities = defaultdict(list)
    for node, comm in partition.items():
        communities[comm].append(node)
    communities = list(communities.values())

    # Calculate modularity using cdlib
    cdlib_communities = NodeClustering(communities, graph=subgraph, method_name="Louvain")
    modularity_score = evaluation.newman_girvan_modularity(subgraph, cdlib_communities).score
    return modularity_score

def calculate_coverage(G, partition, merged_community_id):
    print("Calculating coverage...")
    # סינון הקהילה המאוחדת
    filtered_partition = {node: comm for node, comm in partition.items() if comm != merged_community_id}
    internal_edges = 0
    total_edges = G.number_of_edges()

    for community in set(filtered_partition.values()):
        community_nodes = [node for node, comm in filtered_partition.items() if comm == community]
        internal_edges += G.subgraph(community_nodes).number_of_edges()

    coverage_score = internal_edges / total_edges if total_edges > 0 else 0
    return coverage_score

def calculate_nmi(partition, true_labels, merged_community_id):
    print("Calculating NMI...")
    filtered_partition = {node: comm for node, comm in partition.items() if comm != merged_community_id}
    filtered_true_labels = {node: true_labels[node] for node in filtered_partition.keys() if node in true_labels}

    true_labels_list = [int(label[0]) if label[0] is not None and not np.isnan(label[0]) else -1 for label in filtered_true_labels.values()]
    partition_labels_list = [partition[node] for node in subgraph.nodes if node in filtered_true_labels and filtered_true_labels[node][0] is not None and not np.isnan(filtered_true_labels[node][0])]

    true_labels_array = np.array(true_labels_list)
    partition_labels_array = np.array(partition_labels_list)

    # הסרת ערכים שליליים
    mask = true_labels_array != -1
    true_labels_array = true_labels_array[mask]
    partition_labels_array = partition_labels_array[mask]

    nmi_score = normalized_mutual_info_score(true_labels_array, partition_labels_array)
    return nmi_score
def calculate_edge_cut_ratio(G, partition, merged_community_id):
    print('Calculate edge cut ratio')
    edge_cut_ratios = []
    for community in set(partition.values()):
        if community == merged_community_id:
            continue
        community_nodes = [node for node, comm in partition.items() if comm == community]

        if len(community_nodes) < 3:
            edge_cut_ratios.append(0)
            continue

        internal_edges = G.subgraph(community_nodes).number_of_edges()
        total_edges = sum(dict(G.degree(community_nodes)).values()) / 2
        edge_cut_ratios.append(1 - (internal_edges / total_edges) if total_edges > 0 else 0)

    return np.mean(edge_cut_ratios)  # Return the average edge cut ratio

def calculate_average_degree(subgraph, partition):
    average_degrees = {}
    for community in set(partition.values()):
        community_nodes = [node for node in partition if partition[node] == community]
        subgraph_community = subgraph.subgraph(community_nodes)
        degrees = dict(subgraph_community.degree())
        average_degrees[community] = np.mean(list(degrees.values()))
    return average_degrees

def calculate_all_metrics(G, partition, true_labels_dict, modularity_scores, nmi_scores_dict, edge_cut_ratios_list, average_degrees_list, coverage_scores, merged_community_id):
    print("Calculating all metrics for a partition...")
    subgraph_nodes = set(partition.keys())
    subgraph = G.subgraph(subgraph_nodes)
    modularity_score = calculate_modularity(subgraph, partition, merged_community_id)
    modularity_scores.append(modularity_score)

    average_degrees = calculate_average_degree(subgraph, partition)
    average_degrees_list.append(np.mean(list(average_degrees.values())))

    coverage_score = calculate_coverage(G, partition, merged_community_id)
    coverage_scores.append(coverage_score)

    for attribute in true_labels_dict:
        if attribute not in nmi_scores_dict:
            nmi_scores_dict[attribute] = []
        true_labels = true_labels_dict[attribute]
        filtered_true_labels = {node: true_labels[node] for node in subgraph_nodes if node in true_labels and true_labels[node][0] != -1}
        if not filtered_true_labels:
            print(f"No true labels available for NMI calculation for attribute {attribute}.")
            nmi_score = 0
        else:
            valid_pairs = [(int(label[0]), partition[node]) for node, label in filtered_true_labels.items() if label[0] is not None and label[0] != -1]

            if not valid_pairs:
                print(f"No valid true labels or partition labels available for NMI calculation for attribute {attribute}.")
                nmi_score = 0
            else:
                true_labels_list, partition_labels_list = zip(*valid_pairs)
                true_labels_array = np.array(true_labels_list)
                partition_labels_array = np.array(partition_labels_list)
                nmi_score = normalized_mutual_info_score(true_labels_array, partition_labels_array)

        nmi_scores_dict[attribute].append(nmi_score)

    edge_cut_ratio = calculate_edge_cut_ratio(G, partition, merged_community_id)
    edge_cut_ratios_list.append(edge_cut_ratio)

    return modularity_score, nmi_scores_dict, edge_cut_ratios_list, coverage_scores
def export_data(run, year, G, partition, modularity_scores, nmi_scores, edge_cut_ratios_list, average_degrees_list, coverage_scores, num_communities_list, output_dir, attribute):
    print(f"Generating report for runs up to {run + 1} for attribute {attribute} for year {year}")

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Calculate the minimum length of the lists
    min_length = min(len(modularity_scores), len(nmi_scores), len(edge_cut_ratios_list), len(num_communities_list), len(average_degrees_list), len(coverage_scores))

    # Save metrics to CSV
    metrics_df = pd.DataFrame({
        'Run': list(range(run - min_length + 1, run + 1)),
        'Modularity': modularity_scores[-min_length:],
        'NMI': nmi_scores[-min_length:],
        'Num Communities': num_communities_list[-min_length:],
        'Edge Cut Ratios': edge_cut_ratios_list[-min_length:],
        'Average Degrees': average_degrees_list[-min_length:],
        'Coverage': coverage_scores[-min_length:]
    })
    metrics_csv_path = os.path.join(output_dir, f'report_for_attribute_{attribute}_for_year_{year}_run_{run + 1}.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)

    # Save final partition to Gephi file
    nx.set_node_attributes(G, partition, 'community')
    gephi_path = os.path.join(output_dir, f'final_partition_{year}_run{run + 1}_louvain.gexf')
    nx.write_gexf(G, gephi_path)

    print(f"Finished report for runs up to {run + 1} for attribute {attribute} for year {year}")

def get_true_labels(df, G, attributes):
    labels = {}
    for node in G.nodes():
        try:
            if node in df.index:
                label = tuple(df.loc[node, attr] for attr in attributes)
                # Handle NaN values
                label = tuple(-1 if pd.isna(value) else value for value in label)
            else:
                print(f"Node {node} not found in DataFrame index.")
                label = (None,) * len(attributes)  # Handle missing labels appropriately
        except KeyError as e:
            print(f"KeyError: {e} for node {node} and attributes {attributes}")
            label = (None,) * len(attributes)  # Handle missing labels appropriately
        labels[node] = label
    return labels

def create_community_graph(year, community_results, original_graph):
    if year not in community_results:
        print(f"Year {year} not found in community_results.")
        return

    community_structure = {}
    for community in set(community_results[year].values()):
        community_structure[community] = [node for node in community_results[year] if community_results[year][node] == community]

    print(f"Creating community graph for year {year}")
    community_graph = nx.Graph()
    for community in community_structure.keys():
        community_graph.add_node(community, label=f"Community {community}",
                                 students=community_structure.get(community, []))
    added_edges = set()
    for community1, nodes1 in community_structure.items():
        for node1 in nodes1:
            if node1 not in original_graph:
                continue
            for neighbor in original_graph.neighbors(node1):
                if neighbor in community_results[year]:
                    community2 = community_results[year][neighbor]
                    if community1 != community2:
                        edge = tuple(sorted([community1, community2]))
                        if edge not in added_edges:
                            community_graph.add_edge(community1, community2)
                            added_edges.add(edge)
    print(f"Finished creating community graph for year {year}")

def save_community_to_excel(df, output_dir, year, community, nodes):
    if len(nodes) >= 20:  # בדוק אם הקהילה מכילה לפחות 20 חברים
        community_df = df.loc[nodes]
        file_path = f'{output_dir}/NYU_{int(year)}_community_{community}.xlsx'
        if os.path.exists(file_path):
            os.remove(file_path)
        community_df.to_excel(file_path, index=False)
        print(f"Saved community {community} for year {year} to Excel")

def merge_small_communities(partition, threshold=3):
    community_counts = defaultdict(int)
    for node, community in partition.items():
        community_counts[community] += 1

    small_communities = [community for community, count in community_counts.items() if count < threshold]
    new_community_id = max(partition.values()) + 1
    merged_community_id = new_community_id
    for node, community in partition.items():
        if community in small_communities:
            partition[node] = new_community_id

    # מיפוי מחדש של החלוקה לאחר איחוד הקהילות הקטנות
    unique_communities = sorted(set(partition.values()))
    community_mapping = {old: new for new, old in enumerate(unique_communities, 1)}
    remapped_partition = {node: community_mapping[community] for node, community in partition.items()}

    return remapped_partition, merged_community_id

###------------------------------------------------------------- Part C: Community Detection and Export -----------------------------------------------------###
# הגדרת משתני אחסון לתוצאות המדדים
community_results = {}
num_runs = 5  # Number of times to run the Louvain algorithm
attributes = ['gender', 'major', 'dorm', 'high_school', 'second_major_minor']
output_dir = 'C:/Users/gavri/Desktop/FP/facebook100/output'

print("Starting community detection and export...")

# Main loop: Process each year
for year, group in grouped_by_year:
    print(f"Processing year: {year}")
    nodes = group.index
    subgraph = G.subgraph(nodes)
    print(f"Subgraph for year {year} created with {subgraph.number_of_nodes()} nodes.")

    # Filter the DataFrame for the current year
    df_year = df.loc[nodes]
    print(f"DataFrame columns for year {year}: {df_year.columns}")
    print(f"First 5 rows of DataFrame for year {year}:")
    print(df_year.head())

    for attribute in attributes:
        print(f"Processing attribute: {attribute}")
        if attribute not in df_year.columns:
            print(f"Attribute {attribute} not found in DataFrame columns for year {year}.")
            continue
        true_labels = get_true_labels(df_year, subgraph, [attribute])
        print(f"True labels for {attribute}: {list(true_labels.values())[:10]}")  # Debugging: print first 10 true labels
        true_labels_dict = {attribute: true_labels}

        # Reset metrics for each attribute and year
        modularity_scores = []
        nmi_scores_dict = {attribute: []}
        num_communities_list = []
        edge_cut_ratios_list = []
        average_degrees_list = []
        coverage_scores = []

        partitions = []  # To store partitions from multiple runs
        remapped_partition = {}  # Initialize remapped_partition
        merged_community_id = None  # Initialize merged_community_id
        community_structure = {}  # Reset community_structure for each attribute

        # Run the Louvain algorithm multiple times
        for run in range(num_runs):
            print(f"Run {run + 1} for attribute {attribute} for year {year}")
            partition_result = algorithms.louvain(subgraph)  # Run the algorithm on the subgraph
            partition = {node: community for community, nodes in enumerate(partition_result.communities) for node in nodes}
            remapped_partition, merged_community_id = merge_small_communities(partition, threshold=3)  # איחוד קהילות קטנות ומיפוי מחדש
            partitions.append(remapped_partition)  # Store the resulting partition

            # Calculate metrics for the current partition
            print("Calculating all metrics for a partition...")
            calculate_all_metrics(G, remapped_partition, true_labels_dict, modularity_scores, nmi_scores_dict, edge_cut_ratios_list, average_degrees_list, coverage_scores, merged_community_id)

            # Calculate the number of communities for the current partition
            num_communities = len(set(remapped_partition.values()))
            num_communities_list.append(num_communities)

            # Generate report every 5 runs
            if (run + 1) % 5 == 0:
                # Combine the results of the partitions
                after_five_partition = {}
                for node in subgraph.nodes():
                    communities = [remapped_partition[node] for remapped_partition in partitions]
                    after_five_partition[node] = max(set(communities), key=communities.count)

                export_data(run, year, subgraph, after_five_partition, modularity_scores, nmi_scores_dict[attribute], edge_cut_ratios_list, average_degrees_list, coverage_scores, num_communities_list, output_dir, attribute)

        # Final report for all runs
        print(f"Generating final report for all {num_runs} runs for attribute {attribute} for year {year}")

        # Combine the results of the partitions to the final one
        final_partition = {}
        for node in subgraph.nodes():
            communities = [remapped_partition[node] for remapped_partition in partitions]
            final_partition[node] = max(set(communities), key=communities.count)

        export_data(num_runs, year, subgraph, final_partition, modularity_scores, nmi_scores_dict[attribute], edge_cut_ratios_list, average_degrees_list, coverage_scores, num_communities_list, output_dir, attribute)

        print(f"Finished final report for attribute {attribute} for year {year}")

        community_results[(year, attribute)] = remapped_partition

        # Export each community to an Excel file if it has at least 20 members
        for community, nodes in community_structure.items():
            save_community_to_excel(df, output_dir, year, community, nodes)

        # Print debug message after adding the results to community_results
    print(f"Finished community detection for year {year}")
print("Community detection and export complete.")
