import pandas as pd
import numpy as np

# Load the community network files
network_files = [
    "network_data/CommunityNetwork_200N4566E3C.txt",
    "network_data/CommunityNetwork_20N134E3C(SC).txt",
    "network_data/CommunityNetwork_100N174E3C(LC).txt",
    "network_data/CommunityNetwork_20N76E3C.txt"
]

# Load the results file
results_file = "network_data/Resaults.xlsx"

# Function to read community network files
def read_network_file(file_path):
    return pd.read_csv(file_path, sep='\t')

# Function to read results file
def read_results_file(file_path):
    return pd.read_excel(file_path)

# Load the results data
results_data = read_results_file(results_file)

# Compare communities
def compare_communities(network_df, results_df):
    comparisons = []
    for index, row in results_df.iterrows():
        file_name = row['file name']
        algorithm = row['algorithm']
        algorithm_communities = eval(row['community_nodes'])

        network_file = [file for file in network_files if file_name in file]
        if not network_file:
            continue
        network_file = network_file[0]
        network_df = read_network_file(network_file)
        
        actual_communities = network_df.groupby('community_of_node1')['node1'].apply(list).values.tolist()

        comparison_result = {
            "file_name": file_name,
            "algorithm": algorithm,
            "actual_communities": actual_communities,
            "algorithm_communities": algorithm_communities
        }
        
        comparisons.append(comparison_result)
    
    return comparisons

# Run the comparison
network_df = None
community_comparison = compare_communities(network_df, results_data)

# Print the comparison results
for comparison in community_comparison:
    print(f"File: {comparison['file_name']}, Algorithm: {comparison['algorithm']}")
    print("Actual Communities:", comparison['actual_communities'])
    print("Algorithm Communities:", comparison['algorithm_communities'])
    print()

# Save the comparison results to a file if needed
comparison_df = pd.DataFrame(community_comparison)
comparison_df.to_excel("network_data/comparison_results.xlsx", index=False)
