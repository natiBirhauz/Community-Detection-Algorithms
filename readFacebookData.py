import os
import networkx as nx
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np

# Set the relative path to the "facebook100" folder
folder_path = 'facebook100'
file_name = 'NYU9.mat'

# Construct full file path
file_path = os.path.join(folder_path, file_name)

# Check if the file exists
if not os.path.exists(file_path):
    print(f"The file {file_path} does not exist.")
else:
    # Create an empty graph
    G = nx.Graph()

    # Load .mat file
    data = loadmat(file_path)

    # Check if 'A' and 'local_info' are in the keys
    if 'A' in data and 'local_info' in data:
        # Extract the 'A' matrix and 'local_info'
        A_matrix = data['A']
        local_info = data['local_info']

        # Limit to the first 1000 rows
        num_nodes = min(2000, local_info.shape[0])
        A_matrix = A_matrix[:num_nodes, :num_nodes]
        local_info = local_info[:num_nodes]
        num_nodes=local_info.shape[0]
        # Add nodes and edges to the graph
        for i in range(num_nodes):
            # Initialize node attributes dictionary
            node_attrs = {}

            # Add available attributes
            if local_info[i, 0] != 0:  # student_faculty_status
                node_attrs['student_faculty_status'] = int(local_info[i, 0])
            if local_info[i, 1] != 0:  # gender
                node_attrs['gender'] = int(local_info[i, 1])
            if local_info[i, 2] != 0:  # major
                node_attrs['major'] = int(local_info[i, 2])
            if local_info[i, 3] != 0:  # second_major_minor
                node_attrs['second_major_minor'] = int(local_info[i, 3])
            if local_info[i, 4] != 0:  # dorm_house
                node_attrs['dorm_house'] = int(local_info[i, 4])
            if local_info[i, 5] != 0:  # year
                node_attrs['year'] = int(local_info[i, 5])
            if local_info[i, 6] != 0:  # high_school
                node_attrs['high_school'] = int(local_info[i, 6])

            # Add node with attributes
            G.add_node(i, **node_attrs)
            
            # Add edges
            for j in range(i + 1, num_nodes):
                if A_matrix[i, j] != 0:
                    G.add_edge(i, j)

        # Remove nodes without edges
        nodes_to_remove = [node for node in G.nodes if G.degree[node] == 0]
        G.remove_nodes_from(nodes_to_remove)

        # Choose the attribute you want to visualize
        attribute = 'gender'

        # Extract all unique attributes present in the graph
        attributes = set(nx.get_node_attributes(G, attribute).values())
        
        # Define a color map for all unique attributes using 'rainbow' colormap
        colors = plt.cm.rainbow(np.linspace(0, 1, len(attributes)))
        color_map = {attr: colors[i] for i, attr in enumerate(sorted(attributes))}
        
        # Assign colors to nodes based on their attribute
        node_colors = []
        for node in G.nodes:
            if attribute in G.nodes[node]:
                node_colors.append(color_map[G.nodes[node][attribute]])
            else:
                node_colors.append('gray')  # Assign gray color for nodes without 'attribute' attribute

        # Draw the graph
        pos = nx.spring_layout(G)
        plt.figure(figsize=(12, 12))
        nx.draw(G, pos, with_labels=False, node_size=50, node_color=node_colors, edge_color='gray', alpha=0.7)
        
        # Create legend for attributes
        legend_handles = []
        legend_labels = []
        
        for attr, color in color_map.items():
            legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10))
            legend_labels.append(f'{attribute} {attr}')
        
        plt.legend(handles=legend_handles, labels=legend_labels, loc='upper left', fontsize='x-small', title=f"Node {attribute.capitalize()}s")
        
        plt.title('Facebook Network Graph (NYU9)')
        plt.show()
    else:
        print(f"Missing 'A' or 'local_info' keys in {file_name}")
