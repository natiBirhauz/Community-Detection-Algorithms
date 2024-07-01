# Update the function to read the file from the specified folder
import os

# Adjust the file path to include the folder "network data"
data_folder = "network data"
file_path = os.path.join(data_folder, "social_network5000.txt")
    
def sort_and_deduplicate_network_file(input_file, output_file):
    def remove_double_connections(edges):
        seen = set()
        unique_edges = []
        
        for source, dest, weight in edges:
            if (source, dest) not in seen and (dest, source) not in seen:  # Check both directions
                seen.add((source, dest))
                seen.add((dest, source))  # Mark both directions as seen
                unique_edges.append((source, dest, weight))
        
        return unique_edges
    

    # Read the file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Parse the lines into tuples of (source, dest, weight)
    edges = []
    for line in lines:
        source, dest, weight = line.strip().split(',')
        edges.append((int(source), int(dest), int(weight)))
    
    # Remove double connections
    edges = remove_double_connections(edges)
    
    # Sort the edges by source and then by destination
    sorted_edges = sorted(edges, key=lambda edge: (edge[0], edge[1]))
    
    # Write the sorted edges to the output file
    with open(output_file, 'w') as file:
        for edge in sorted_edges:
            file.write(f"{edge[0]},{edge[1]},{edge[2]}\n")

# Usage with the specified input and output files
sort_and_deduplicate_network_file(file_path, 'randomNetwork2_20N75E.txt')
