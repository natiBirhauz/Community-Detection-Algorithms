def sort_and_deduplicate_network_file(input_file, output_file):
    def remove_double_connections(edges):
        seen = set()
        unique_edges = []
        
        for source, dest, weight in edges:
            if (source, dest) not in seen:
                seen.add((source, dest))
                unique_edges.append((source, dest, weight))
        
        return unique_edges
    
    # Read the file
    with open(input_file, 'r') as file:
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

# Usage
sort_and_deduplicate_network_file('sorted_network100.txt', 'sorted_network_by_source.txt')
