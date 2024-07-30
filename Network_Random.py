import pandas as pd
import random

# Function to generate random connections in a social network
def generate_social_network_data(num_rows):
    data = {
        'node1': [],
        'node2': [],
        'weight': []
    }

    for _ in range(num_rows):
        node1 = random.randint(1, 20)  # Assuming we have unique nodes
        node2 = random.randint(1, 20)
        while node2 == node1:  # Ensure node1 and node2 are different
            node2 = random.randint(1, 20)
        
        weight = random.randint(1, 100)  # Random weight between 0.1 and 10.0

        data['node1'].append(node1)
        data['node2'].append(node2)
        data['weight'].append(round(weight, 2))

    return pd.DataFrame(data)

# Generate the social network data with 5000 rows
num_rows = 220
social_network_df = generate_social_network_data(num_rows)

# Display the first few rows of the dataframe
print(social_network_df.head())

# Save the dataframe to a CSV file
social_network_df.to_csv('saturated connectio12ns.csv', index=False)


