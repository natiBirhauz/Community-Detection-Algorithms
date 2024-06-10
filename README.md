fr_algorithm.py shows the graph using Fruchterman-Reingold force-directed algorithm.

--------------------------------------------------------------------------------

for the  Girvan-Newman algorithm run on consule : 	python cmty.py graph.txt
python cmty.py social_network100.txt

## Input
 
A weighted graph G. See graph.txt file as a sample for the input graph format. It's a CSV file where each line has the following format: 

	u,v,w 

Above line specifies that there is an edge between node u and v with positive weight w. 
The lowest id should be zero and the nodes id's increase. If you want to used this code for an unweighted graph, 
simply set the weight equal to one on each input line.

## Output

This code runs Girvan-Newman algorithm and returns a list of detected communities with maximum modularity.
