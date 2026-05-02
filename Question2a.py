import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os


# Configuration
data_path = "fb100/data/"
networks = {
    "Caltech": "Caltech36.gml",
    "MIT": "MIT8.gml",
    "Johns Hopkins": "Johns Hopkins55.gml"
}


def degree_distribution():

	plt.figure(figsize=(10, 6))


	for name, filename in networks.items():
		file_path = os.path.join(data_path, filename)

		# Loading the network
		if not os.path.exists(file_path):
			print(f"Warning: {file_path} not found.")
			continue
		# if not
		
		G = nx.read_gml(file_path)

		# Extracting the Largest Connected Component (LCC)
		lcc_nodes = max(nx.connected_components(G), key=len)
		G_lcc = G.subgraph(lcc_nodes)

		# Getting the degree sequence
		# G_lcc.degree() returns a DegreeView (node, degree)
		degrees = [d for n, d in G_lcc.degree()]

		# Calculating the distribution using Numpy
		# We count the frequency of each degree value
		degree_counts = np.bincount(degrees)
		degree_values = np.arange(len(degree_counts))

		# Filtering out zero counts for cleaner plotting (especially for log scale)
		nonzero = degree_counts > 0
		x = degree_values[nonzero]
		y = degree_counts[nonzero]

		# Plotting
		plt.loglog(x, y / G_lcc.number_of_nodes(), 'o', label=name, alpha=0.7)
	#for 
	
	
	# Formatting the plot
	plt.title("Degree Distribution of FB100 Networks (LCC)")
	plt.xlabel("Degree (k)")
	plt.ylabel("P(k)")
	plt.legend()
	plt.grid(True, which="both", ls="-", alpha=0.2)
	plt.show()

#degree_distribution


def Clustering_Coefficient():
	
	results = {}
	
	for name, filename in networks.items():
		file_path = os.path.join(data_path, filename)
		
		if not os.path.exists(file_path):
			continue
			
		# Load and extract LCC
		G = nx.read_gml(file_path)
		lcc_nodes = max(nx.connected_components(G), key=len)
		G_lcc = G.subgraph(lcc_nodes)
		
		# 1. Global Clustering Coefficient (Transitivity)
		global_cc = nx.transitivity(G_lcc)
		
		# 2. Mean Local Clustering Coefficient
		mean_local_cc = nx.average_clustering(G_lcc)
		
		# 3. Edge Density
		# nx.density() uses the formula 2*E / (V * (V-1)) for undirected graphs
		density = nx.density(G_lcc)
		
		results[name] = {
			"Global CC": round(global_cc, 4),
			"Mean Local CC": round(mean_local_cc, 4),
			"Edge Density": f"{density:.6f}" # Density is often very small
		}

	# Display Results
	print(f"{'University':<15} | {'Global CC':<10} | {'Mean Local CC':<15} | {'Edge Density'}")
	print("-" * 65)
	for univ, metrics in results.items():
		print(f"{univ:<15} | {metrics['Global CC']:<10} | {metrics['Mean Local CC']:<15} | {metrics['Edge Density']}")
	#for
	
#Clustering_Coefficient



degree_distribution()
Clustering_Coefficient()