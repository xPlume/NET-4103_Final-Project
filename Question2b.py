import networkx as nx
import matplotlib.pyplot as plt
import os

# Configuration
data_path = "fb100/data/"
networks = {
    "Caltech": "Caltech36.gml",
    "MIT": "MIT8.gml",
    "Johns Hopkins": "Johns Hopkins55.gml"
}

# Create a figure with three subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, (name, filename) in zip(axes, networks.items()):
	file_path = os.path.join(data_path, filename)

	if not os.path.exists(file_path):
		ax.set_title(f"{name} (File Not Found)")
		continue
	#if not
	
	# Load and extract LCC
	G = nx.read_gml(file_path)
	lcc_nodes = max(nx.connected_components(G), key=len)
	G_lcc = G.subgraph(lcc_nodes)
	
	# 1. Calculate degrees and local clustering coefficients
	# clustering() returns a dict: {node: value}
	# degree() returns a DegreeView: (node, degree)
	local_clustering = nx.clustering(G_lcc)
	degrees = dict(G_lcc.degree())
	
	# 2. Align values by node
	x = [degrees[node] for node in G_lcc.nodes()]
	y = [local_clustering[node] for node in G_lcc.nodes()]
	
	# 3. Plotting
	ax.scatter(x, y, alpha=0.3, s=15, edgecolors='none')
	ax.set_title(f"{name}")
	ax.set_xlabel("Degree (k)")
	if ax == axes[0]:
		ax.set_ylabel("Local Clustering Coefficient")
	#if
	
	ax.grid(True, linestyle='--', alpha=0.5)

#for

plt.tight_layout()
plt.suptitle("Degree vs. Local Clustering Coefficient in FB100 Networks", y=1.05, fontsize=16)
plt.show()