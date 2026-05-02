import networkx as nx
import numpy as np
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import os
import time



# ---------- Question 4.b
class LinkPrediction(ABC):
	def __init__(self, graph):
		self.graph = graph
		self.N = len(graph)
	#def
	
	def neighbors(self, v):
		neighbors_list = self.graph.neighbors(v)
		return list(neighbors_list)
	#def
	
	@abstractmethod
	def fit(self):
		raise NotImplementedError("Fit must be implemented")
	#def
#class



class CommonNeighbors(LinkPrediction):
	def __init__(self, graph):
		super(CommonNeighbors, self).__init__(graph)
	#def
	
	def fit(self, u, v):
		"""Returns the number of neighbors that u and v have in common."""
		set_u = set(self.neighbors(u))
		set_v = set(self.neighbors(v))
		return len(set_u.intersection(set_v))
	#def
#class


class Jaccard(LinkPrediction):
	def __init__(self, graph):
		super(Jaccard, self).__init__(graph)
	#def
	
	def fit(self, u, v):
		"""Returns the size of intersection divided by the size of union."""
		set_u = set(self.neighbors(u))
		set_v = set(self.neighbors(v))
		intersection = len(set_u.intersection(set_v))
		union = len(set_u.union(set_v))
		return intersection / union if union > 0 else 0.0
	#def
#class


class AdamicAdar(LinkPrediction):
	def __init__(self, graph):
		super(AdamicAdar, self).__init__(graph)
	#def
	
	def fit(self, u, v):
		"""Returns the sum of 1/log(degree) of common neighbors."""
		set_u = set(self.neighbors(u))
		set_v = set(self.neighbors(v))
		common = set_u.intersection(set_v)
		
		score = 0.0
		for neighbor in common:
			degree = len(self.neighbors(neighbor))
			if degree > 1:  # log(1) is 0, so we avoid division by zero
				score += 1.0 / np.log(degree)
			#if
		#for
		return score
	#def
#class








# ---------- Question 4 d. Evaluating the performance of our model
def evaluate_efficiency(data_path, selected_universities, fraction=0.1, k_values=[50, 100, 200, 400]):
	efficiency_results = {}
	
	for name, filename in selected_universities.items():
		file_path = os.path.join(data_path, filename)
		if not os.path.exists(file_path):
			continue
		#if not
		
		# Load and get LCC
		G = nx.read_gml(file_path)
		lcc_nodes = max(nx.connected_components(G), key=len)
		G_lcc = G.subgraph(lcc_nodes).copy()
		
		# Setup test environment (remove edges)
		edges = list(G_lcc.edges())
		np.random.shuffle(edges)
		num_to_remove = int(len(edges) * fraction)
		removed_edges = set(edges[:num_to_remove])
		G_train = G_lcc.copy()
		G_train.remove_edges_from(removed_edges)
		
		# Identify non-edges to predict (sample for speed on larger graphs)
		nodes = list(G_train.nodes())
		potential_edges = []
		
		# Limit sampling to 10,000 pairs to keep execution time reasonable
		num_samples = min(10000, (len(nodes)*(len(nodes)-1))//2)
		while len(potential_edges) < num_samples:
			u, v = np.random.choice(nodes, 2, replace=False)
			if not G_train.has_edge(u, v):
				potential_edges.append((u, v))
			#if
		#while
		
		predictors = {
			"Common Neighbors": CommonNeighbors(G_train),
			"Jaccard": Jaccard(G_train),
			"Adamic/Adar": AdamicAdar(G_train)
		}
		
		univ_timings = {}
		for metric_name, model in predictors.items():
			start_time = time.perf_counter()
			for u, v in potential_edges:
				model.fit(u, v)
			#for
			
			end_time = time.perf_counter()
			univ_timings[metric_name] = end_time - start_time
		#for
		
		efficiency_results[name] = {
			"timings": univ_timings,
			"n_nodes": G_lcc.number_of_nodes(),
			"n_edges": G_lcc.number_of_edges()
		}
	#for
	
	
	return efficiency_results
#def 





# Run evaluation
data_dir = "fb100/data/"
test_univs = {
	"Caltech": "Caltech36.gml",
	"Johns Hopkins": "Johns Hopkins55.gml",
	"Princeton": "Princeton12.gml",
	"Vanderbilt": "Vanderbilt48", 
	"Rochester": "Rochester38.gml", 
}
results = evaluate_efficiency(data_dir, test_univs)

# Plotting
labels = list(results.keys())
cn_times = [results[u]['timings']['Common Neighbors'] for u in labels]
j_times = [results[u]['timings']['Jaccard'] for u in labels]
aa_times = [results[u]['timings']['Adamic/Adar'] for u in labels]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width, cn_times, width, label='Common Neighbors')
ax.bar(x, j_times, width, label='Jaccard')
ax.bar(x + width, aa_times, width, label='Adamic/Adar')

ax.set_ylabel('Execution Time (seconds)')
ax.set_title('Metric Efficiency Comparison (10k Predictions)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()