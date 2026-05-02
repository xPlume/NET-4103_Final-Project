import networkx as nx
import numpy as np
from abc import ABC, abstractmethod
import random
import os



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








# ---------- Question 4.c
def evaluate_link_prediction(G, fraction, k_values):
	# Prepare edges for removal
	edges = list(G.edges())
	num_to_remove = int(len(edges) * fraction)

	# Ensure graph stays connected (optional but recommended for stability)
	# For a simple implementation, we just shuffle and remove
	random.shuffle(edges)
	removed_edges = set(edges[:num_to_remove])

	# Create the partial graph (Training set)
	G_train = G.copy()
	G_train.remove_edges_from(removed_edges)

	# Instantiate predictors
	predictors = {
		"Common Neighbors": CommonNeighbors(G_train),
		"Jaccard": Jaccard(G_train),
		"Adamic/Adar": AdamicAdar(G_train)
	}

	# Identify all non-existent edges in G_train to score
	# To follow point 3 strictly: $|V| x |V|$ minus existing edges in G_train
	nodes = list(G_train.nodes())
	potential_edges = []

	# Using a subset of non-edges if graph is too large, 
	# but for Caltech (small) we can do the full set.
	for i in range(len(nodes)):
		for j in range(i + 1, len(nodes)):
			u, v = nodes[i], nodes[j]
			if not G_train.has_edge(u, v):
				potential_edges.append((u, v))
			#if
		#for
	#for

	results = {}

	for name, model in predictors.items():
		# Score each pair
		scores = []
		for u, v in potential_edges:
			p = model.fit(u, v)
			scores.append(((u, v), p))
		#for
		
		# Sort in decreasing order of confidence
		scores.sort(key=lambda x: x[1], reverse=True)
		
		model_results = []
		for k in k_values:
			# Take top k pairs
			top_k_edges = set([pair for pair, score in scores[:k]])
			
			# Compute Intersection (True Positives)
			# TP = Correctly predicted removed edges
			tp = len(top_k_edges.intersection(removed_edges))
			fp = k - tp
			fn = len(removed_edges) - tp
			
			# Metrics
			precision = tp / (tp + fp) if (tp + fp) > 0 else 0
			recall = tp / (tp + fn) if (tp + fn) > 0 else 0
			top_k_rate = tp / k  # As defined: % of correctly classified among top k
			
			model_results.append({
				"k": k,
				"precision": precision,
				"recall": recall,
				"top@k_rate": top_k_rate
			})
			
		#for 
		
		results[name] = model_results
	#for
	
	return results
#def


# Choosing Caltech as it is a smaller network
G = nx.read_gml("fb100/data/Caltech36.gml")
lcc = G.subgraph(max(nx.connected_components(G), key=len))
stats = evaluate_link_prediction(lcc, 0.1, [50, 100, 200, 400])

print(stats)