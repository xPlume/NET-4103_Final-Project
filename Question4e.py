import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import Data
import networkx as nx
import os



class GANSageEncoder(nn.Module):
	def __init__(self, in_channels, hidden_channels, out_channels):
		super(GANSageEncoder, self).__init__()
		# Layer 1: Aggregates features from immediate neighbors
		self.conv1 = SAGEConv(in_channels, hidden_channels)
		# Layer 2: Aggregates from 2-hop neighborhood to capture wider topology
		self.conv2 = SAGEConv(hidden_channels, out_channels)
	#def

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index).relu()
		x = self.conv2(x, edge_index)
		return x
	#def
#class



class LinkPredictor(nn.Module):
	def forward(self, x_i, x_j):
		# Dot product decoder: higher similarity = higher link probability
		return (x_i * x_j).sum(dim=-1)
	#def
#class



def load_fb100_as_pyg(file_path):
    # Converts a Facebook100 GML file to a PyTorch Geometric Data object.
    G = nx.read_gml(file_path)
    # Ensure nodes are indexed 0 to N-1
    G = nx.convert_node_labels_to_integers(G)
    
    # Extract edges
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    
    # Extract features (Major, Dorm, Status, Gender)
    # We convert categorical attributes into a feature tensor
    attrs = ['major_index', 'dorm', 'student_fac', 'gender']
    x = []
    for node in G.nodes(data=True):
        feat = [node[1].get(a, 0) for a in attrs]
        x.append(feat)
	#for
	
    x = torch.tensor(x, dtype=torch.float)
    
    return Data(x=x, edge_index=edge_index)
#def



def train_link_predictor(data, epochs=100):
	# Split edges into training, validation, and test sets
	data = train_test_split_edges(data)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = GANSageEncoder(data.num_features, 64, 32).to(device)
	predictor = LinkPredictor().to(device)
	optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

	data = data.to(device)

	for epoch in range(1, epochs + 1):
		model.train()
		optimizer.zero_grad()
		
		# Encode: Get node embeddings
		z = model(data.x, data.train_pos_edge_index)
		
		# Decode: Predict for positive (real) and negative (fake) edges
		pos_out = predictor(z[data.train_pos_edge_index[0]], z[data.train_pos_edge_index[1]])
		# Negative sampling: internal PyG utility provides train_neg_edge_index
		neg_edge_index = torch.randint(0, z.size(0), data.train_pos_edge_index.size(), device=device)
		neg_out = predictor(z[neg_edge_index[0]], z[neg_edge_index[1]])
		
		# Loss: Binary Cross Entropy (Maximized for pos, Minimized for neg)
		loss = -torch.log(torch.sigmoid(pos_out) + 1e-15).mean() - \
		torch.log(1 - torch.sigmoid(neg_out) + 1e-15).mean()
		
		loss.backward()
		optimizer.step()
		
		if epoch % 10 == 0:
			print(f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}')
		#if
		
	#for
	
	return model, predictor
#def



pyg_data = load_fb100_as_pyg('fb100/data/Caltech36.gml')
trained_model, trained_predictor = train_link_predictor(pyg_data)