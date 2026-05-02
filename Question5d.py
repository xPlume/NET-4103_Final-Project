import torch
import torch.optim as optim
import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error  # New import for MAE

from Question5b import GCN, train_step



def normalize_adj(adj):
	adj = sp.coo_matrix(adj)
	rowsum = np.array(adj.sum(1))
	d_inv_sqrt = np.power(rowsum, -0.5).flatten()
	d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
	d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
	return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
#def



def sparse_mx_to_torch_sparse_tensor(sparse_mx):
	sparse_mx = sparse_mx.tocoo().astype(np.float32)
	indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
	values = torch.from_numpy(sparse_mx.data)
	shape = torch.Size(sparse_mx.shape)
	return torch.sparse.FloatTensor(indices, values, shape)
#def



def run_experiment(file_path, target_attr, removal_rate):
	# Load the Facebook100 GML file
	G = nx.read_gml(file_path)
	adj = nx.adjacency_matrix(G)
	
	# Apply renormalization trick: A_hat = A + I
	adj_tilde = adj + sp.eye(adj.shape[0])
	adj_normalized = normalize_adj(adj_tilde)
	adj_tensor = sparse_mx_to_torch_sparse_tensor(adj_normalized)
	
	# Extract and Encode Labels
	labels_raw = [str(G.nodes[n].get(target_attr, "unknown")) for n in G.nodes()]
	le = LabelEncoder()
	labels = torch.LongTensor(le.fit_transform(labels_raw))
	
	# Create Feature Matrix (X)
	n_nodes = len(G.nodes())
	features = torch.eye(n_nodes) 
	
	# Create Masks for "Missing" Attributes
	indices = np.random.permutation(n_nodes)
	train_size = int(n_nodes * (1 - removal_rate))
	
	train_idx = indices[:train_size]
	test_idx = indices[train_size:]

	train_mask = torch.zeros(n_nodes, dtype=torch.bool)
	train_mask[train_idx] = True
	
	# Initialize Model from Question5b
	n_feat = features.shape[1]
	n_class = len(le.classes_)
	model = GCN(n_feat=n_feat, n_hid=32, n_class=n_class, dropout_rate=0.5)
	optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
	
	# Training
	for epoch in range(101):
		loss = train_step(model, optimizer, features, adj_tensor, labels, train_mask)
	#for

	# Evaluation
	model.eval()
	with torch.no_grad():
		output = model(features, adj_tensor)
		
		# Get predictions for the removed (test) nodes
		preds = output[test_idx].max(1)[1]
		actuals = labels[test_idx]
		
		# Calculate Accuracy using Eq. 1: (1/n) * sum(indicator function)
		correct = preds.eq(actuals).sum().item()
		acc = correct / len(test_idx)
		
		# Calculate Mean Absolute Error
		mae = mean_absolute_error(actuals.cpu().numpy(), preds.cpu().numpy())
	#with
	
	return acc, mae
#def





# Configuration
FILE_NAME = "fb100/data/Duke14.gml"
ATTRIBUTES = ["dorm", "major", "gender"]
REMOVAL_RATES = [0.1, 0.2, 0.3]

final_results = {}

print(f"Starting Evaluation on {FILE_NAME}")
print("-" * 60)

for attr in ATTRIBUTES:
	for rate in REMOVAL_RATES:
		print(f">> Testing Attribute: {attr} | Removal Rate: {rate*100:.0f}%")
		try:
			accuracy, mae_score = run_experiment(FILE_NAME, attr, rate)
			final_results[(attr, rate)] = (accuracy, mae_score)
			print(f"   Success! Acc: {accuracy:.4f} | MAE: {mae_score:.4f}\n")
		except Exception as e:
			print(f"   Error processing {attr}: {e}\n")
		#except
	#for
#for


# Final Summary Table
print("\n" + "="*65)
print(f"{'Attribute':<12} | {'Removed %':<10} | {'Accuracy':<12} | {'MAE':<10}")
print("-" * 65)

for (attr, rate), (acc, mae) in final_results.items():
	print(f"{attr:<12} | {int(rate*100):<10}% | {acc:<12.4f} | {mae:<10.4f}")
#for

print("="*65)