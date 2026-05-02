import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):

	def __init__(self, in_features, out_features):
		super(GCNLayer, self).__init__()
		self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
		nn.init.xavier_uniform_(self.weight)
	#def
	
	def forward(self, x, adj):
		# Support = X * W
		support = torch.mm(x, self.weight)
		# Output = A_hat * Support
		output = torch.spmm(adj, support)
		return output
	#def
#class


class GCN(nn.Module):
	def __init__(self, n_feat, n_hid, n_class, dropout_rate):
		super(GCN, self).__init__()
		self.gc1 = GCNLayer(n_feat, n_hid)
		self.gc2 = GCNLayer(n_hid, n_class)
		self.dropout = dropout_rate
	#def

	def forward(self, x, adj):
		# First layer with ReLU activation
		x = F.relu(self.gc1(x, adj))
		# Dropout to introduce stochasticity and prevent overfitting
		x = F.dropout(x, self.dropout, training=self.training)
		# Second layer (output layer)
		x = self.gc2(x, adj)
		# Softmax for multi-class classification
		return F.log_softmax(x, dim=1)
	#def
#class


def train_step(model, optimizer, features, adj, labels, train_mask):
	model.train()
	optimizer.zero_grad()
	output = model(features, adj)
	# Loss calculated only on the labeled training nodes (mask)
	loss = F.nll_loss(output[train_mask], labels[train_mask])
	loss.backward()
	optimizer.step()
	return loss.item()
#def
