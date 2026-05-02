import networkx as nx
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_path = "fb100/data/"
all_files = [f for f in os.listdir(data_path) if f.endswith('.gml')]

# Configuration
attributes = {
    'Status': 'student_fac',
    'Major': 'major_index',
    'Dorm': 'dorm',
    'Gender': 'gender'
}

data_list = []

for filename in all_files:
	file_path = os.path.join(data_path, filename)
	G = nx.read_gml(file_path)
	
	# Analyze the Largest Connected Component
	lcc_nodes = max(nx.connected_components(G), key=len)
	G_lcc = G.subgraph(lcc_nodes)
	
	n = G_lcc.number_of_nodes()
	
	# Calculate degree assortativity
	deg_assort = nx.degree_assortativity_coefficient(G_lcc)
	
	# Calculate attribute assortativities
	results = {'n': n, 'Degree': deg_assort}
	for label, key in attributes.items():
		try:
			# Handle potential missing attributes in some files
			coeff = nx.attribute_assortativity_coefficient(G_lcc, key)
			results[label] = coeff
		except KeyError:
			results[label] = None
		#except
	#for
	
	data_list.append(results)
	
#for

df = pd.DataFrame(data_list)


# Plotting 
fig_scatter, axes_scatter = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
fig_hist, axes_hist = plt.subplots(1, 5, figsize=(25, 5))

cols = ['Status', 'Major', 'Degree', 'Dorm', 'Gender']


for i, col in enumerate(cols):
	# Scatter Plot
	axes_scatter[i].scatter(df['n'], df[col], alpha=0.6)
	axes_scatter[i].axhline(0, color='red', linestyle='--') # Line of no assortativity
	axes_scatter[i].set_xscale('log')
	axes_scatter[i].set_title(f'{col} vs Size (n)')
	axes_scatter[i].set_xlabel('Network Size (n)')
	if i == 0:
		axes_scatter[i].set_ylabel('Assortativity Coefficient')
	#if
	
	# Histogram
	sns.histplot(df[col].dropna(), kde=True, ax=axes_hist[i], color='teal')
	axes_hist[i].axvline(0, color='red', linestyle='--')
	axes_hist[i].set_title(f'{col} Distribution')
	
#for


plt.tight_layout()
plt.show()