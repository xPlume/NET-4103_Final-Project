import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def calculate_ig(G, nodes):
    """Calculates Intra-group Density (IG)."""
    num_nodes = len(nodes)
    if num_nodes < 2:
        return 0
    
    subgraph = G.subgraph(nodes)
    num_edges = subgraph.number_of_edges()
    
    # IG = (2 * num_edges) / (num_nodes * (num_nodes - 1))
    IG = (2 * num_edges) / (num_nodes * (num_nodes - 1))
    return IG

def process_all_universities(directory_path):
    # Find all .gml files in the folder
    file_pattern = os.path.join(directory_path, "*.gml")
    file_list = glob.glob(file_pattern)
    
    all_results = []
    
    print(f"Found {len(file_list)} universities. Starting analysis...")

    for file_path in file_list:
        uni_name = os.path.basename(file_path).replace(".gml", "")
        
        try:
            # Load GML file
            G = nx.read_gml(file_path)
            
            # 1. Structural Groups (Louvain)
            # Note: Ensure networkx is version 2.7+ for this
            communities = nx.community.louvain_communities(G, seed=42)
            ig_struct = [calculate_ig(G, c) for c in communities if len(c) > 5]
            avg_ig_struct = np.mean(ig_struct) if ig_struct else 0

            # 2. Attribute Groups (e.g., 'year')
            # GML files store attributes in node dictionaries
            years = nx.get_node_attributes(G, 'year')
            year_groups = {}
            for node, year_val in years.items():
                if year_val and year_val != 0:
                    year_groups.setdefault(year_val, []).append(node)
            
            ig_year = [calculate_ig(G, nodes) for nodes in year_groups.values() if len(nodes) > 5]
            avg_ig_year = np.mean(ig_year) if ig_year else 0
            
            all_results.append({
                "University": uni_name,
                "Structural_IG": avg_ig_struct,
                "Year_IG": avg_ig_year,
                "Size": G.number_of_nodes()
            })
            
        except Exception as e:
            print(f"Skipping {uni_name} due to error: {e}")

    return pd.DataFrame(all_results)

# --- Execution ---



# Loading the data present in the directory Data6
path = "Data6/"
df_results = process_all_universities(path)


plt.figure(figsize=(12, 7))

# Plotting Structural vs Year density
plt.scatter(df_results['Size'], df_results['Structural_IG'], alpha=0.6, label='Structural (Louvain)', color='blue')
plt.scatter(df_results['Size'], df_results['Year_IG'], alpha=0.6, label='Attribute (Grad Year)', color='red')

plt.title("Intra-group Density (IG) across the FB100 Dataset")
plt.xlabel("University Size (Number of Students)")
plt.ylabel("Density (IG)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()