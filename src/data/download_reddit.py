import os
from torch_geometric.datasets import Reddit

# Set up the data root directory for PyTorch Geometric datasets
root = os.environ.get("PYG_DATA_ROOT", os.path.expandvars("$SCRATCH/pyg_datasets"))
print("Using root:", root)
dataset = Reddit(root=root)
print(f"Dataset: {dataset}:")
print('====================')
print("Done.")