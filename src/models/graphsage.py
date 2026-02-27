import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

"""GraphSAGE model implementation using stacked SAGEConv layers.

Args:
    in_channels (int): Number of input node features.
    hidden_channels (int): Number of hidden features in intermediate layers.
    out_channels (int): Number of output features.
    num_layers (int, optional): Total number of GraphSAGE layers (>= 2). Defaults to 2.
    dropout (float, optional): Dropout probability applied after each hidden layer. Defaults to 0.5.

Forward:
    x (Tensor): Node feature matrix of shape [num_nodes, in_channels].
    edge_index (LongTensor): Graph connectivity in COO format.

Returns:
    Tensor: Output node representations of shape [num_nodes, out_channels].
"""
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        assert num_layers >= 2
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
