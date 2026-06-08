from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Any

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    class nn:
        class Module:
            def __init__(self, *args, **kwargs): pass
            def eval(self): return self
            def train(self): return self
            def to(self, *args, **kwargs): return self
            def __call__(self, *args, **kwargs): return self

NODE_TYPES = [
    "alert",
    "incident",
    "entity",
    "account",
    "device",
    "ip",
    "domain",
    "file",
    "evidence",
    "mitre",
    "cti",
    "timestamp",
    "contradiction",
]
NODE_TYPE_TO_IDX = {nt: idx for idx, nt in enumerate(NODE_TYPES)}


def get_node_features(graph: nx.DiGraph, node: Any) -> np.ndarray:
    """Extract a 17-dimensional feature vector for a node in a DERG graph.
    
    Features:
    - reliability_weight (1)
    - confidence_score (1)
    - contradiction_weight (1)
    - risk_score (1)
    - one-hot node type (13)
    """
    node_data = graph.nodes[node]
    features = np.zeros(17, dtype=np.float32)
    features[0] = float(node_data.get("reliability_weight", 0.7))
    features[1] = float(node_data.get("confidence_score", 0.7))
    features[2] = float(node_data.get("contradiction_weight", 0.0))
    features[3] = float(node_data.get("risk_score", 0.0))
    
    nt = node_data.get("node_type", "unknown")
    if nt in NODE_TYPE_TO_IDX:
        features[4 + NODE_TYPE_TO_IDX[nt]] = 1.0
    return features


class GCNLayer(nn.Module):
    """Pure PyTorch implementation of a Graph Convolution (GCN) layer.
    
    H^(l+1) = ReLU(D_hat^-1/2 * A_hat * D_hat^-1/2 * H^(l) * W)
    """
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        if HAS_TORCH:
            self.linear = nn.Linear(in_features, out_features, bias=False)
            self.bias = nn.Parameter(torch.zeros(out_features))
            # Initialize weights
            nn.init.xavier_uniform_(self.linear.weight)
        else:
            self.linear = None
            self.bias = None

    def forward(self, x: torch.Tensor, adj_hat: torch.Tensor) -> torch.Tensor:
        # x shape: (N, in_features)
        # adj_hat shape: (N, N)
        if not HAS_TORCH:
            return x
        # Message passing: A_hat * X
        support = torch.matmul(adj_hat, x)
        # Linear projection: (A_hat * X) * W
        out = self.linear(support) + self.bias
        return out


class DERGGNNEncoder(nn.Module):
    """GNN encoder for DERG graphs that outputs a graph-level embedding."""
    def __init__(self, input_dim: int = 17, hidden_dim: int = 32, output_dim: int = 16) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        if HAS_TORCH:
            self.gcn1 = GCNLayer(input_dim, hidden_dim)
            self.gcn2 = GCNLayer(hidden_dim, output_dim)
            self.fc = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim)
            )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : node features tensor of shape (N, input_dim)
        adj : adjacency matrix tensor of shape (N, N)
        """
        if not HAS_TORCH:
            return torch.zeros((1, self.output_dim))
            
        N = x.size(0)
        # A_hat = A + I_N
        A_hat = adj + torch.eye(N, device=adj.device)
        
        # Degree matrix D_hat
        rowsum = A_hat.sum(dim=1)
        d_inv_sqrt = torch.pow(rowsum, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # Symmetrically normalized adjacency matrix: D_hat^-1/2 * A_hat * D_hat^-1/2
        adj_normalized = torch.matmul(torch.matmul(D_inv_sqrt, A_hat), D_inv_sqrt)
        
        # GCN propagation
        h = F.relu(self.gcn1(x, adj_normalized))
        h = self.gcn2(h, adj_normalized)
        
        # Global Pooling (Mean pool across nodes)
        g = torch.mean(h, dim=0, keepdim=True) # shape: (1, output_dim)
        
        # FC projection
        out = self.fc(g)
        return out


def compute_derg_embeddings(
    graphs: list[nx.DiGraph],
    encoder: DERGGNNEncoder | None = None,
    output_dim: int = 16,
    seed: int = 42,
) -> np.ndarray:
    """Compute graph embeddings for a batch of NetworkX DiGraphs.
    
    If PyTorch is not installed or graph list is empty, returns zero embeddings.
    """
    n_graphs = len(graphs)
    if n_graphs == 0:
        return np.zeros((0, output_dim), dtype=np.float32)
        
    if not HAS_TORCH:
        # Fallback to zero features if torch is not installed
        return np.zeros((n_graphs, output_dim), dtype=np.float32)
        
    # Set seed for reproducible initialization
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if encoder is None:
        encoder = DERGGNNEncoder(input_dim=17, hidden_dim=32, output_dim=output_dim)
    
    encoder = encoder.to(device)
    encoder.eval()
        
    embeddings = []
    with torch.no_grad():
        for graph in graphs:
            if graph.number_of_nodes() == 0:
                embeddings.append(np.zeros(output_dim, dtype=np.float32))
                continue
                
            # 1. Node feature matrix
            nodes = list(graph.nodes)
            x_np = np.vstack([get_node_features(graph, node) for node in nodes])
            x = torch.tensor(x_np, dtype=torch.float32, device=device)
            
            # 2. Adjacency matrix
            adj_np = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.float32)
            adj = torch.tensor(adj_np, dtype=torch.float32, device=device)
            
            # 3. GNN forward
            emb = encoder(x, adj)
            embeddings.append(emb.cpu().numpy()[0])
            
    return np.vstack(embeddings)
