import torch
import torch.nn as nn
from copy import deepcopy
from utils.registry import NETWORK_REGISTRY

class MultiHeadAttention(nn.Module):
    """ Multi-head attention module """
    def __init__(self, num_heads, d_model):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        batch_dim = query.size(0)
        query, key, value = [ll(x).view(batch_dim, self.dim, self.num_heads, -1) 
                             for ll, x in zip(self.proj, (query, key, value))]
        
        # Compute attention scores and apply attention
        scores = torch.einsum("bdhn,bdhm->bhnm", query, key) / self.dim ** 0.5
        prob = torch.nn.functional.softmax(scores, dim=-1)
        result = torch.einsum("bhnm,bdhm->bdhn", prob, value)
        
        return self.merge(result.contiguous().view(batch_dim, self.dim * self.num_heads, -1))

class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, feature_dim)
        self.mlp = nn.Sequential(
            nn.Conv1d(feature_dim * 2, feature_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(feature_dim * 2, feature_dim, kernel_size=1)
        )
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))

@NETWORK_REGISTRY.register()
class CrossAttentionRefinementNet(nn.Module):
    """
    Cross-attention refinement network for CoE
    """
    def __init__(self, n_in=128, num_head=4, gnn_dim=128, n_layers=1):
        super().__init__()
        self.n_in = n_in
        self.layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.layers.append(AttentionalPropagation(gnn_dim, num_head))
    
    def forward(self, features_x, features_y):
        # Transpose features for Conv1d operations
        desc0, desc1 = features_x.transpose(1, 2), features_y.transpose(1, 2)
        
        # Apply cross-attention layers
        for layer in self.layers:
            desc0 = desc0 + layer(desc0, desc1)
            desc1 = desc1 + layer(desc1, desc0)
        
        # Transpose back to original dimension ordering
        return desc0.transpose(1, 2), desc1.transpose(1, 2)