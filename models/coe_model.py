import torch
import torch.nn.functional as F

from models.base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device, to_numpy
from utils.geometry_util import compute_hks_autoscale, compute_wks_autoscale
from utils.fmap_util import nn_query, fmap2pointmap, pointmap2fmap
from utils.coe_util import pad_data

@MODEL_REGISTRY.register()
class CoeModel(BaseModel):
    """
    CoE: Deep Coupled Embedding for Non-Rigid Point Cloud Correspondences model
    """
    def __init__(self, opt):
        self.with_refine = opt.get('refine', -1)
        self.partial = opt.get('partial', False)
        self.non_isometric = opt.get('non-isometric', False)
        self.attention = opt.get('attention', False)
        super(CoeModel, self).__init__(opt)
        
    def feed_data(self, data):
        """Process input data and compute loss"""
        # Get data pair
        data_x, data_y = data['first'], data['second']
        
        # Move data to device
        data_x = to_device(data_x, self.device)
        data_y = to_device(data_y, self.device)
        
        # Extract vertices, eVecs, eVals, etc.
        vertices_x = data_x['verts']
        vertices_y = data_y['verts']
        eVecs_x = data_x['evecs']
        eVecs_y = data_y['evecs']
        eVals_x = data_x['evals']
        eVals_y = data_y['evals']
        Ls_x = data_x['L']
        Ls_y = data_y['L']
        Ms_x = data_x['mass']
        Ms_y = data_y['mass']
        faces_x = data_x.get('faces')
        faces_y = data_y.get('faces')
        
        max_verts = max(vertices_x.shape[1], vertices_y.shape[1])
        
        vertices_x = pad_data(vertices_x, max_verts)
        vertices_y = pad_data(vertices_y, max_verts)
        eVecs_x = pad_data(eVecs_x, max_verts)
        eVecs_y = pad_data(eVecs_y, max_verts)
        Ms_x = pad_data(Ms_x, max_verts)
        Ms_y = pad_data(Ms_y, max_verts)

        if Ls_x.is_sparse:
            Ls_x = Ls_x.to_dense()
        if Ls_y.is_sparse:
            Ls_y = Ls_y.to_dense()
        Ls_x = pad_data(Ls_x, max_verts)
        Ls_y = pad_data(Ls_y, max_verts)
             
        # Prepare descriptors (using wks features as in URLSSM)
        if 'descriptor' in data_x:
            descriptors_x = data_x['descriptor']
            descriptors_y = data_y['descriptor']
        else:
            # Compute WKS descriptors if not provided
            descriptors_x = compute_hks_autoscale(eVals_x, eVecs_x, count=16)
            descriptors_y = compute_hks_autoscale(eVals_y, eVecs_y, count=16)
        
        descriptors_x = pad_data(descriptors_x, max_verts)
        descriptors_y = pad_data(descriptors_y, max_verts)
        
        # Feature extraction
        feat_x = self.networks['feature_extractor'](vertices_x, faces_x)
        feat_y = self.networks['feature_extractor'](vertices_y, faces_y)
        
        # Apply cross attention if enabled
        if self.attention and 'attention' in self.networks:
            feat_x, feat_y = self.networks['attention'](feat_x, feat_y)
        # Stack features for consistent loss calculation
        consistent_bases = torch.stack([feat_x, feat_y], dim=1).squeeze(0)
        eVals_stacked = torch.stack([eVals_x, eVals_y], dim=1)
        Ls_stacked = torch.stack([Ls_x, Ls_y], dim=1)
        Ms_stacked = torch.stack([Ms_x, Ms_y], dim=1)
        
        descriptors_stacked = torch.stack([descriptors_x.squeeze(), descriptors_y.squeeze()], dim=0).unsqueeze(0)
        
        
        # Compute loss
        total_loss, loss_details = self.losses['consistent_loss'](
            eVals_stacked, consistent_bases.unsqueeze(0), 
            Ls_stacked, Ms_stacked, descriptors_stacked)
        
        self.loss_metrics = loss_details
        self.loss_metrics['l_total'] = total_loss
    
    def optimize_parameters(self):
        """Optimize model parameters"""
        super().optimize_parameters()
    
    def validate_single(self, data, timer):
        """Perform validation on a single data sample"""
        # Get data
        data_x, data_y = data['first'], data['second']
        
        # Move data to device
        data_x = to_device(data_x, self.device)
        data_y = to_device(data_y, self.device)
        
        # Start timer
        timer.start()
        
        # Extract features (following URLSSM patterns)
        vertices_x = data_x['verts']
        vertices_y = data_y['verts']
        faces_x = data_x.get('faces')
        faces_y = data_y.get('faces')

        # Extract features
        feat_x = self.networks['feature_extractor'](vertices_x, faces_x)
        feat_y = self.networks['feature_extractor'](vertices_y, faces_y)
        
        # Apply cross attention if enabled
        if self.attention and 'attention' in self.networks:
            feat_x, feat_y = self.networks['attention'](feat_x, feat_y)
        
        # Normalize features
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
        
        # rename feat to embeddings
        embeddings_x = feat_x.squeeze()
        embeddings_y = feat_y.squeeze()
        
        # Find correspondences
        p2p = nn_query(embeddings_x, embeddings_y)
        # p2p to fmap
        
        Cxy = pointmap2fmap(p2p, embeddings_y, embeddings_x) # [k, k]
        
        # p2p to permutation matrix Pyx
        Pyx = embeddings_y @ Cxy @ embeddings_x.transpose(0, 1)

        # Stop timer
        timer.record()
        
        return p2p, Pyx, Cxy  
