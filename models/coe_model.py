import torch
import torch.nn.functional as F

from models.base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device, to_numpy
from utils.geometry_util import compute_hks_autoscale, compute_wks_autoscale
from utils.fmap_util import nn_query

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
        
        # Prepare descriptors (using wks features as in URLSSM)
        if 'descriptor' in data_x:
            descriptors_x = data_x['descriptor']
            descriptors_y = data_y['descriptor']
        else:
            # Compute WKS descriptors if not provided
            descriptors_x = compute_hks_autoscale(eVals_x.unsqueeze(0), eVecs_x.unsqueeze(0), 
                                                 Ms_x.unsqueeze(0), n_descr=128)
            descriptors_y = compute_hks_autoscale(eVals_y.unsqueeze(0), eVecs_y.unsqueeze(0), 
                                                 Ms_y.unsqueeze(0), n_descr=128)
        
        # Feature extraction
        feat_x = self.networks['feature_extractor'](vertices_x, faces_x, 
                                                   mass=Ms_x, L=Ls_x, 
                                                   evals=eVals_x, evecs=eVecs_x)
        feat_y = self.networks['feature_extractor'](vertices_y, faces_y, 
                                                   mass=Ms_y, L=Ls_y, 
                                                   evals=eVals_y, evecs=eVecs_y)
        
        # Apply cross attention if enabled
        if self.attention and 'attention' in self.networks:
            feat_x, feat_y = self.networks['attention'](feat_x, feat_y)
        
        # Stack features for consistent loss calculation
        consistent_bases = torch.stack([feat_x, feat_y], dim=1)
        eVals_stacked = torch.stack([eVals_x, eVals_y], dim=1).unsqueeze(0)
        Ls_stacked = torch.stack([Ls_x, Ls_y], dim=1).unsqueeze(0)
        Ms_stacked = torch.stack([Ms_x, Ms_y], dim=1).unsqueeze(0)
        descriptors_stacked = torch.stack([descriptors_x.squeeze(), descriptors_y.squeeze()], dim=1).unsqueeze(0)
        
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
        
        # Save state for refinement
        if self.with_refine > 0:
            state_dict = {'networks': self._get_networks_state_dict()}
            
            # Perform test-time refinement
            for _ in range(self.with_refine):
                self.feed_data(data)
                self.optimize_parameters()
        
        # Extract features (following URLSSM patterns)
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
        
        # Extract features
        feat_x = self.networks['feature_extractor'](vertices_x, faces_x, 
                                                   mass=Ms_x, L=Ls_x, 
                                                   evals=eVals_x, evecs=eVecs_x)
        feat_y = self.networks['feature_extractor'](vertices_y, faces_y, 
                                                   mass=Ms_y, L=Ls_y, 
                                                   evals=eVals_y, evecs=eVecs_y)
        
        # Apply cross attention if enabled
        if self.attention and 'attention' in self.networks:
            feat_x, feat_y = self.networks['attention'](feat_x, feat_y)
        
        # Normalize features
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
        
        # Find correspondences
        p2p = nn_query(feat_x, feat_y)
        
        # Stop timer
        timer.record()
        
        # Restore previous state if refinement was used
        if self.with_refine > 0:
            self.resume_model(state_dict, net_only=True, verbose=False)
        
        return p2p, None, None  # Return point-to-point map