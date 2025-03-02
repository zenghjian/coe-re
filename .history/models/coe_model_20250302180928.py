import torch
import torch.nn.functional as F

from models.base_model import BaseModel
from utils.registry import MODEL_REGISTRY
from utils.tensor_util import to_device
from utils.geometry_util import compute_hks_autoscale, compute_wks_autoscale

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
        file_names, vertices, eVals, eVecs, Ls, Ms, descriptors, gradXs, gradYs = data
        
        # Move data to device
        vertices = to_device(vertices, self.device)
        eVals = to_device(eVals, self.device)
        eVecs = to_device(eVecs, self.device)
        Ls = to_device(Ls, self.device)
        Ms = to_device(Ms, self.device)
        descriptors = to_device(descriptors, self.device)
        gradXs = to_device(gradXs, self.device)
        gradYs = to_device(gradYs, self.device)
        
        # Process through feature extractor
        inputs = (vertices, eVecs, eVals, Ls, Ms, gradXs, gradYs)
        consistent_bases = self.networks['feature_extractor'](inputs)
        
        # Apply cross attention if enabled
        if self.attention and 'attention' in self.networks:
            consistent_bases_refined = self.networks['attention'](
                consistent_bases[:, 0], consistent_bases[:, 1])
            consistent_bases = torch.stack(consistent_bases_refined, dim=1)
        
        # Compute losses
        total_loss, loss_details = self.losses['consistent_loss'](eVals, consistent_bases, Ls, Ms, descriptors)
        
        self.loss_metrics = loss_details
        self.loss_metrics['l_total'] = total_loss
    
    def optimize_parameters(self):
        """Optimize model parameters"""
        super().optimize_parameters()
    
    def validate_single(self, data, timer):
        """Perform validation on a single data sample"""
        # Get data
        file_names, vertices, eVals, eVecs, Ls, Ms, descriptors, gradXs, gradYs = data
        
        # Move data to device
        vertices = to_device(vertices, self.device)
        eVals = to_device(eVals, self.device)
        eVecs = to_device(eVecs, self.device)
        Ls = to_device(Ls, self.device)
        Ms = to_device(Ms, self.device)
        descriptors = to_device(descriptors, self.device)
        gradXs = to_device(gradXs, self.device)
        gradYs = to_device(gradYs, self.device)
        
        # Start timer
        timer.start()
        
        # Test-time refinement
        if self.with_refine > 0:
            for _ in range(self.with_refine):
                self.feed_data(data)
                self.optimize_parameters()
        
        # Forward pass
        inputs = (vertices, eVecs, eVals, Ls, Ms, gradXs, gradYs)
        consistent_bases = self.networks['feature_extractor'](inputs)
        
        if self.attention and 'attention' in self.networks:
            consistent_bases_refined = self.networks['attention'](
                consistent_bases[:, 0], consistent_bases[:, 1])
            consistent_bases = torch.stack(consistent_bases_refined, dim=1)
        
        # Extract features from two shapes
        feat_x = consistent_bases[:, 0]
        feat_y = consistent_bases[:, 1]
        
        # Normalize features
        feat_x = F.normalize(feat_x, dim=-1, p=2)
        feat_y = F.normalize(feat_y, dim=-1, p=2)
        
        # Compute correspondence using nearest neighbor
        dist = torch.cdist(feat_x.squeeze(), feat_y.squeeze())
        p2p = dist.argmin(dim=0)
        
        # Finish timing
        timer.record()
        
        return p2p, None, None  # Return point-to-point map