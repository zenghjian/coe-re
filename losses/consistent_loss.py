import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class ConsistentLoss(nn.Module):
    """
    Consistent loss for CoE model
    """
    def __init__(self, mu_off=1, mu_pos=50, mu_ortho=1000, A_ortho=True):
        super().__init__()
        self.mu_off = mu_off
        self.mu_pos = mu_pos
        self.mu_ortho = mu_ortho
        self.A_ortho = A_ortho
        
    def forward(self, evals, consistent_bases, Ls, Ms, descriptors):
        """
        Compute the consistent loss
        
        Args:
            evals (torch.Tensor): Eigenvalues [B, 2, K]
            consistent_bases (torch.Tensor): Consistent basis [B, 2, N, K]
            Ls (torch.Tensor): Laplacian matrices [B, 2, N, N]
            Ms (torch.Tensor): Mass matrices [B, 2, N]
            descriptors (torch.Tensor): Shape descriptors [B, 2, N, D]
        """
        loss_components = {}
        
        # Off-diagonality penalty loss
        if self.mu_off > 0:
            off_penalty_loss = self._off_penalty_loss(consistent_bases[:, 0], Ls[:, 0], evals[:, 0]) + \
                               self._off_penalty_loss(consistent_bases[:, 1], Ls[:, 1], evals[:, 1])
            loss_components['off_penalty_loss'] = off_penalty_loss * self.mu_off
        else:
            loss_components['off_penalty_loss'] = torch.tensor(0.0, device=evals.device)
            
        # Contrastive loss for embedding consistency
        if self.mu_pos > 0:
            contrastive_loss = self._pos_contrastive_loss(
                descriptors[:, 0], Ms[:, 0], consistent_bases[:, 0],
                descriptors[:, 1], Ms[:, 1], consistent_bases[:, 1]
            )
            loss_components['pos_contrastive_loss'] = contrastive_loss * self.mu_pos
        else:
            loss_components['pos_contrastive_loss'] = torch.tensor(0.0, device=evals.device)
            
        # Orthogonality loss
        if self.mu_ortho > 0:
            ortho_loss = self._ortho_loss(consistent_bases[:, 0], Ms[:, 0]) + \
                         self._ortho_loss(consistent_bases[:, 1], Ms[:, 1])
            loss_components['ortho_loss'] = ortho_loss * self.mu_ortho
        else:
            loss_components['ortho_loss'] = torch.tensor(0.0, device=evals.device)
            
        # Compute total loss
        total_loss = sum(loss_components.values())
        
        return total_loss, loss_components
        
    def _off_penalty_loss(self, consistent_bases, Ls, evals):
        """
        Compute off-diagonality penalty loss
        
        Args:
            consistent_bases (torch.Tensor): Consistent basis [B, V, K]
            Ls (torch.Tensor): Laplacian matrix [B, V, V]
            evals (torch.Tensor): Eigenvalues [B, K]
        """
        # Transpose basis: [B, K, V]
        PhisT = torch.transpose(consistent_bases, 1, 2)
        
        # PhisT * L: [B, K, V]
        PhisTL = torch.bmm(PhisT, Ls)
        
        # PhisT * L * Phis: [B, K, K]
        PhisTLPhis = torch.bmm(PhisTL, consistent_bases)
        
        # Create diagonal matrix from eigenvalues: [B, K, K]
        evals_diag = torch.diag_embed(evals)
        
        # Compute Frobenius norm of the difference
        off_penalty = torch.linalg.matrix_norm(PhisTLPhis - evals_diag).sum()
        
        return off_penalty
        
    def _pos_contrastive_loss(self, descriptors_x, Ms_x, consistent_bases_x, 
                              descriptors_y, Ms_y, consistent_bases_y):
        """
        Compute contrastive loss for embedding consistency
        """
        # Project descriptors to frequency domain
        proj_x = self._frequency_domain_projection(descriptors_x, Ms_x, consistent_bases_x)
        proj_y = self._frequency_domain_projection(descriptors_y, Ms_y, consistent_bases_y)
        
        # Compute Frobenius norm of the difference
        contrastive_loss = torch.linalg.matrix_norm(proj_x - proj_y).sum()
        
        return contrastive_loss
        
    def _ortho_loss(self, consistent_bases, Ms):
        """
        Compute orthogonality loss
        """
        batch_size = consistent_bases.shape[0]
        k = consistent_bases.shape[2]
        I_k = torch.eye(k, device=consistent_bases.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Transpose basis: [B, K, V]
        Phi_transpose = consistent_bases.transpose(1, 2)
        
        # Create diagonal mass matrix: [B, V, V]
        M = torch.diag_embed(Ms)
        
        # Compute Phi^T * M * Phi - I: [B, K, K]
        ortho_diff = torch.bmm(torch.bmm(Phi_transpose, M), consistent_bases) - I_k
        
        # Compute Frobenius norm
        ortho_loss = torch.linalg.matrix_norm(ortho_diff).sum()
        
        return ortho_loss
        
    def _frequency_domain_projection(self, descriptors, Ms, consistent_bases):
        """
        Project descriptors to frequency domain
        """
        if self.A_ortho:
            # Create diagonal mass matrix: [B, V, V]
            Ms_diag = torch.diag_embed(Ms)
            
            # Weighted consistent basis: [B, V, K]
            weighted_cb = torch.bmm(Ms_diag, consistent_bases)
            
            # Transpose descriptors: [B, D, V]
            descriptorsT = torch.transpose(descriptors, 1, 2)
            
            # Project to frequency domain: [B, D, K]
            projection = torch.bmm(descriptorsT, weighted_cb)
        else:
            # Transpose descriptors: [B, D, V]
            descriptorsT = torch.transpose(descriptors, 1, 2)
            
            # Project to frequency domain: [B, D, K]
            projection = torch.bmm(descriptorsT, consistent_bases)
            
        return projection