import torch

def pad_data(data, max_length, padding_value=0):
    """
    Pad data to the maximum length, only along the vertex dimension (second dimension)
    
    Args:
        data (torch.Tensor): Data to be padded, possible shapes:
                             (1, n_vertices, 3) - vertex coordinates
                             (1, n_vertices, k) - features or eigenvectors
                             (1, n_vertices, n_vertices) - matrices (e.g., Laplacian)
                             (1, n_vertices) - vectors (e.g., mass vector)
        max_length (int): Target length for the vertex dimension
        padding_value (float): Value used for padding, default is 0
        
    Returns:
        torch.Tensor: Data padded along the vertex dimension
    """
    # If already long enough, return directly
    if data.shape[1] >= max_length:
        return data
    
    # Calculate padding size
    pad_size = max_length - data.shape[1]
    
    if len(data.shape) == 2:  # Shape (1, n_vertices)
        # Pad along second dimension
        padded_data = torch.cat([
            data, 
            torch.full((data.shape[0], pad_size), padding_value, 
                      dtype=data.dtype, device=data.device)
        ], dim=1)
        
    elif len(data.shape) == 3 and data.shape[2] != data.shape[1]:  # Shape (1, n_vertices, 3) or (1, n_vertices, k)
        # Pad along second dimension
        padded_data = torch.cat([
            data, 
            torch.full((data.shape[0], pad_size, data.shape[2]), padding_value,
                      dtype=data.dtype, device=data.device)
        ], dim=1)
        
    elif len(data.shape) == 3 and data.shape[2] == data.shape[1]:  # Shape (1, n_vertices, n_vertices)
        # Create zero tensor of target size
        padded_data = torch.full((data.shape[0], max_length, max_length), padding_value,
                                dtype=data.dtype, device=data.device)
        # Copy original data to top-left corner
        padded_data[:, :data.shape[1], :data.shape[2]] = data
        
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    
    return padded_data

