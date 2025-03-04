import datetime
import os
import wandb
import torch
from .dist_util import master_only

class WandbLogger:
    """Initialize wandb logger.
    
    Args:
        opt (dict): Config dict. It should contain:
            name (str): Experiment name.
            project (str, optional): Project name in wandb.
            entity (str, optional): Entity name in wandb.
            resume (bool, optional): Resume wandb session. Default: False.
            id (str, optional): Experiment id in wandb session. Default: None.
            tags (list, optional): List of tags to be added to experiment. Default: None.
            config (dict, optional): Configuration to be added to experiment.
        start_iter (int, optional): Start step. Default: 1.
    """

    def __init__(self, opt, start_iter=1):
        self.exp_name = opt['name']
        self.start_iter = start_iter
        
        # get wandb_opt
        wandb_opt = opt.get('wandb', {})
        wandb_project = wandb_opt.get('project', 'spectral-shape-matching')
        wandb_entity = wandb_opt.get('entity', None)
        wandb_run_id = wandb_opt.get('id', None)
        wandb_resume = wandb_opt.get('resume', False)
        wandb_tags = wandb_opt.get('tags', None)
        
        # init wandb
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            id=wandb_run_id,
            name=self.exp_name,
            resume=wandb_resume,
            tags=wandb_tags,
            config=opt,
            dir=os.path.join(opt['path']['experiments_root'], 'wandb')
        )
        
        # Set up environment
        os.makedirs(os.path.join(opt['path']['experiments_root'], 'wandb'), exist_ok=True)
        self.wandb = wandb

    @master_only
    def log_metrics(self, log_dict, step=None):
        """Log training metrics to wandb.
        
        Args:
            log_dict (dict): Dict contains metrics to be logged.
            step (int, optional): Step number. Default: None.
        """
        # Remove epoch, iter, etc. from log_dict
        clean_log_dict = log_dict.copy()
        for k in ['epoch', 'iter', 'lrs', 'time', 'data_time']:
            if k in clean_log_dict:
                clean_log_dict.pop(k)
                
        # Log to wandb
        self.wandb.log(clean_log_dict, step=step)
        
    @master_only
    def log_validation_metrics(self, log_dict, step=None):
        """Log validation metrics to wandb.
        
        Args:
            log_dict (dict): Dict contains metrics to be logged.
            step (int, optional): Step number. Default: None.
        """
        # Add prefix 'val/' to all metrics
        val_log_dict = {f'val/{k}': v for k, v in log_dict.items()}
        
        # Log to wandb
        self.wandb.log(val_log_dict, step=step)
        
    @master_only
    def log_image(self, key, img, step=None, dataformats='HWC'):
        """Log image to wandb.
        
        Args:
            key (str): Image key/name.
            img (np.array): Image to be logged.
            step (int, optional): Step number. Default: None.
            dataformats (str, optional): Image data format. Default: 'HWC'.
        """
        self.wandb.log({key: wandb.Image(img, caption=key)}, step=step)
        
    @master_only
    def log_images(self, image_dict, step=None):
        """Log multiple images to wandb.
        
        Args:
            image_dict (dict): {key: img} dict.
            step (int, optional): Step number. Default: None.
        """
        log_dict = {k: wandb.Image(v, caption=k) for k, v in image_dict.items()}
        self.wandb.log(log_dict, step=step)
        
    @master_only
    def log_checkpoint(self, filename, step=None):
        """Log checkpoint to wandb.
        
        Args:
            filename (str): Checkpoint filename.
            step (int, optional): Step number. Default: None.
        """
        artifact = self.wandb.Artifact(
            name=f"checkpoint-{step}",
            type="model", 
            description=f"Model checkpoint at step {step}"
        )
        artifact.add_file(filename)
        self.wandb.log_artifact(artifact)
        
    @master_only
    def log_plot(self, key, figure, step=None):
        """Log matplotlib figure to wandb.
        
        Args:
            key (str): Figure key/name.
            figure (matplotlib.figure.Figure): Figure to be logged.
            step (int, optional): Step number. Default: None.
        """
        self.wandb.log({key: wandb.Image(figure)}, step=step)
        
    @master_only
    def close(self):
        """Close wandb session."""
        self.wandb.finish()