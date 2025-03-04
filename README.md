## [Unsupervised Learning of Robust Spectral Shape Matching (SIGGRAPH/TOG 2023)](https://dongliangcao.github.io/urssm/)
![img](figures/teaser.jpg)

## Installation
```bash 
conda create -n fmnet python=3.8 # create new viertual environment
conda activate fmnet
conda install pytorch cudatoolkit -c pytorch # install pytorch
pip install -r requirements.txt # install other necessary libraries via pip
```

## Dataset
To train and test datasets used in this paper, please download the datasets from the this [link](https://drive.google.com/file/d/1zbBs3NjUIBBmVebw38MC1nhu_Tpgn1gr/view?usp=share_link) and put all datasets under ../data/
```Shell
├── data
    ├── FAUST_r
    ├── FAUST_a
    ├── SCAPE_r
    ├── SCAPE_a
    ├── SHREC19_r
    ├── TOPKIDS
    ├── SMAL_r
    ├── DT4D_r
    ├── SHREC20
    ├── SHREC16
    ├── SHREC16_test
```
We thank the original dataset providers for their contributions to the shape analysis community, and that all credits should go to the original authors.

## Data preparation
For data preprocessing, we provide *[preprocess.py](preprocess.py)* to compute all things we need.
Here is an example for FAUST_r.
```python
python preprocess.py --data_root ../data/FAUST_r/ --no_normalize --n_eig 200
```

## Train
To train the model on a specified dataset.
```python
python train.py --opt options/train/faust.yaml 
```
You can visualize the training process in tensorboard.
```bash
tensorboard --logdir experiments/
```

## Test
To test the model on a specified dataset.
```python
python test.py --opt options/test/faust.yaml 
```
The qualitative and quantitative results will be saved in [results](results) folder.

## Texture Transfer
An example of texture transfer is provided in *[texture_transfer.py](texture_transfer.py)*
```python
python texture_transfer.py
```

## Pretrained models
You can find all pre-trained models in [checkpoints](checkpoints) for reproducibility.

## Partial Shape Matching on SHREC’16
There were two issues with the partial shape matching experiments on the SHREC'16 dataset related to the training/test splits [Bracha et al. 2023, Ehm et al. 2024]. Below, we provide additional evaluations that substantiate our claims:
| Geo err (x100)    | CUTS on CUTS | CUTS on HOLES | HOLES on CUTS | HOLES on HOLES | CUTS on CUTS'24* | HOLES on CUTS'24* |
| ----------------  | :----------: | :------------:|:------------: |:------------:  | :---------------:|:-----------------:|
| Ours original**   |   3.3        | 13.7          |5.2            |  9.1           | 3.4              |5.5                |
| Ours new***       |   3.2        | 13.5          |5.6            |  8.2           | 3.2              |5.9                |

\*   CUTS'24 refers to the new test split from [Ehm et al. 2024], the split can be found [here](https://github.com/vikiehm/geometrically-consistent-partial-partial-shape-matching/tree/main/CUTS24).

**  Pretrained on TOSCA

*** Pretrained on FAUST + SCAPE + SMAL + DT4D-H, 25 test-time adaptation iterations

[[Bracha et al. 2023] A. Bracha, T. Dages, R. Kimmel, On Partial Shape Correspondence and Functional Maps, arXiv 2023](https://arxiv.org/abs/2310.14692).

[[Ehm et al. 2024] V. Ehm, M. Gao, P. Roetzer, M. Eisenberger, D. Cremers, F. Bernard, Partial-to-Partial Shape Matching with Geometric Consistency, CVPR 2024](https://arxiv.org/abs/2404.12209).


## Acknowledgement
The implementation of DiffusionNet is based on [the official implementation](https://github.com/nmwsharp/diffusion-net).

The framework implementation is adapted from [Unsupervised Deep Multi Shape Matching](https://github.com/dongliangcao/Unsupervised-Deep-Multi-Shape-Matching).

# Weights & Biases Integration

This project now includes Weights & Biases (wandb) integration for experiment tracking, metrics visualization, and model checkpointing.

## Features

- Automatic logging of training losses
- Validation metrics tracking
- Checkpoint saving and versioning
- Experiment organization with projects, tags, and run names
- Hyperparameter tracking and comparison

## Setup

1. Install the wandb package (already added to requirements.txt):
```bash
pip install wandb
```

2. Sign up for a wandb account at [wandb.ai](https://wandb.ai) if you don't have one

3. Log in to wandb from the command line:
```bash
wandb login
```

## Usage

### Enabling wandb in Config Files

Add a `wandb` section to your configuration YAML files to enable wandb logging:

```yaml
# Enable wandb logging
wandb:
  use_wandb: true
  project: "spectral-shape-matching"  # Your project name
  entity: null  # Optional: your wandb username or team name
  id: null  # Optional: for resuming runs
  resume: false  # Optional: resume wandb logging
  tags: ["faust", "training"]  # Optional: tags for the run
```

See `options/train/wandb_config_example.yaml` for a complete example.

### Logged Metrics

The following metrics are automatically logged:

- **Training Metrics**:
  - All loss terms (total loss, regularization losses, etc.)
  - Learning rates
  - Training time metrics
  
- **Validation Metrics**:
  - Average geodesic error
  - AUC of the PCK curve
  - Other validation metrics as defined in the model

- **Testing Metrics**:
  - Same metrics as validation, but tagged with test dataset name

### Model Checkpoints

The integration automatically saves model checkpoints to wandb when they are saved locally. This allows for easy tracking of model versions and their corresponding metrics.

### Visualization

In the wandb UI, you can:
- Compare runs with different hyperparameters
- Visualize loss curves and validation metrics
- Track system resource usage
- View model architecture and hyperparameters

## Disabling wandb

To disable wandb logging, either:

1. Remove the `wandb` section from your config file, or
2. Set `use_wandb: false` in the `wandb` section

## Offline Mode

If you want to log metrics without an internet connection:

```bash
wandb offline
```

Later, when you have internet access, you can sync your runs:

```bash
wandb sync [WANDB_DIR]
```

## Additional Resources

- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [W&B Python SDK Reference](https://docs.wandb.ai/ref/python)