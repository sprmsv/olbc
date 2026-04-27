<h3 align="center"> <a href="https://arxiv.org/abs/2602.04923"> "Imposing Boundary Conditions on Neural Operators via Learned Function Extensions" </a>  </h3>
<h5 align="center">  Sepehr Mousavi, Siddhartha Mishra, Laura De Lorenzis </h5>
<p align="center"> <img src="assets/compmech.png" height="80"/> <img src="assets/camlab.png" height="70"/> </p>

<p style="color:red"> TODO: add some space between the logos. </p>


<h4 align="center">  Abstract </h4>

<p align="center">  Neural operators have emerged as powerful surrogates for the solution of partial differential equations (PDEs), yet their ability to handle general, highly variable boundary conditions (BCs) remains limited. Existing approaches often fail when the solution operator exhibits strong sensitivity to boundary forcings. We propose a general framework for conditioning neural operators on complex non-homogeneous BCs through function extensions. Our key idea is to map boundary data to latent pseudo-extensions defined over the entire spatial domain, enabling any standard operator learning architecture to consume boundary information. The resulting operator, coupled with an arbitrary domain-to-domain neural operator, can learn rich dependencies on complex BCs and input domain functions at the same time. To benchmark this setting, we construct 18 challenging datasets spanning Poisson, linear elasticity, and hyperelasticity problems, with highly variable, mixed-type, component-wise, and multi-segment BCs on diverse geometries. Our approach achieves state-of-the-art accuracy, outperforming baselines by large margins, while requiring no hyperparameter tuning across datasets. Overall, our results demonstrate that learning boundary-to-domain extensions is an effective and practical strategy for imposing complex BCs in existing neural operator frameworks, enabling accurate and robust scientific machine learning models for a broader range of PDE-governed problems.
 </p>

<hr>


<p align="center"> <img src="assets/architecture.gif" alt="architecture" width="900"/> </p>


<p style="color:red"> TODO: add a short description of the architecture.  </p>

## Sample Results

**Model Estimates:**
<p align="center">
  <img src="assets/estimates/boomcircletri.gif" width="30%" alt="Boom Circle Triangle Estimates"/>
  <img src="assets/estimates/circlehollow.gif" width="30%" alt="Circle Hollow Estimates"/>
  <img src="assets/estimates/squarehollow.gif" width="30%" alt="Square Hollow Estimates"/>
</p>

## Installation

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/sprmsv/olbc.git
cd olbc
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

Follow the instructions in [this Zenodo repository](https://doi.org/10.5281/zenodo.18377370) for downloading the datasets, and organize them in a data directory with the following structure:

```
.../
  |__ poisson-circle-bc1/
      |__ train.nc
      |__ test.nc
  |__ poisson-square-bc4/
      |__ train.nc
      |__ test.nc
  |__ elasticity-circlehollow-m1/
      |__ train.nc
      |__ test.nc
  |__ ...
```

Each dataset directory contains HDF5 and NetCDF4 files for training, validation, and testing.

## Quick Start

### Training

Train a neural operator on a dataset:

```bash
python -m ol.train --datadir <path/to/data> --datapath <dataset/name> --epochs 100 --n_train 512 --n_valid 64
```

**Key Arguments:**
- `--datadir`: Path to the folder containing datasets (required)
- `--datapath`: Relative path inside the data directory (required)
- `--epochs`: Number of training epochs (default: 20)
- `--n_train`: Number of training samples (default: 16)
- `--n_valid`: Number of validation samples (default: 16)
- `--exp`: Experiment name for organizing results (default: '000')
- `--core_name`: Operator architecture - `XRIGNO` or `XGAOT` (default: 'XRIGNO')
- `--batch_size`: Batch size per device (default: 2)

View all available arguments:

```bash
python -m ol.train --help
```

During training, checkpoints and metrics are saved to `./ol/experiments/E<exp>/<datapath>/<timestamp>/`.

### Testing

Evaluate a trained model on test data:

```bash
python -m ol.test --exp <path/to/experiment> --datadir <path/to/data> --datapath <dataset/name>
```

**Key Arguments:**
- `--exp`: Relative path of the experiment (required)
- `--datadir`: Path to the folder containing datasets (required)
- `--datapath`: Relative path inside the data directory (required)
- `--batch_size_per_device`: Batch size for inference (default: 16)

Results (metrics, errors, plots) are saved to `<experiment>/tests/`.

### Key Components

- **`ol.models`**: Neural operator architectures
  - `rigno.py`: Region Interaction Graph Neural Operator
  - `gaot.py`: Graph-based Attention Operator Transformer
  - `common.py`: Base classes and utilities

- **`ol.dataset`**: Dataset loading and preprocessing
  - `dataset.py`: HDF5 I/O and normalization
  - `metadata.py`: Dataset configuration

- **`ol.graph`**: Graph construction
  - `graphbuilder.py`: Multi-scale graph hierarchy
  - `entities.py`: Typed graph representations

- **`ol.metrics`**: Evaluation metrics
  - Lp-norm relative error
  - Chamfer distance for critical point detection
  - Recall at tolerance

## Citation

<p style="color:red"> TODO: update the citation </p>

```bibtex
@inproceedings{mousavi2026imposing,
      title={Imposing Boundary Conditions on Neural Operators via Learned Function Extensions},
      author={Sepehr Mousavi and Siddhartha Mishra and Laura De Lorenzis},
      year={2026},
      eprint={2602.04923},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.04923},
}
```
