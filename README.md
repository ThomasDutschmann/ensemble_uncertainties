ensemble_uncertainties
==============================
[//]: # (Badges)

Framework to evaluate predictive uncertainties by generating k-fold cross-validation ensembles. Currently under construction!


# Project description
## Evaluation of the Ensemble Variance as an Estimator of Prediction Uncertainty

Especially during the early stages of drug development, molecular property prediction relies heavily on machine learning techniques to guide the emerging experimental design. For a comprehensive evaluation of model quality, it is crucial to outline the limitations of a model beyond its predictive performance. While alternative strategies to obtain such predictive uncertainties exist, the construction of ensembles remains the established standard. However, most uncertainty evaluations consider only a small and well-known selection of datasets, and only one or two different sets of descriptors. Whether the ensemble method produces reasonable uncertainties for virtually any chemoinformatics setup requires a more diversified evaluation. This project aims at automatizing the evaluation of ensemble-based uncertainties by generating balanced subsamples of the dataset using k-fold cross-validation.


# Installation using conda

## Prerequisites
Anaconda and Git should be installed. See [Anaconda's website](https://www.anaconda.com) and [Git's website](https://git-scm.com/downloads) for download.

## How to install

1. Clone repository:
```console
git clone https://github.com/ThomasDutschmann/ensemble_uncertainties.git
```

2. Change directory:
```console
cd ensemble_uncertainties/
```

3. Create conda environment:

```console
conda env create -n ensemble_uncertainties -f conda-envs/ensemble_uncertainties_env.yaml
```

4. Activate the environment:

```console
conda activate ensemble_uncertainties_env
```

5. Install the maxsmi package:
```console
pip install -e .
```

# How To Use
## Examples
### Running an evaluation

The exectuable provides helpful information of all the (necessary) command line arguments:

```console
python ensemble_uncertainties/exectuable.py -h
```
