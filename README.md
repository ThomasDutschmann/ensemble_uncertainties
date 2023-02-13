ensemble_uncertainties
==============================
[//]: # (Badges)

Framework to evaluate predictive uncertainties by generating k-fold cross-validation ensembles.


# Project Description
## Evaluation of the Ensemble Variance as an Estimator of Prediction Uncertainty

Especially during the early stages of drug development, molecular property prediction relies heavily on machine learning techniques to guide the emerging experimental design. For a comprehensive evaluation of model quality, it is crucial to outline the limitations of a model beyond its predictive performance.<sup>[1]</sup> While alternative strategies to obtain such predictive uncertainties exist, the construction of ensembles remains the established standard.<sup>[2,3]</sup> However, most uncertainty evaluations consider only a small and well-known selection of datasets, and only one or two different sets of descriptors. Whether the ensemble method produces reasonable uncertainties for virtually any chemoinformatics setup requires a more diversified evaluation. This project aims to automatize the evaluation of ensemble-based uncertainties by generating balanced subsamples of the dataset using k-fold cross-validation.

<img width="1223" alt="cover_picture" src="https://user-images.githubusercontent.com/12691168/152493385-6a37d480-2723-4fd6-b959-670123562d52.png">

# Installation Using conda

## Prerequisites
Anaconda and Git should be installed. If not, check out [Anaconda's website](https://www.anaconda.com) and [Git's website](https://git-scm.com/downloads).

## How To Install

1. Clone repository:
```console
git clone https://git.rz.tu-bs.de/impc/baumannlab/ensemble_uncertainties.git
```

2. Change directory:
```console
cd ensemble_uncertainties/
```

3. Create conda environment:

```console
conda env create -n ensemble_uncertainties -f conda-envs/ensemble_uncertainties.yaml
```

4. Activate environment:

```console
conda activate ensemble_uncertainties
```

5. Install ensemble_uncertainties as a package:
```console
pip install -e .
```

# How To Use
## Running the main executable

The executable provides helpful information of all the (necessary) command line arguments:

```console
python ensemble_uncertainties/run_ensembling.py -h
```

For example, if you want to evaluate support vector regression on the provided Tetrahymena toxicity data set<sup>[4]</sup> (first 100 entries, featurized by RDKit descriptors) with 5 repetitions and a 5-fold, storing the output in my_test_results_folder/, you run the executable like so:

```console
python ensemble_uncertainties/run_ensembling.py -r 5 -n 5 -x test_data/tetrahymena/tetrah_rdkit.csv -y test_data/tetrahymena/tetrah_y.csv -m svm_rbf -t regression -o my_test_results_folder/ -v
```

To test the classification case, the first 100 entries of the CYP1A2 dataset from the applicability domain study by Klingspohn et al. is provided.<sup>[5]</sup>

Note: It is important that the CSV-files are semicolon-separated, have a header, have an index column named "id" at the first position and that the output values in the y-files are in a column named "y". See the provided test files in test_data/.


## Development using ensemble_uncertainties

Parts of the framework can also be used inside Python to conveniently estimate prediction uncertainties. Consider the provided example script below. Again, be reminded that external files need to provide the required format described above. Otherwise, they have to be converted.

```python
# Run inside ensemble_uncertainties/ for the data paths to work

import ensemble_uncertainties as eu

from sklearn.svm import SVR


# Load data
tetrah_rdkit_path = 'test_data/tetrahymena/tetrah_rdkit.csv'
tetrah_y_path = 'test_data/tetrahymena/tetrah_y.csv'
X, y = eu.load_data(tetrah_rdkit_path, tetrah_y_path)

# Set evaluator using some custom settings
svr_evaluator = eu.RegressionEvaluator(
    SVR(),
    repetitions=5,
    n_splits=5,
    verbose=False
)

# Run Evaluator
svr_evaluator.perform(X, y)

# Get results
pred_quality = svr_evaluator.test_ensemble_quality
auco = svr_evaluator.auco
rho = svr_evaluator.rho

# Print results
print(f'R^2:             {pred_quality:.3f}')
print(f'AUCO:            {auco:.3f}')
print(f"Spearman's rho:  {rho:.3f}")


# Expected output:
# R^2:             0.829
# AUCO:            17.464
# Spearman's rho:  0.345
```


### References

[1] A. Tropsha, A. Golbraikh, Curr. Pharm. Des. 2007, 13, 3494–3504.

[2] L. Hirschfeld, K. Swanson, K. Yang, R. Barzilay, C. W. Coley, J. Chem. Inf. Model. 2020, 60, 3770–3780.

[3] T.-M. Dutschmann, K. Baumann, Molecules 2021, 26, 6.

[4] F. Cheng, J. Shen, Y. Yu, W. Li, G. Liu, P. W. Lee, Y. Tang, Chemosphere 2011, 82, 1636–1643.

[5] W. Klingspohn, M. Mathea, A. ter Laak, N. Heinrich, K. Baumann, J. Cheminform. 2017, 9, 44.
