ensemble_uncertainties
==============================
[//]: # (Badges)

Framework to evaluate predictive uncertainties by generating k-fold cross-validation ensembles.


# Project description
## Evaluation of the Ensemble Variance as an Estimator of Prediction Uncertainty

Especially during the early stages of drug development, molecular property prediction relies heavily on machine learning techniques to guide the emerging experimental design. For a comprehensive evaluation of model quality, it is crucial to outline the limitations of a model beyond its predictive performance.<sup>[1]</sup> While alternative strategies to obtain such predictive uncertainties exist, the construction of ensembles remains the established standard.<sup>[2,3]</sup> However, most uncertainty evaluations consider only a small and well-known selection of datasets, and only one or two different sets of descriptors. Whether the ensemble method produces reasonable uncertainties for virtually any chemoinformatics setup requires a more diversified evaluation. This project aims at automatizing the evaluation of ensemble-based uncertainties by generating balanced subsamples of the dataset using k-fold cross-validation.

<img width="973" alt="cover_picture" src="https://user-images.githubusercontent.com/12691168/152394859-37bee0da-9033-41ca-9376-b87ae42f0721.png">

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
conda env create -n ensemble_uncertainties -f conda-envs/ensemble_uncertainties.yaml
```

4. Activate the environment:

```console
conda activate ensemble_uncertainties
```

5. Install it as a package:
```console
pip install -e .
```

# How To Use
## Running an evaluation

The executable provides helpful information of all the (necessary) command line arguments:

```console
python ensemble_uncertainties/exectuable.py -h
```

For example, if you want to evaluate support vector regression on the provided Tetrahymena toxicity data set<sup>[4]</sup> featurized by RDKit descriptors with 5 repetitions and a 5-fold, storing the output in my_test_results_folder/, you run the executable like so:

```console
python ensemble_uncertainties/executable.py -r 5 -n 5 -x test_data/tetrahymena/tetrah_X.csv -y test_data/tetrahymena/tetrah_y.csv -m SVM -t regression -o my_test_results_folder/ -v
```

To test the classification case, the CYP1A2 dataset from the applicability domain study by Klingspohn et al. is provided.<sup>[5]</sup>

Note: It is important that the CSV-files are semicolon-separated, have a header, have an index column named "id" at the first position and that the output values in the y-files are in a column named "y". See the provided test files in test_data/.
Furthermore, when using the default environment, the only available version of TensorFlow is the one with GPU support.


### References

[1] A. Tropsha, A. Golbraikh, Curr. Pharm. Des. 2007, 13, 3494–3504.

[2] L. Hirschfeld, K. Swanson, K. Yang, R. Barzilay, C. W. Coley, J. Chem. Inf. Model. 2020, 60, 3770–3780.

[3] T.-M. Dutschmann, K. Baumann, Molecules 2021, 26, 6.

[4] F. Cheng, J. Shen, Y. Yu, W. Li, G. Liu, P. W. Lee, Y. Tang, Chemosphere 2011, 82, 1636–1643.

[5] W. Klingspohn, M. Mathea, M., A. ter Laak, N. Heinrich, K. Baumann, J. Cheminform. 2017, 9, 44.
