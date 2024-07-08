# PCA
 The `PCA_functions.py` file contains Python functions that can help you do PCA and create plots from PCA data quickly.
 The supported plots are scores plots, scree plots, and loadings plots. The functions also include preprocessing using the SNV (Standard Nomal Variate) method.
 The `PCA_code.ipynb` file illustrates how the functions are used using UV/vis data from whisky samples. The data was obtained from a 3rd year undergraduate chemistry laboratory carried out by me at the University of Edinburgh.

You can import the funcions by having the `PCA_functions.py` in the same directory as your code and import it using `import PCA_functions.py`. In order to streamline your coding, you can import the file as `import PCA_functions.py as pcf`.

For the scripts to work, you will to have the following libraries installed in your Python environment:
- NumPy
- Matplotlib
- scikit-learn
- SciPy

**N.B.**
The data must be in the following format:
| Data Points Names | Variable 1    | Variable 2    | ...           |
|-------------------|---------------|---------------|---------------|
| Data Point Name 1 | ####          | ####          | ...           |
| Data Point Name 2 | ####          | ####          | ...           |
| ...               | ...           | ...           | ...           |

Should it not be, you should transpose the data (for `np.ndarray`, you can use `array.T`), since the scripts were made to work on variables column-wise.

## Standard Nomal Variate (SNV)
the data to be used in PCA must first be pre-processed, as one single variable might contribute much more to the variance than other variables thus causing the results to be artificially skewed towards this variable. One such pre-processing technique is the Standard Normal Variate (SNV) method, which centers the data around the zero rather than around the mean and each data point is such that its standard deviation σ~i~ is 1. This is done by subtracting the mean of all values of a certain variable and dividing these centered values by the standard deviation.

## Scores Plots
Scores plots show can show clusters of data points by plotting the data points using typically two principal components (PCs).

## Scree Plots
Scree plots show the variance explained by the single PCs.

## Loadings Plots
Loadings plots are useful in showing how each variable contriubutes to and correlates with the single PCs.
