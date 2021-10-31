## SELCON: Training Data Subset Selection for Regression with Controlled Generalization Error

#### Authors of the paper: [Durga Sivasubramanian](https://scholar.google.com/citations?user=4JXFWTwAAAAJ&hl=en) (durgas@cse.iitb.ac.in), [Rishabh Iyer](https://sites.google.com/view/rishabhiyer/home?authuser=0/) (rishabh.iyer@utdallas.edu),  [Ganesh Ramakrishnan](https://www.cse.iitb.ac.in/~ganesh/) (ganesh@cse.iitb.ac.in), [Abir De](https://abir-de.github.io) (abir@cse.iitb.ac.in)

#### The code was cleaned and modularized by [Sanidhya Anand](sanidhya.anand03@gmail.com) 
#### [Project Website](https://abir-de.github.io/projects/selcon/selcon.html#)

### Overview
This directory contains code necessary to run the SELCON algorithm which is a data subset selection algorithm for efficient training.
In a nutshell, it aims to select the training subset as well as the model parameters subject to a set of constraints ensuring that the error on validation set remains below an acceptable level. We show that solving the above problem is equivalent to minimizing a monotone and approximate submodular function.

If you use this code in your paper, please use:

	@inproceedings{durga2021training,
        		title={Training Data Subset Selection for Regression with Controlled Generalization Error},
			author={Durga, S and Iyer, Rishabh and Ramakrishnan, Ganesh and De, Abir},
			booktitle={International Conference on Machine Learning},
			pages={9202--9212},
			year={2021},
			 }
### Installation
You can install the package using <br>
`pip install selcon`

To run this code fully, you'll need PyTorch (we're using version 1.4.0) and scikit-learn. We've been running our code in Python 3.7.

### Usage
SELCON package can be utilised in Linear Subset Selection or Deep Subset Selection methods as:

#### SELCON for linear model
```
from SELCON.datasets import load_def_data, get_data
from SELCON.linear import Regression
```
`load_def_data` provides functionality for using the datasets used for the experiments in the paper (provided you have them available in the 'Dataset' directory)
```
reg = Regression()

# Converts specified numpy arrays to torch tensors (assuming data has been split previously)
X_trn, X_val, Y_trn, Y_val = get_data(x_train, x_val, y_train, y_val)
# Trains SELCON model for a subset fraction of 0.03 on the training subset (no fairness)
reg.train_model(X_trn, Y_trn, X_val, Y_val, fraction = 0.03)
# Return optimal subset indices
subset_idxs = reg.return_subset()

# Returns the optimal subset of the training data for further use
X_sub = X_trn[subset_idxs]
y_sub = Y_trn[subset_idxs]
```

#### SELCON for Deep model
```
from SELCON.datasets import load_def_data, get_data
from SELCON.deep import DeepSelection
```
```
reg = DeepSelection()

# Converts specified numpy arrays to torch tensors (assuming data has been split into train-val sets previously)
X_trn, X_val, Y_trn, Y_val = get_data(x_train, x_val, y_train, y_val)
# Trains SELCON model for a subset fraction of 0.03 on the training subset (with fairness)
reg.train_model_fair(X_trn, Y_trn, X_val, Y_val, fraction = 0.03)
# Return optimal subset indices
subset_idxs = reg.return_subset()

# Returns the optimal subset of the training data for further use
X_sub = X_trn[subset_idxs]
y_sub = Y_trn[subset_idxs]
```
