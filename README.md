# Fair-data-selection

## To run the gradient descent code

To run the code just use `python3 run_fair_subset.py`. In **run_fair_subset.py**, please set the datadir(location of the datasets), dataset name(as used in utils/custom_datasets), fraction and other parameters. 

## Description of the gradient descent code

- **Subset_Whole_Val.py** (which is called by **run_fair_subset.py**) has two functions, *train_model_fair* and *train_model*. Former uses the validation set constraints and latter doesn't.

- **Subset_Whole_Val.py** implements the Algorithm 1 in the overleaf in the function *train_model_fair*

- Both *train_model_fair* and *train_model* and are capable of running different variants such as training with full, random and specific subset. This can be achieved by appropriately providing the type and the intial subset.

- When type is *'Fair_subset'*, two function of **FindSubset_Vect** class from **model/Find_Fair_Subset.py** is invoked namely, *precompute* and *return_subset*.

- *precompute* implements vectorised version of lines 2-5 in Algorithm 1 (Algorithm2) and *return_subset* implements vectorised version of lines 9-12 in Algorithm 1 (Algorithm3).


## For CVX
Please use the tradeoff.py file in test directory to run CVX solver.

To run experiments on synthetic data:
1. Set data_name = "synthetic" in line no.52
2. Set sigma values in line 506 and 508 in utils/custom_dataset.py

