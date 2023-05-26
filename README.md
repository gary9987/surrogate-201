# surrogate-201
## Train aggregate
- trainGAE_ensemble.py
## Run experiment for multiple runs
- run_GAE_experiment.py
  - We need to set the following parameters in the file
      ```python 
      train_sample_list = [50]
      valid_sample_list = [10]
      budget_list = [192]
      dataset_name = 'nb101' # can be nb101 cifar10-valid cifar100 ImageNet16-120
      ```