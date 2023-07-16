# surrogate-201
- For my master thesis, Efficient Neural Architecture Generation with an Invertible Neural Network for Neural Architecture Search, is using single nvp.
## Data preprocessing
- The training script will download the preprocessed data automatically. This step can be skipped.
### NAS-Bench-101
- Install NAS-Bench-101 https://github.com/google-research/nasbench
- Query the nb101 data (This step should run on tf1.x environment)
  ```python
  python datasets/query_nb101.py
  ```
- Transform to spektral graph dataset
  ```python
  cd datasets
  python nb101_dataset.py
  ```
- The preprocessed data will be saved in `NasBench101Dataset`
### NAS-Bench-201
- Install NAS-Bench-201 (`pip install nats_bench`)
- Follow this [instruction](https://github.com/D-X-Y/NATS-Bench#preparation-and-download) to download benchmark file, save the file to $TORCH_HOME usually is located in `~/.torch/`. The benchmark file we used is `NATS-tss-v1_0-3ffb9-simple`.
- Query the nb201 data
  ```python
  cd datasets
  python datasets/query_nb201.py
  ```
- Transform to spektral graph dataset
  ```python
  cd datasets
  python nb201_dataset.py 
  ```
- The preprocessed data will be saved in `NasBench201Dataset`
## Singe NVP
### Pre-train
- Modify line `train_phase = [1, 0]` in `trainGAE_two_phase.py`
- `Python trainGAE_two_phase.py`
- Then move the pre-trained model to the path to match the following code in `trainGAE_two_phase.py`
```python
if dataset_name == 'nb101':
    pretrained_weight = 'logs/phase1_nb101_CE_64/modelGAE_weights_phase1'
else:
    pretrained_weight = 'logs/phase1_nb201_CE_64/modelGAE_weights_phase1'
```
### Run experiment for multiple runs
- We can adjust the parameters in `run_GAE_experiment_single.py`
- `python run_GAE_experiment_single.py`

## Aggregate NVP
### Pre-train 
- The same as single NVP
### Run experiment for multiple runs
- run_GAE_experiment.py
  - We can set the following parameters in the file or default parameters will be used
      ```python 
      train_sample_list = [50]
      valid_sample_list = [10]
      budget_list = [192]
      dataset_name = 'nb101' # can be nb101 cifar10-valid cifar100 ImageNet16-120
      ```
- `python run_GAE_experiment.py`
## TSNE Visualization
- Using `trainGAE_for_visualization.py` to record the arch in each iteration
- Using `tsne.ipynb` to visualize the arch. Only need to change `model_dir = ` to the path of the output of `trainGAE_for_visualization`.
## Searching Curve
- Using `search_curve_visualization.py` to plot the search curve