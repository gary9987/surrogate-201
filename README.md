# surrogate-201
- For my master thesis, Efficient Neural Architecture Generation with an Invertible Neural Network for Neural Architecture Search, is using single nvp.
## Requirement Package
- python==3.8
- tensorflow==2.10.0
- spektral==1.2.0
- wget
- nats_bench
- matplotlib
### Conda env for example
```
conda create -n tf2 python==3.8
conda activate tf2
pip install tensorflow==2.10
pip install spektral==1.2.0
pip install wget
pip install nats_bench
pip install matplotlib
```
## Data preprocessing
- The training script will download the preprocessed data automatically. This step can be skipped.
- The preprocessed data have a copy in CML server `/project/n/gychen/`
### NAS-Bench-101
- Data preprocessing for NAS-Bench-101 is required tf1.x environment. We use conda for example.
- Set up tf1.x env (conda env as an example)
  ```
  conda create -n tf1.15 python==3.7
  conda activate tf1.15
  pip install tensorflow==1.15
  ```
- Install NAS-Bench-101 https://github.com/google-research/nasbench
  ```bash
  git clone https://github.com/google-research/nasbench
  cd nasbench
  pip install -e .
  ```
- Query the nb101 data (This step should run on tf1.x environment)
  ```python
  cd datasets
  python query_nb101.py
  ```
- Transform to spektral graph dataset (This step should run on tf2.x environment)
  ```python
  export PYTHONPATH=$PWD
  python datasets/nb101_dataset.py
  ```
- The preprocessed data will be saved in `NasBench101Dataset`
### NAS-Bench-201
- Install NAS-Bench-201 (`pip install nats_bench`)
- Follow this [instruction](https://github.com/D-X-Y/NATS-Bench#preparation-and-download) to download benchmark file, save the file to $TORCH_HOME usually is located in `~/.torch/`. The benchmark file we used is `NATS-tss-v1_0-3ffb9-simple`.
- Query the nb201 data
  ```python
  cd datasets
  python query_nb201.py
  ```
- Transform to spektral graph dataset
  ```python
  cd datasets
  python nb201_dataset.py 
  ```
- The preprocessed data will be saved in `NasBench201Dataset`
## Single NVP
### Pre-train
- Modify line `train_phase = ` to  `train_phase = [1, 0]` in `trainGAE_two_phase.py`
- For NAS-Bench-101: `Python trainGAE_two_phase.py --dataset nb101`
- For NAS-Bench-201: `Python trainGAE_two_phase.py`
- Then move the pre-trained model to the path to match the following code in `trainGAE_two_phase.py`
```python
if dataset_name == 'nb101':
    pretrained_weight = 'logs/phase1_nb101_CE_64/modelGAE_weights_phase1'
else:
    pretrained_weight = 'logs/phase1_nb201_CE_64/modelGAE_weights_phase1'
```
### Run experiment for multiple runs
- Modify line `train_phase = ` to `train_phase = [0, 1]` in `trainGAE_two_phase.py`
- We can adjust the parameters in `run_GAE_experiment_single.py`
- RUN `python run_GAE_experiment_single.py`

## Aggregate NVP
### Pre-train 
- The same as single NVP
### Run experiment for multiple runs
- Modify line `train_phase = ` to `train_phase = [0, 1]` in `trainGAE_ensemble.py`
- run_GAE_experiment.py
  - We can set the following parameters in the file or default parameters will be used
      ```python 
      train_sample_list = [50]
      valid_sample_list = [10]
      budget_list = [192]
      dataset_name = 'nb101' # can be nb101 cifar10-valid cifar100 ImageNet16-120
      ```
- RUN `python run_GAE_experiment.py`
## TSNE Visualization
- Using `trainGAE_for_visualization.py` to record the arch in each iteration
- Using `tsne.ipynb` to visualize the arch. Only need to change `model_dir = ` to the path of the output of `trainGAE_for_visualization`.
## Evaluation Results and Searching Curve
- The pickle record file will in the `InvertNAS/yyyymmdd-hhmmsstopk_finetuneFalse_rfinetuneFalse_rankTrue_randomSFalse_ensemble_2NN_4*5*256` folder.
- Using the function `plot_search_curve` in `search_curve_visualization.py` and pass the file path of pickle record to plot the searching curve and show the NAS results on each query budget.
- RUN `python search_curve_visualization.py`
## Acknowledgement
Code base from
- [NAS-Bench-101](https://github.com/google-research/nasbench)
- [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201)
- [NATS-Bench (NAS-Bench-201)](https://github.com/D-X-Y/NATS-Bench)
- [Naszilla](https://github.com/naszilla/naszilla)
- [NASLib](https://github.com/automl/NASLib)
- [jaekookang/invertible_neural_networks](https://github.com/jaekookang/invertible_neural_networks)