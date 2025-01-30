


<br />
<br />
<br />

# FedSTaS: Client Stratification and Data Level Sampling for Efficient Federated Learning
ICML 2025 (5135)

## Dependencies
+ Python (>=3.6)
+ PyTorch (>=1.7.1)
+ NumPy (>=1.19.2)
+ Scikit-Learn (>=0.24.1)
+ Scipy (>=1.6.1)

To install all dependencies:
```
pip install -r requirements.txt
```

## Running an experiment

Here we provide the implementation of Stratified Client Selection Scheme along with MNIST and CIFAR-10 dataset. This code takes as input:

- The `dataset` used.
- The data `partition` method used. partition ∈ { iid, dir_{alpha}}
- The `sampling` scheme used. sampling ∈ { ours, comp_grads, dp_comp_grads }
- The percentage of clients sampled `sample_ratio`. We consider 100 clients in all our datasets and use thus sample_ratio=0.1.
- The learning rate `lr` used.
- The batch size `batch_size` used.
- The number of SGD run locally `n_SGD` used.
- The number of rounds of training `n_iter`.
- The number of strata `strata_num` used in ours sampling.
- The learning rate `decay` used after each SGD. We consider no decay in our experiments, decay=1.
- The local loss function regularization parameter `mu`. FedProx with µ = 0 and without systems heterogeneity (no stragglers) corresponds to FedAvg.
- The `seed` used to initialize the training model. We use 0 in all our experiments.
- Force a boolean equal `force` to True when a simulation has already been run but needs to be rerun.
- The privacy parameter `privacy` used in DP sampling (default=3)
- The maximum response value `M` for the Estimator in DP sampling (default=300)
- The desired client ratio `K_desired` for local data sampling (default=0.5)
- The number of strata `d_prime`.
+ To train and evaluate on MNIST:
```


+ To train and evaluate on MNIST (FedSTS):
```
python main_mnist.py --dataset=MNIST \
    --partition=iid \
    --sampling=ours \
    --sample_ratio=0.1 \
    --lr=0.01 \
    --batch_size=64 \
    --n_SGD=20 \
    --n_iter=100 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=True \
    --d_prime=10
```


+ To train and evaluate on MNIST (FedSTaS):
```
python main_mnist.py --dataset=MNIST \
    --partition=dir_0.1 \
    --sampling=comp_grads \
    --sample_ratio=0.1 \
    --lr=0.01 \
    --batch_size=64 \
    --n_SGD=20 \
    --n_iter=100 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=True \
    --K_desired=0.5 \
    --d_prime=10
```

+ To train and evaluate with DP sampling on CIFAR10 (FedSTas):
```
python main_cifar10.py --dataset=CIFAR10 \
    --partition=dir_0.01 \
    --sampling=dp_comp_grads \
    --sample_ratio=0.1 \
    --lr=0.01 \
    --batch_size=64 \
    --n_SGD=20 \
    --n_iter=300 \
    --strata_num=10 \
    --decay=1.0 \
    --mu=0.0 \
    --seed=0 \
    --force=True \
    --privacy=3 \
    --M=300 \
    --K_desired=0.5 \
    --d_prime=10
```

Every experiment saves by default the training loss, the testing accuracy, and the sampled clients at every iteration in the folder `saved_exp_info`. 

```
## Plotting line graphs for training loss and test accuracy

Here we provide the implementation to plot training loss and accuracy line graphs for different plot types along with MNIST and CIFAR-10 dataset. This code takes as input:
- The `plot_type` used. plot_type ∈ { comparison, fedstas_comparison } where comparison compares all models while fedstas_comparison compares the FedSTaS models trained with different privacy budgets, e. 
- The `dataset` used. dataset ∈ { MNIST, CIFAR10 }
- The data `partition` method used. partition ∈ { iid, dir_{alpha} }
- The percentage of clients sampled `sample_ratio`. We consider 100 clients in all our datasets and use thus sample_ratio=0.1.
- The batch size `batch_size` used.
```
To generate training loss and test accuracy line graphs for comparing FedSTaS models trained on MNIST data with varying privacy budgets, e:
```
python main_plots.py --dataset=MNIST \
    --plot_type=fedstas_comparison \
    --partition=dir_0.01 \
    --sample_ratio=0.1 \
    --batch_size=64
```
To generate training loss and test accuracy line graphs for comparing all models trained on CIFAR10 data:
```
python main_plots.py --dataset=CIFAR10 \
    --plot_type=comparison \
    --partition dir_0.1 \
    --sample_ratio 0.1 \
    --batch_size=64
