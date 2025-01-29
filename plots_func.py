import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import dirichlet

def load_pkl(metric, file_name):
    """Load pickle file containing training history."""
    with open(f"saved_exp_info/{metric}/{file_name}.pkl", "rb") as f:
        return pickle.load(f)

def plot_dirichlet_distribution(alpha, dataset_sizes, dataset='MNIST'):
    """Plot Dirichlet distribution visualization."""
    if len(dataset_sizes) == 0:
        print("Warning: No dataset sizes provided for Dirichlet distribution plot")
        return
        
    plt.figure(figsize=(12, 6))
    x = range(len(dataset_sizes))
    plt.bar(x, dataset_sizes, alpha=0.7, label='Client Data Size')
    plt.title(f'Dirichlet Distribution (α={alpha})')
    plt.xlabel('Client Index')
    plt.ylabel('Dataset Size')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_dirichlet_alpha{alpha}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_stratification_results(strata_sizes, strata_assignments, alpha, dataset='MNIST'):
    """Plot stratification results."""
    if not strata_sizes or not strata_assignments:
        print("Warning: No stratification data provided")
        return
        
    plt.figure(figsize=(15, 5))
    
    # Plot stratum sizes
    plt.subplot(1, 2, 1)
    x = range(len(strata_sizes))
    plt.bar(x, strata_sizes, alpha=0.7, label='Stratum Size')
    plt.title(f'Stratum Sizes (α={alpha})')
    plt.xlabel('Stratum Index')
    plt.ylabel('Number of Clients')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_stratification_alpha{alpha}.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_algorithm_comparison(results, alpha, q, dataset='MNIST'):
    """Plot algorithm comparison for given α and q values."""
    plt.figure(figsize=(15, 5))
    
    # Plot training loss
    plt.subplot(1, 2, 1)
    for method, data in results.items():
        if 'train_loss' in data and len(data['train_loss']) > 0:
            plt.plot(data['train_loss'], '-', linewidth=2, label=method)
    alpha_label = f"α={alpha}" if alpha != "N/A" else "IID/Shard"
    plt.title(f'Training Loss ({alpha_label}, q={q})')
    plt.xlabel('Round')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend(loc='upper right')
    
    # Plot test accuracy
    plt.subplot(1, 2, 2)
    for method, data in results.items():
        if 'test_acc' in data and len(data['test_acc']) > 0:
            plt.plot(data['test_acc'], '-', linewidth=2, label=method)
    plt.title(f'Test Accuracy ({alpha_label}, q={q})')
    plt.xlabel('Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'plots/{dataset}_comparison_alpha{alpha}_q{q}.png', bbox_inches='tight', dpi=300)
    plt.close()
