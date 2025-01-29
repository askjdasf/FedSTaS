#!/usr/bin/env python
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import config
from utils import *
from copy import deepcopy
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

if config.USE_GPU:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# turn data into Variable, do .cuda() when USE_GPU is True
def get_variable(x):
    x = Variable(x)
    return x.cuda() if config.USE_GPU else x
    # requires_grad=True with tensor x in newer PyTorch versions

def stratify_clients_compressed_gradients(args, compressed_grads):
    """
    Args:
        args: Arguments
        compressed_grads: Compressed gradients from clients
    """
    # Uses compressed gradients directly - no need for PCA
    data = compressed_grads
    print("Shape of compressed gradients:", data.shape)

    # Prototype Based Clustering: KMeans
    model = KMeans(n_clusters=args.strata_num)
    model.fit(data)
    pred_y = model.predict(data)
    pred_y = list(pred_y)
    result = []
    
    # put indexes into result
    for num in range(args.strata_num):
        one_type = []
        for index, value in enumerate(pred_y):
            if value == num:
                one_type.append(index)
        result.append(one_type)
    print("Stratification result:", result)
    
    save_path = f'dataset/stratify_result/{args.dataset}_{args.partition}.pkl'
    with open(save_path, 'wb') as output:
        pickle.dump(result, output)

    # print silhouette_score
    #s_score = metrics.silhouette_score(data, pred_y, sample_size=len(data), metric='euclidean')
    #print("strata_num：", args.strata_num, " silhouette_score：", s_score, "\n")
    
    return result

def accuracy_dataset(model, dataset):
    """Compute the accuracy {}% of `model` on `test_data`"""

    correct = 0

    for features, labels in dataset:

        features = get_variable(features)
        labels = get_variable(labels)

        predictions = model(features)
        _, predicted = predictions.max(1, keepdim=True)

        correct += torch.sum(predicted.view(-1, 1) == labels.view(-1, 1)).item()

    accuracy = 100 * correct / len(dataset.dataset)

    return accuracy

def loss_dataset(model, train_data, loss_classifier):
    """Compute the loss of `model` on `test_data`"""
    loss = 0
    for idx, (features, labels) in enumerate(train_data):

        features = get_variable(features)
        labels = get_variable(labels)

        predictions = model(features)
        loss += loss_classifier(predictions, labels)

    loss /= idx + 1 #average loss, idx is batch index
    return loss

def local_learning(model, mu: float, optimizer, train_data, n_SGD: int, loss_classifier):
    model_0 = deepcopy(model)

    for _ in range(n_SGD):
        train_data_iter = iter(train_data)
        features, labels = next(train_data_iter)

        features = get_variable(features)
        labels = get_variable(labels)

        optimizer.zero_grad()

        predictions = model(features)

        batch_loss = loss_classifier(predictions, labels)
        
        tensor_1 = list(model.parameters())
        tensor_2 = list(model_0.parameters())
        norm = sum(
            [
                torch.sum((tensor_1[i] - tensor_2[i]) ** 2)
                for i in range(len(tensor_1))
            ]
        )
        batch_loss += mu / 2 * norm
        
        batch_loss.backward()
        optimizer.step()

def FedProx_random_sampling(
    model,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay,
    mu,
):
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples) #(k,)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))
    
    for k, dl in enumerate(training_sets):

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)
        
    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):

        clients_params = []

        np.random.seed(i)
        sampled_clients = random.sample([x for x in range(K)], n_sampled)

        for k in sampled_clients:

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_classifier,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        new_model = deepcopy(model)
        weights_ = [weights[client] for client in sampled_clients]

        for layer_weigths in new_model.parameters():
            layer_weigths.data.sub_(sum(weights_) * layer_weigths.data)

        for k, client_hist in enumerate(clients_params):
            for idx, layer_weights in enumerate(new_model.parameters()):
                contribution = client_hist[idx].data * weights_[k]
                layer_weights.data.add_(contribution)

        model = new_model

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(
                loss_dataset(model, dl, loss_classifier).detach()
            )

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(
            f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
        )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist

def FedProx_importance_sampling(
    model,
    n_sampled,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr,
    file_name: str,
    decay,
    mu,
):
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):

        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):

        clients_params = []

        np.random.seed(i)
        sampled_clients = np.random.choice(
            K, size=n_sampled, replace=True, p=weights
        )

        for k in sampled_clients:

            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            local_learning(
                local_model,
                mu,
                local_optimizer,
                training_sets[k],
                n_SGD,
                loss_classifier,
            )

            # GET THE PARAMETER TENSORS OF THE MODEL
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)

            sampled_clients_hist[i, k] = 1

        # CREATE THE NEW GLOBAL MODEL
        new_model = deepcopy(model)
        weights_ = [1 / n_sampled] * n_sampled
        for layer_weigths in new_model.parameters():
            layer_weigths.data.sub_(layer_weigths.data)

        for k, client_hist in enumerate(clients_params):
            for idx, layer_weights in enumerate(new_model.parameters()):
                contribution = client_hist[idx].data * weights_[k]
                layer_weights.data.add_(contribution)

        model = new_model

        # COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(
                loss_dataset(model, dl, loss_classifier).detach()
            )

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(
            f"====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}"
        )

        # DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
        lr *= decay

    # SAVE THE DIFFERENT TRAINING HISTORY
    #    save_pkl(models_hist, "local_model_history", file_name)
    #    save_pkl(server_hist, "server_history", file_name)
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist

def FedProx_stratified_sampling(
    args,
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    decay,
    mu,
    d_prime: int,
):
    """
    Modified FedProx with stratified sampling based on gradient norms and K_desired samples
    """
    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    #print("Clients' weights:", weights)


    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):
        # 1. Get compressed gradients from all clients
        compressed_grads, grad_indices = collect_compressed_gradients(model, training_sets, d_prime)

        # 2. Stratify clients based on compressed gradients ******************************
        # Use compressed gradients for stratification
        stratify_result = stratify_clients_compressed_gradients(args, compressed_grads)

        N_STRATA = len(stratify_result)
        SIZE_STRATA = [len(cls) for cls in stratify_result]
        N_CLIENTS = sum(len(c) for c in stratify_result)  # number of clients

        # 3. Server computes the m_h *****************************************************
        # cal_allocation_number_NS uses Neyman allocation with N_h and S_h to calculate m_h
        # Note: S_h is calculated using compressed gradients, not restored gradients
        allocation_number = []
        if config.WITH_ALLOCATION and not args.partition == 'shard':
            allocation_number = cal_allocation_number_NS(stratify_result, compressed_grads, SIZE_STRATA,
                                                         args.sample_ratio)
        print(f"Allocation numbers (if any): {allocation_number}")

        # 4. Compute sampling probabilities based on gradient norms
        chosen_p = np.zeros((N_STRATA, N_CLIENTS)).astype(float)
        for j, cls in enumerate(stratify_result):
            client_grad_norms = []
            for k in cls:
                grad_norm = np.linalg.norm(compressed_grads[k])
                client_grad_norms.append(grad_norm)

            # Find Summation of ||Z_t^k|| for this stratum
            sum_norms = sum(client_grad_norms)
            # Find p_t^k
            for idx, k in enumerate(cls):
                chosen_p[j][k] = round(client_grad_norms[idx] / sum_norms, 12) \
                                 if sum_norms != 0 else 0
        
        # Sampling clients based on stratification
        if config.WITH_ALLOCATION and not args.partition == 'shard':
            selects = sample_clients_with_allocation(chosen_p, allocation_number)
        else:
            choice_num = int(100 * args.sample_ratio / args.strata_num)
            selects = sample_clients_without_allocation(chosen_p, choice_num)
        if args.partition == 'iid':
            selects = choice(100, int(100 * args.sample_ratio), replace=False,
                             p=[0.01 for _ in range(100)])
            
        #selected = []
        #for _ in selects:
        #    selected.append(_)
        #print("Chosen clients: ", selected)
        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []

        for k in selects:
            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)


            #total_samples = len(training_sets[k].dataset)

            full_subset = training_sets[k].dataset
            train_loader = torch.utils.data.DataLoader(
                full_subset,
                batch_size=args.batch_size,
                shuffle=True
            )

            # Local training with FedProx
            local_learning(
                local_model,
                mu,
                local_optimizer,
                train_loader,
                n_SGD,
                loss_classifier,
            )

            # Append parameters for aggregation
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))
            sampled_clients_for_grad.append(k)
            sampled_clients_hist[i, k] = 1

        # Create the new global model by aggregating client updates
        new_model = deepcopy(model)
        for layer_weights in new_model.parameters():
            layer_weights.data.zero_()
        n_contrib = len(clients_params)
        if n_contrib > 0:
            weights_ = [1.0 / n_sampled] * n_contrib

            # Add the contributions
            for k, client_hist in enumerate(clients_params):
                for idx, layer_weights in enumerate(new_model.parameters()):
                    layer_weights.data.add_(client_hist[idx].data * weights_[k])

            model = new_model
        else:
            # If no clients contributed (edge case), model stays the same
            pass

        # Compute the loss/accuracy of the different clients with the new model
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(loss_dataset(model, dl, loss_classifier).detach())

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(f"====> i: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}")

        lr *= decay

    # Save the training history
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist

def FedProx_stratified_dp_sampling(
    args,
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    decay,
    mu,
    alpha: float,  # Privacy parameter from FedSampling
    M: int,        # Maximum response value for the Estimator
    K_desired: int, # Desired sample size
    d_prime: int,   
):  
    # Initialize Estimator for privacy-preserving sampling
    train_users = {k: range(len(dl.dataset)) for k, dl in enumerate(training_sets)}
    estimator = Estimator(train_users, alpha, M)

    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    print("Clients' weights:", weights)

    # 1. each client sends compressed gradients **************************************
    # Get compressed gradients from all clients
    compressed_grads, grad_indices = collect_compressed_gradients(model, training_sets, d_prime)

    # 2. Stratify clients based on compressed gradients ******************************
    # Use compressed gradients for stratification
    stratify_result = stratify_clients_compressed_gradients(args, compressed_grads)

    N_STRATA = len(stratify_result)
    SIZE_STRATA = [len(cls) for cls in stratify_result]
    N_CLIENTS = sum(len(c) for c in stratify_result)  # number of clients

    # 3. Server computes the m_h *****************************************************
    # cal_allocation_number_NS uses Neyman allocation with N_h and S_h to calculate m_h
    # Note: S_h is calculated using compressed gradients, not restored gradients
    allocation_number = []
    if config.WITH_ALLOCATION and not args.partition == 'shard':
        allocation_number = cal_allocation_number_NS(stratify_result, compressed_grads, SIZE_STRATA, args.sample_ratio)
    print(allocation_number)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):
        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []

        # Estimate the total population size with privacy preservation
        hatN = estimator.estimate()
        print(f"Estimated population size (hatN): {hatN}")

        # Sampling clients based on stratification and privacy-preserving estimates
        chosen_p = np.zeros((N_STRATA, N_CLIENTS)).astype(float)
        for j, cls in enumerate(stratify_result):
            for k in range(N_CLIENTS):
                if k in cls:
                    chosen_p[j][k] = round(1 / SIZE_STRATA[j], 12)
        

        if config.WITH_ALLOCATION and not args.partition == 'shard':
            selects = sample_clients_with_allocation(chosen_p, allocation_number)
        else:
            choice_num = int(100 * args.sample_ratio / args.strata_num)
            selects = sample_clients_without_allocation(chosen_p, choice_num)
        if args.partition == 'iid':
            selects = choice(100, int(100 * args.sample_ratio), replace=False,
                             p=[0.01 for _ in range(100)])
            
        selected = []
        for _ in selects:
            selected.append(_)
        print("Chosen clients: ", selected)

        for k in selected:
            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            # local data sampling
            sampled_features, sampled_labels = local_data_sampling(
                training_sets[k], 
                K_desired, 
                hatN
            )

            if sampled_features is not None and len(sampled_features) > 0:
               
                sampled_dataset = torch.utils.data.TensorDataset(sampled_features, sampled_labels)
                sampled_loader = torch.utils.data.DataLoader(
                    sampled_dataset,
                    batch_size=args.batch_size,
                    shuffle=True
                )
            # Local training with FedProx
            local_learning(
                local_model,
                mu,
                local_optimizer,
                sampled_loader,
                n_SGD,
                loss_classifier,
            )

            # Append parameters for aggregation
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))
            sampled_clients_for_grad.append(k)
            sampled_clients_hist[i, k] = 1

        # Create the new global model by aggregating client updates
        new_model = deepcopy(model)
        # Data-size proportional weights
        #weights_ = [weights[client] for client in selected]
        weights_ = [1/n_sampled]*n_sampled

        for layer_weights in new_model.parameters():
            layer_weights.data.sub_(sum(weights_) * layer_weights.data)

        for k, client_hist in enumerate(clients_params):
            for idx, layer_weights in enumerate(new_model.parameters()):
                contribution = client_hist[idx].data * weights_[k]
                layer_weights.data.add_(contribution)

        model = new_model

        # Compute the loss/accuracy of the different clients with the new model
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(loss_dataset(model, dl, loss_classifier).detach())

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(f"====> i: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}")

        lr *= decay

    # Save the training history
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist

def calculate_aggregation_weights(stratify_result, chosen_p, selected_clients, n_sampled, weights=None, weighting_scheme='proposed', training_sets=None):
    """
    Calculate aggregation weights with different schemes and stability measures.
    weighting_scheme: 'uniform', 'size_prop', or 'proposed'
    n_sampled: number of clients to be sampled (based on q ratio)
    training_sets: dictionary of client training datasets
    """
    if weighting_scheme == 'uniform':
        # Simple uniform weighting based on n_sampled
        return [1.0 / n_sampled] * n_sampled
    
    elif weighting_scheme == 'size_prop':
        # Data-size proportional weighting
        if weights is None:
            return [1.0 / n_sampled] * n_sampled
        weights_ = [weights[client] for client in selected_clients]
        weights_sum = sum(weights_)
        return [w / weights_sum for w in weights_]
    
    else:  # proposed scheme with stability measures
        N_h = [len(cls) for cls in stratify_result]  # Size of each stratum
        N = n_sampled  # Total number of clients
        
        # Count selected clients in each stratum
        m_h = [0] * len(stratify_result)
        for k in selected_clients:
            for h, cls in enumerate(stratify_result):
                if k in cls:
                    m_h[h] += 1
                    break
        
        # Create mapping of client to stratum
        client_to_stratum = {}
        for h, cls in enumerate(stratify_result):
            for k in cls:
                client_to_stratum[k] = h
        
        
        # Calculate weights with stability measures
        weights_ = []
        for k in selected_clients:
            h = client_to_stratum[k]
            if m_h[h] > 0 and training_sets is not None:  # Avoid division by zero
                # Calculate p_tk and N for this client
                total_samples = len(training_sets[k].dataset)  # N is total samples for this client
                K_desired = 2048  # This should match your setup
                p_tk = min(1.0, max(0.0, float(K_desired) / float(total_samples)))
                
                # Add small epsilon to avoid division by very small numbers
                epsilon = 1e-8
                p_tk = max(p_tk, epsilon)
                
                # Calculate weight according to the formula
                stratum_weight = N_h[h] / N  # Proportion of clients in stratum h
                weight = stratum_weight * (1.0 / (m_h[h] * p_tk))  # Include inverse probability weight
                
                # Add some bounds to prevent extreme weights
                max_weight = 10.0 / p_tk  # Scale maximum weight by inverse probability
                weight = min(weight, max_weight)
                weights_.append(weight)
                
                # Print debug info
                print(f"Client {k} - Stratum {h}, total_samples: {total_samples}, "
                      f"p_tk: {p_tk:.4f}, stratum_weight: {stratum_weight:.4f}, "
                      f"m_h: {m_h[h]}, raw_weight: {weight:.4f}")
            else:
                weights_.append(0)
        
        # Print statistics for debugging
        if len(weights_) > 0:
            print(f"Weight stats before scaling - Min: {min(weights_):.4f}, Max: {max(weights_):.4f}, "
                  f"Mean: {sum(weights_)/len(weights_):.4f}")
        
        # Normalize weights to sum to 1
        weights_sum = sum(weights_)
        if weights_sum > 0:
            weights_ = [w / weights_sum for w in weights_]
            print(f"Final weight sum: {sum(weights_):.6f}")
        else:
            # Fallback to uniform weights if something goes wrong
            weights_ = [1.0 / n_sampled] * n_sampled
        
        return weights_

def FedProx_stratified_sampling_compressed_gradients(
    args,
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    decay,
    mu,
    K_desired: float,
    d_prime: int,
):  
    """
    Federated learning with stratified sampling using compressed gradients (non-DP version)
    """
    #print("Running FedProx with stratified sampling using compressed gradients")
    #print(f"Number of sampled clients (n_sampled): {n_sampled}")

    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    weights = n_samples / np.sum(n_samples)
    #print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):
        # 1. Get compressed gradients from all clients
        compressed_grads, grad_indices = collect_compressed_gradients(model, training_sets, d_prime)

        # 2. Stratify clients based on compressed gradients ******************************
        # Use compressed gradients for stratification
        stratify_result = stratify_clients_compressed_gradients(args, compressed_grads)

        N_STRATA = len(stratify_result)
        SIZE_STRATA = [len(cls) for cls in stratify_result]
        N_CLIENTS = sum(len(c) for c in stratify_result)  # number of clients

        # 3. Server computes the m_h *****************************************************
        # cal_allocation_number_NS uses Neyman allocation with N_h and S_h to calculate m_h
        # Note: S_h is calculated using compressed gradients, not restored gradients
        allocation_number = []
        if config.WITH_ALLOCATION and not args.partition == 'shard':
            allocation_number = cal_allocation_number_NS(stratify_result, compressed_grads, SIZE_STRATA,
                                                         args.sample_ratio)
        print(f"Allocation numbers (if any): {allocation_number}")

        # 4. Compute sampling probabilities based on gradient norms
        chosen_p = np.zeros((N_STRATA, N_CLIENTS)).astype(float)
        for j, cls in enumerate(stratify_result):
            client_grad_norms = []
            for k in cls:
                grad_norm = np.linalg.norm(compressed_grads[k])
                client_grad_norms.append(grad_norm)

            # Find Summation of ||Z_t^k|| for this stratum
            sum_norms = sum(client_grad_norms)
            # Find p_t^k
            for idx, k in enumerate(cls):
                chosen_p[j][k] = round(client_grad_norms[idx] / sum_norms, 12) \
                                 if sum_norms != 0 else 0
        
        # Sampling clients based on stratification
        if config.WITH_ALLOCATION and not args.partition == 'shard':
            selects = sample_clients_with_allocation(chosen_p, allocation_number)
        else:
            choice_num = int(100 * args.sample_ratio / args.strata_num)
            selects = sample_clients_without_allocation(chosen_p, choice_num)
        if args.partition == 'iid':
            selects = choice(100, int(100 * args.sample_ratio), replace=False,
                             p=[0.01 for _ in range(100)])
            
        #selected = []
        #for _ in selects:
        #    selected.append(_)
        #print("Chosen clients: ", selected)
        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []
        for k in selects:
            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)
            
            # Calculate sampling probability based on actual dataset size
            #total_samples = len(training_sets[k].dataset)
            # Ensure sampling probability is valid (between 0 and 1)
            #sampling_prob = K_desired #prop = 0.5
            #print(f"Client {k} - Total samples: {total_samples}, K_desired: {K_desired}, Sampling prob: {sampling_prob}")
            
            # Sample data points
            sampled_features = []
            sampled_labels = []
            
            for features, labels in training_sets[k]:
                sample_mask = np.random.binomial(n=1, p=K_desired, size=len(features))
                selected_features = features[sample_mask == 1]
                selected_labels = labels[sample_mask == 1]
                
                if len(selected_features) > 0:
                    sampled_features.append(selected_features)
                    sampled_labels.append(selected_labels)

            if sampled_features:
                sampled_features = torch.cat(sampled_features)
                sampled_labels = torch.cat(sampled_labels)
                if sampled_features is not None and len(sampled_features) > 0:
                    sampled_dataset = torch.utils.data.TensorDataset(sampled_features, sampled_labels)
                    sampled_loader = torch.utils.data.DataLoader(
                        sampled_dataset,
                        batch_size=args.batch_size,
                        shuffle=True
                    )

                # Local training with FedProx
                local_learning(
                    local_model,
                    mu,
                    local_optimizer,
                    sampled_loader,
                    n_SGD,
                    loss_classifier,
                )

            # Append parameters for aggregation
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))
            sampled_clients_for_grad.append(k)
            sampled_clients_hist[i, k] = 1

        #if not clients_params:  # Skip aggregation if no clients had samples
        #    print("Warning: No clients had valid samples in this round")
        #    continue

        # Create the new global model by aggregating client updates
        new_model = deepcopy(model)
        
        # Calculate weights using the new function with stability measures
        '''
        weights_ = calculate_aggregation_weights(
            stratify_result, 
            chosen_p, 
            sampled_clients_for_grad,
            n_sampled=n_sampled, 
            weighting_scheme='proposed',
            training_sets=training_sets
        )
        '''

        #print(f"Round {i+1} - Sum of weights: {sum(weights_):.6f} (should be close to {1.0/n_sampled:.6f})")

        # Aggregate model updates
        for layer_weights in new_model.parameters():
            layer_weights.data.zero_()
        n_contrib = len(clients_params)
        if n_contrib > 0:
            weights_ = [1.0 / n_sampled] * n_contrib

            # Add the contributions
            for k, client_hist in enumerate(clients_params):
                for idx, layer_weights in enumerate(new_model.parameters()):
                    layer_weights.data.add_(client_hist[idx].data * weights_[k])

            model = new_model
        else:
            # If no clients contributed (edge case), model stays the same
            pass

        # Compute the loss/accuracy of the different clients with the new model
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(loss_dataset(model, dl, loss_classifier).detach())

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(f"====> i: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}")

        lr *= decay

    # Save the training history
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist
import math
def FedProx_stratified_dp_sampling_compressed_gradients(
    args,
    model,
    n_sampled: int,
    training_sets: list,
    testing_sets: list,
    n_iter: int,
    n_SGD: int,
    lr: float,
    file_name: str,
    decay,
    mu,
    privacy: float,  # Privacy parameter from FedSampling
    M: int,        # Maximum response value for the Estimator
    K_desired: float, # Desired sample size prop
    d_prime: int,  # Compression parameter
):  
    # Initialize Estimator for privacy-preserving sampling
    alpha = (math.exp(privacy) - 1) / (math.exp(privacy) + M - 2)
    train_users = {k: range(len(dl.dataset)) for k, dl in enumerate(training_sets)}
    estimator = Estimator(train_users, alpha, M)

    K = len(training_sets)  # number of clients
    n_samples = np.array([len(db.dataset) for db in training_sets])
    #num_data = sum(len(dl.dataset) for dl in training_sets)
    # K_desired is now derived from the total data * fraction
    #K_desired = int(num_data * K_desired)
    clipped_total = 0
    for dl in training_sets:
        real_size = len(dl.dataset)
        clipped_total += min(real_size, M - 1)
    K_desired_num = int(clipped_total * K_desired)
    weights = n_samples / np.sum(n_samples)
    #print("Clients' weights:", weights)

    loss_hist = np.zeros((n_iter + 1, K))
    acc_hist = np.zeros((n_iter + 1, K))

    for k, dl in enumerate(training_sets):
        loss_hist[0, k] = float(loss_dataset(model, dl, loss_classifier).detach())
        acc_hist[0, k] = accuracy_dataset(model, dl)

    # LOSS AND ACCURACY OF THE INITIAL MODEL
    server_loss = np.dot(weights, loss_hist[0])
    server_acc = np.dot(weights, acc_hist[0])
    print(f"====> i: 0 Loss: {server_loss} Test Accuracy: {server_acc}")

    sampled_clients_hist = np.zeros((n_iter, K)).astype(int)

    for i in range(n_iter):
        # 1. Get compressed gradients from all clients
        compressed_grads, grad_indices = collect_compressed_gradients(model, training_sets, d_prime)

        # 2. Stratify clients based on compressed gradients ******************************
        # Use compressed gradients for stratification
        stratify_result = stratify_clients_compressed_gradients(args, compressed_grads)

        N_STRATA = len(stratify_result)
        SIZE_STRATA = [len(cls) for cls in stratify_result]
        N_CLIENTS = sum(len(c) for c in stratify_result)  # number of clients

        # 3. Server computes the m_h *****************************************************
        # cal_allocation_number_NS uses Neyman allocation with N_h and S_h to calculate m_h
        # Note: S_h is calculated using compressed gradients, not restored gradients
        allocation_number = []
        if config.WITH_ALLOCATION and not args.partition == 'shard':
            allocation_number = cal_allocation_number_NS(stratify_result, compressed_grads, SIZE_STRATA,
                                                         args.sample_ratio)
        print(f"Allocation numbers (if any): {allocation_number}")

        # Estimate the total population size with privacy preservation
        hatN = estimator.estimate()
        print(f"Estimated population size (hatN): {hatN}")

        # 4. Server computes p_t^k ***************************************************
        # Note: ||Z_t^k|| is calculated using compressed gradients, not restored gradients
        chosen_p = np.zeros((N_STRATA, N_CLIENTS)).astype(float)
        for j, cls in enumerate(stratify_result):
            client_grad_norms = []
            for k in cls:
                    # Find ||Z_t^k|| of this client and store it in an array
                    grad_norm = np.linalg.norm(compressed_grads[k])
                    client_grad_norms.append(grad_norm)

            # Find Summation of ||Z_t^k|| for this stratum
            sum_norms = sum(client_grad_norms)
            # Find p_t^k
            for idx, k in enumerate(cls):
                chosen_p[j][k] = round(client_grad_norms[idx] / sum_norms, 12) \
                                 if sum_norms != 0 else 0
        
        # Sampling clients based on stratification and privacy-preserving estimates
        if config.WITH_ALLOCATION and not args.partition == 'shard':
            selects = sample_clients_with_allocation(chosen_p, allocation_number)
        else:
            choice_num = int(100 * args.sample_ratio / args.strata_num)
            selects = sample_clients_without_allocation(chosen_p, choice_num)
        if args.partition == 'iid':
            selects = choice(100, int(100 * args.sample_ratio), replace=False,
                             p=[0.01 for _ in range(100)])
            
        #selected = []
        #for _ in selects:
        #    selected.append(_)
        #print("Chosen clients: ", selected)
        clients_params = []
        clients_models = []
        sampled_clients_for_grad = []
        for k in selects:
            local_model = deepcopy(model)
            local_optimizer = optim.SGD(local_model.parameters(), lr=lr)

            # local data sampling
            sampled_features, sampled_labels = local_data_sampling(
                training_sets[k], 
                K_desired_num,
                hatN
            )

            if sampled_features is not None and len(sampled_features) > 0:
               
                sampled_dataset = torch.utils.data.TensorDataset(sampled_features, sampled_labels)
                sampled_loader = torch.utils.data.DataLoader(
                    sampled_dataset,
                    batch_size=args.batch_size,
                    shuffle=True
                )
            # Local training with FedProx
            local_learning(
                local_model,
                mu,
                local_optimizer,
                sampled_loader,
                n_SGD,
                loss_classifier,
            )

            # Append parameters for aggregation
            list_params = list(local_model.parameters())
            list_params = [tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)
            clients_models.append(deepcopy(local_model))
            sampled_clients_for_grad.append(k)
            sampled_clients_hist[i, k] = 1

        # Create the new global model by aggregating client updates
        new_model = deepcopy(model)
        # Aggregate model updates
        for layer_weights in new_model.parameters():
            layer_weights.data.zero_()
        n_contrib = len(clients_params)
        if n_contrib > 0:
            weights_ = [1.0 / n_sampled] * n_contrib

            # Add the contributions
            for k, client_hist in enumerate(clients_params):
                for idx, layer_weights in enumerate(new_model.parameters()):
                    layer_weights.data.add_(client_hist[idx].data * weights_[k])

            model = new_model
        else:
            # If no clients contributed (edge case), model stays the same
            pass


        # Compute the loss/accuracy of the different clients with the new model
        for k, dl in enumerate(training_sets):
            loss_hist[i + 1, k] = float(loss_dataset(model, dl, loss_classifier).detach())

        for k, dl in enumerate(testing_sets):
            acc_hist[i + 1, k] = accuracy_dataset(model, dl)

        server_loss = np.dot(weights, loss_hist[i + 1])
        server_acc = np.dot(weights, acc_hist[i + 1])

        print(f"====> i: {i + 1} Loss: {server_loss} Server Test Accuracy: {server_acc}")

        # Decrease the learning rate
        lr *= decay

    # Save the training history
    save_pkl(loss_hist, "loss", file_name)
    save_pkl(acc_hist, "acc", file_name)

    torch.save(
        model.state_dict(), f"saved_exp_info/final_model/{file_name}.pth"
    )

    return model, loss_hist, acc_hist

def run(args, model_mnist, n_sampled, list_dls_train, list_dls_test, file_name):
    """RUN FEDAVG WITH RANDOM SAMPLING"""
    if args.sampling == "random" and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_random_sampling(
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
        )

    """RUN FEDAVG WITH IMPORTANCE SAMPLING"""
    if args.sampling == "importance" and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_importance_sampling(
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
        )

    """RUN FEDAVG WITH OURS SAMPLING"""
    if (args.sampling == "ours") and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_stratified_sampling(
            args,
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
            args.d_prime,
        )
        
    """RUN FEDAVG WITH dp sampling """
    if (args.sampling == "dp") and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_stratified_dp_sampling(
            args,
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
            args.alpha,
            args.M,
            args.K_desired,
            args.d_prime,
            )
    """RUN FEDAVG WITH dp sampling and compressed client gradients"""
    if (args.sampling == "comp_grads") and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_stratified_sampling_compressed_gradients(
            args,
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
            args.K_desired,
            args.d_prime,
            )
    """RUN FEDAVG WITH dp sampling and compressed client gradients"""
    if (args.sampling == "dp_comp_grads") and (
            not os.path.exists(f"saved_exp_info/acc/{file_name}.pkl") or args.force
    ):
        FedProx_stratified_dp_sampling_compressed_gradients(
            args,
            model_mnist,
            n_sampled,
            list_dls_train,
            list_dls_test,
            args.n_iter,
            args.n_SGD,
            args.lr,
            file_name,
            args.decay,
            args.mu,
            args.privacy,
            args.M,
            args.K_desired,
            args.d_prime,
            )
