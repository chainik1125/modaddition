import os
import sys
import dill
import shutil

# Add the path to the PYTHONPATH
new_path = "/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/Code"
if new_path not in sys.path:
    sys.path.append(new_path)
    os.environ["PYTHONPATH"] = os.pathsep.join(sys.path)

from dataclasses import dataclass
from timeit import default_timer as timer
import time
import pickle
from tqdm.auto import tqdm # for loop progress bar
import itertools as it
import numpy as np
import random
from matplotlib import pyplot as plt
import plotly.colors as plc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import inspect

# machine learning imports

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import copy
#import kaleido

################Seeds
from cluster.data_objects import seed_average_onerun
from mnist.cluster_run_average import MLP_mnist
from Analysis.activations_Ising import open_files_in_leaf_directories,load_model
from cluster.cluster_run_average import TrainArgs
from cluster.cluster_run_average import CNN
from cluster.cluster_run_average import CNN_nobias
from contextlib import contextmanager
import functools
from cluster.cluster_run_average import ModularArithmeticDataset
from cluster.cluster_run_average import Square, MLP
import glob

runfiles=['/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/grokfast/mnist_none_wd20e+00_usegrokfast_none.pt',
          '/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/grokfast/mnist_ma_w100_l010_wd20e+00_usegrokfast_ma.pt']

runfiles_lowwd=['/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/grokfast/mnist_ma_w100_l010_wd10e-02_usegrokfast_ma.pt',
                '/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/grokfast/mnist_none_wd10e-02_usegrokfast_none.pt']

def plot_grokfast(run_file_list):
    fig=make_subplots(rows=1,cols=1)
    colors=['red','blue']
    for run_file in run_file_list:
        run_dict=torch.load(run_file)

        fig.add_trace(go.Scatter(x=run_dict['its'],y=run_dict['train_acc'],mode='lines',line=dict(color=colors[run_file_list.index(run_file)],dash='dash'),showlegend=True,name=f'Train, {run_dict["args"]["filter"]}'),row=1,col=1)
        fig.add_trace(go.Scatter(x=run_dict['its'],y=run_dict['val_acc'],mode='lines',line=dict(color=colors[run_file_list.index(run_file)],dash='solid'),showlegend=True,name=f'Test, {run_dict["args"]["filter"]}'),row=1,col=1)
        
        fig.update_yaxes(title_text='Accuracy')
        fig.update_yaxes(title_text='Train batches')
        fig.show()
#plot_grokfast(runfiles_lowwd)

def parse_seeds(filename):
    """Extracts x, y, z values from the filename."""
    dataseed = filename.split('dataseed_')[1].split('_')[0]
    sgdseed = filename.split('sgdseed_')[1].split('_')[0]
    initseed = filename.split('initseed_')[1].split('_')[0]
    wd=filename.split('wd_')[1].split('_')[0]
    wm=filename.split('wm_')[1].split('_')[0]
    return dataseed, sgdseed, initseed, wd, wm


def move_duplicates(source_folder, destination_folder):
    """Moves duplicate files to a separate folder, including files in subfolders."""
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    seen_seeds = set()

    for root, _, files in os.walk(source_folder):
        for filename in files:
            if filename.startswith("dataseed"):  # replace .ext with the actual file extension
                x, y, z,w,v = parse_seeds(filename)
                seeds_tuple = (x, y, z,w,v)


                if seeds_tuple in seen_seeds:
                    shutil.move(os.path.join(root, filename), os.path.join(destination_folder, filename))
                else:
                    seen_seeds.add(seeds_tuple)

def organize_files_by_wm(source_folder, hidden_layers, description):
    """Organizes files into folders based on their wm value."""
    for filename in os.listdir(source_folder):
        if filename.startswith("dataseed"):  # replace .ext with the actual file extension
            wm_value = parse_seeds(filename)[-1]
            folder_name = f"hiddenlayer_{hidden_layers}_desc_{description}_wm_{wm_value}"
            folder_path = os.path.join(source_folder, folder_name)

            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            shutil.move(os.path.join(source_folder, filename), os.path.join(folder_path, filename))

source_folder = "/Users/dmitrymanning-coe/Documents/Research/Grokking/mnistcluster/mnist_wd_0.1_longer_duplicates"
hidden_layers = [70, 35]  # Replace with your desired hidden layer structure
description = "mnist"      # Replace with your desired description
#organize_files_by_wm(source_folder, hidden_layers, description)

source_folder="/Users/dmitrymanning-coe/Documents/Research/Grokking/mnistcluster/mnist_wd_0.1_longer_duplicates_3"
destination_folder="/Users/dmitrymanning-coe/Documents/Research/Grokking/mnistcluster/mnist_wd_0.1_longer_duplicates_4"
#move_duplicates(source_folder, destination_folder)


nongrok_foldername="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/test_runs/hiddenlayer_[512]_desc_modadd_wm_0.1"
foldername="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/test_runs/hiddenlayer_[512]_desc_modadd_wm_15.0"


grok_object=open_files_in_leaf_directories(foldername)[0]
nongrok_object=open_files_in_leaf_directories(nongrok_foldername)[0]
grok_object.cosine_sim(nongrok_object).show()
exit()

def manual_adamw_update(optimizer_dic, model):
    state_dict = optimizer_dic
    param_groups = state_dict['param_groups']
    print(param_groups)
    
    state = state_dict['state']

    total_updates = {}
    weight_decay_updates = {}

    # Get a list of parameter names for better readability
    param_names = [name for name, _ in model.named_parameters()]

    for param_group in param_groups:
        for param_id, param in enumerate(param_group['params']):
            if param in state:
                param_state = state[param]
                exp_avg = param_state['exp_avg']
                exp_avg_sq = param_state['exp_avg_sq']
                step = param_state['step']

                beta1, beta2 = param_group['betas']
                lr = param_group['lr']
                weight_decay = param_group['weight_decay']
                eps = param_group['eps']

                # Compute bias-corrected first and second moment estimates
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                exp_avg_hat = exp_avg / bias_correction1
                exp_avg_sq_hat = exp_avg_sq / bias_correction2

                # Compute the update
                denom = (exp_avg_sq_hat.sqrt() / (bias_correction2 ** 0.5)).add_(eps)
                step_size = lr / bias_correction1
                update = exp_avg_hat / denom

                # Access the actual parameter tensor
                param_tensor = list(model.parameters())[param_id]

                # Compute weight decay update
                weight_decay_update = weight_decay * param_tensor.data if weight_decay != 0 else torch.zeros_like(param_tensor.data)

                # Store the updates
                total_updates[param_names[param_id]] = update.clone()
                weight_decay_updates[param_names[param_id]] = weight_decay_update.clone()

                # Apply the total update (including weight decay)
                param_tensor.data.add_(update + weight_decay_update, alpha=-step_size)

    return total_updates, weight_decay_updates
            






count=0
for object in objects:
    if count<1:
        epochs=object.model_epochs()
        weight_decay=[]
        gradient=[]
        end_grads=[]
        end_wds=[]
        end_model,end_model_state_dict=load_model(object,epochs[-1])
        end_opt=object.models[epochs[-1]]['optimizer']
        end_grad_updates, end_weight_decay_updates = manual_adamw_update(end_opt,end_model)
        
        
        end_max_index1=torch.argmax(end_grad_updates['model.1.weight'])
        end_max_index1=torch.unravel_index(end_max_index1,end_grad_updates['model.3.weight'].shape)

        end_max_index2=torch.argmax(end_grad_updates['model.1.weight'])
        end_max_index2=torch.unravel_index(end_max_index2,end_grad_updates['model.3.weight'].shape)
        
        for epoch in object.model_epochs():
            opt=object.models[epoch]['optimizer']
            model,model_state_dict=load_model(object,epoch)
            grad_updates, weight_decay_updates = manual_adamw_update(opt,model)
            end_grad1=grad_updates['model.1.weight'][end_max_index1].item()
            end_grad2=grad_updates['model.3.weight'][end_max_index2].item()
            end_grads.append(torch.tensor([end_grad1,end_grad2]))
            total1=torch.quantile(grad_updates['model.1.weight'],0.8).item()
            total2=torch.quantile(grad_updates['model.3.weight'],0.8).item()
            gradient.append(torch.tensor([total1,total2]))
            end_wd1=weight_decay_updates['model.1.weight'][end_max_index1].item()
            end_wd2=weight_decay_updates['model.3.weight'][end_max_index2].item()
            end_wds.append(torch.tensor([end_wd1,end_wd2]))
            wd1=torch.quantile(weight_decay_updates['model.1.weight'],0.8).item()
            wd2=torch.quantile(weight_decay_updates['model.3.weight'],0.8).item()
            weight_decay.append(torch.tensor([wd1,wd2]))
        
        epochs=torch.tensor(epochs)
        weight_decay=torch.stack(weight_decay)
        gradient=torch.stack(gradient)
        end_wds=torch.stack(end_wds)
        end_grads=torch.stack(end_grads)
        

        fig=make_subplots(rows=3,cols=1)
        fig.update_layout(title_text=f'Gradient and Weight Decay, wd={object.trainargs.weight_decay},wm={object.trainargs.weight_multiplier}')
        fig.add_trace(go.Scatter(x=epochs,y=gradient[:,0],name='Gradient 1'),row=1,col=1)
        fig.add_trace(go.Scatter(x=epochs,y=weight_decay[:,0],name='WD 1'),row=1,col=1)
        fig.add_trace(go.Scatter(x=epochs,y=gradient[:,1],name='Gradient 1'),row=2,col=1)
        fig.add_trace(go.Scatter(x=epochs,y=weight_decay[:,1],name='WD 1'),row=2,col=1)
        fig.add_trace(go.Scatter(x=epochs,y=end_grads[:,0],name='End grad 1'),row=3,col=1)
        fig.add_trace(go.Scatter(x=epochs,y=end_wds[:,0],name='end wd 1'),row=3,col=1)
        
        fig.update_yaxes(type='log')
        fig.show()
        object.traincurves_and_iprs(object,remove_wdloss=False).show()
        count+=1
