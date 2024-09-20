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
from mnist.data_objects import seed_average_onerun
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
import einops
import pandas as pd

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


#cosine_plots
# nongrok_foldername="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/test_runs/hiddenlayer_[70, 35]_desc_mnist_wm_1.0"
# foldername="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/test_runs/hiddenlayer_[70, 35]_desc_mnist_wm_10.0"


# grok_object=open_files_in_leaf_directories(foldername)[0]



# nongrok_object=open_files_in_leaf_directories(nongrok_foldername)[0]
# methods = [func for func in dir(nongrok_object) if callable(getattr(nongrok_object, func)) and not func.startswith("__")]

# grok_object.cosine_sim(nongrok_object).show()


#Linear plots
nongrok_foldername="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/test_runs/linear/hiddenlayer_[512]_desc_modadd_wm_17.0"
foldername="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/test_runs/linear/hiddenlayer_[512]_desc_modadd_wm_0.11"


grok_object=open_files_in_leaf_directories(foldername)[0]

nongrok_object=open_files_in_leaf_directories(nongrok_foldername)[0]

# print(f'trainargs grok: {vars(grok_object.trainargs)}')
# print(f'trainargs nongrok: {vars(nongrok_object.trainargs)}')


#grok_object.linear_decomposition_plot(nongrok_object).show()
#grok_object.traincurves_and_iprs(nongrok_object).show()
#grok_object.cosine_sim(nongrok_object).show()

#methods = [func for func in dir(nongrok_object) if callable(getattr(nongrok_object, func)) and not func.startswith("__")]

#grok_object.linear_decomposition_plot(nongrok_object).show()
# grok_object.cosine_sim(nongrok_object).show()
# grok_object.traincurves_and_iprs(nongrok_object).show()
# grok_object.weights_histogram_epochs2(nongrok_object).show()


#Let's try to do the Fourier transform
def transform_tensors(twop_vectors):
    if twop_vectors.shape[0]==97:
        P=twop_vectors.shape[0]
        twop_vectors = twop_vectors.reshape(-1, 2*P)
        N=twop_vectors.shape[0]
        
    elif twop_vectors.shape[0]//2==int:
        N, twoP = twop_vectors.shape
        P = twoP // 2
    
    else:
        P=twop_vectors.shape[0]
        twop_vectors = twop_vectors.reshape(-1, 2*P)
        N=twop_vectors.shape[0]
        
    

    # Find the non-zero indices in the entire Nx2P tensor
    number,indices=torch.nonzero(twop_vectors,as_tuple=True)
    first_indices=indices[::2]
    second_indices=indices[1::2]


    
    # Extract row indices, and the two parts of column indices
    # row_indices = sortednz[:, 0]
    # col_indices = sortednz[:, 1]
    # print(row_indices)
    # print(col_indices)
    # exit()
    # Split the col_indices into two parts (for P and P)
    first_half = first_indices    # Corresponds to which of the two parts of P
    second_half = second_indices % P    # Actual column in the P-dimension

    # Create an empty NxPxP tensor
    psquared = torch.zeros(N, P, P)

    # Fill the psquared tensor based on the calculated indices
    psquared[torch.arange(len(first_half)), first_half, second_half] = 1
    return psquared

def make_all_twop(nongrok_object,p=None):
    P = nongrok_object.trainargs.P
    if p is not None:
        P=p

    # Create a tensor of shape (P, P, 2P) filled with zeros
    all_twop_inputs = torch.zeros(P, P, 2*P)

    # Set the appropriate indices to 1
    all_twop_inputs[torch.arange(P), :, torch.arange(P)] = 1
    all_twop_inputs[:, torch.arange(P), torch.arange(P, 2*P)] = 1
    
    

    # Reshape the tensor to (P*P, 2P) from (PxPx2P)

    #all_twop_inputs = all_twop_inputs.reshape(-1, 2*P)

    return all_twop_inputs

def fourier_new(runobject):
    epoch=runobject.model_epochs()[-1]
    P=runobject.trainargs.P
    model= load_model(runobject,epoch)[0]
    weights=[p for p in model.parameters() if p.dim()>1]
    all_twop_inputs=make_all_twop(runobject)
    all_psquared_inputs=transform_tensors(all_twop_inputs)
    
    # coeffs=torch.zeros(P,P,2)
    # for i in range(P):
    #     for j in range(P):
    #         coeffs[i,j,0]=i
    #         coeffs[i,j,1]=j
    
    coeffs=torch.zeros(P,P,2)
    rows = torch.arange(P).unsqueeze(1).expand(P, P)
    cols = torch.arange(P).expand(P, P)

    # Assign the row and column indices to the last dimension
    coeffs[..., 0] = rows
    coeffs[..., 1] = cols

    ks=2*torch.pi*coeffs/P
    
    fourier_W=torch.zeros(P,P,weights[0].shape[0])
    for i in tqdm(range(P)):
        for j in range(P):
            start=time.time()
            exponential=torch.exp(-1j*(ks[i,j,0]*coeffs[:,:,0]+ks[i,j,1]*coeffs[:,:,1]))
            exponential=exponential.unsqueeze(-1)
            presum=(all_twop_inputs@weights[0].T)*exponential
            
            postsum=torch.sum(presum,dim=0,dtype=np.complex64)
            postsum2=torch.sum(postsum,dim=0,dtype=np.complex64)
            
            fourier_W[i,j]=postsum2
            
            
            
def fourier_new_numpy(runobject):
    epoch=runobject.model_epochs()[-1]
    P=runobject.trainargs.P
    model= load_model(runobject,epoch)[0]
    weights=[p for p in model.parameters() if p.dim()>1]
    all_twop_inputs=make_all_twop(runobject)
    all_psquared_inputs=transform_tensors(all_twop_inputs)
    
    # coeffs=torch.zeros(P,P,2)
    # for i in range(P):
    #     for j in range(P):
    #         coeffs[i,j,0]=i
    #         coeffs[i,j,1]=j
    
    coeffs=torch.zeros(P,P,2)
    rows = torch.arange(P).unsqueeze(1).expand(P, P)
    cols = torch.arange(P).expand(P, P)

    # Assign the row and column indices to the last dimension
    coeffs[..., 0] = rows
    coeffs[..., 1] = cols

    ks=2*torch.pi*coeffs/P
    
    ks=ks.detach().numpy()
    coeffs=coeffs.detach().numpy()
    weights=[weights[0].detach().numpy()]
    all_twop_inputs=all_twop_inputs.detach().numpy()
    

    fourier_W=np.zeros((P,P,weights[0].shape[0]),dtype=np.complex64)
    for i in tqdm(range(P)):
        for j in range(P):
            
            exponential=np.exp(-1j*(ks[i,j,0]*coeffs[:,:,0]+ks[i,j,1]*coeffs[:,:,1]))
            exponential/=np.linalg.norm(exponential)
            exponential=np.expand_dims(exponential,axis=-1)
            presum=(all_twop_inputs@weights[0].T)*exponential


            postsum=np.sum(presum,axis=0,dtype=np.complex64)
            postsum2=np.sum(postsum,axis=0,dtype=np.complex64)
            
            fourier_W[i,j]=postsum2

    return fourier_W



def fourier_test(P):
    all_twop_inputs=make_all_twop(nongrok_object,P)
    all_psquared_inputs=transform_tensors(all_twop_inputs)
    print(f'all_psquared_inputs shape: {all_psquared_inputs.shape}')

    
    coeffs=torch.zeros(P,P,2)
    rows = torch.arange(P).unsqueeze(1).expand(P, P)
    cols = torch.arange(P).expand(P, P)

    # Assign the row and column indices to the last dimension
    coeffs[..., 0] = rows
    coeffs[..., 1] = cols

    ks=2*torch.pi*coeffs/P

    weights=[torch.cat((torch.exp(-1j*2*torch.pi*torch.arange(P)/P),torch.exp(-1j*2*torch.pi*torch.arange(P)/P))) for i in range(3)]
    weights=torch.stack(weights)
    weights=[weights]



    fourier_W=np.zeros((P,P,weights[0].shape[0]),dtype=np.complex64)
    ks=ks.detach().numpy()
    coeffs=coeffs.detach().numpy()
    weights=[weights[0].detach().numpy()]
    all_twop_inputs=all_twop_inputs.detach().numpy()
    print(f'fourier_W shape: {fourier_W.shape}')
    
    for i in range(P):
        for j in range(P):
            
            exponential=np.exp(-1j*(ks[i,j,0]*coeffs[:,:,0]+ks[i,j,1]*coeffs[:,:,1]))
            exponential/=np.linalg.norm(exponential)
            exponential=np.expand_dims(exponential,axis=-1)
            presum=(all_twop_inputs@weights[0].T)*exponential

            postsum=np.sum(presum,axis=0,dtype=np.complex64)
            postsum2=np.sum(postsum,axis=0,dtype=np.complex64)
            
            

            fourier_W[i,j]=postsum2

    

            
            

    
    return fourier_W


def fourier_test_vectorized(P):
    all_twop_inputs = make_all_twop(nongrok_object, P)
    all_psquared_inputs = transform_tensors(all_twop_inputs)
    print(f'all_psquared_inputs shape: {all_psquared_inputs.shape}')

    # Create coefficients tensor
    coeffs = torch.stack(torch.meshgrid(torch.arange(P), torch.arange(P)), dim=-1)

    ks = 2 * torch.pi * coeffs / P

    # Create weights tensor
    weights = torch.exp(-1j * 2 * torch.pi * torch.arange(P) / P)
    weights = torch.stack([weights, weights, weights])
    weights = weights.unsqueeze(0)

    # Convert to numpy for compatibility with the rest of the code
    ks = ks.numpy()
    coeffs = coeffs.numpy()
    weights = weights.numpy()
    all_twop_inputs = all_twop_inputs.numpy()

    # Prepare exponential term
    exponential = np.exp(-1j * (ks[:,:,0,None,None] * coeffs[:,:,0] + ks[:,:,1,None,None] * coeffs[:,:,1]))
    exponential /= np.linalg.norm(exponential, axis=(2,3), keepdims=True)
    exponential = exponential[:,:,:,:,None]

    # Compute Fourier transform
    presum = (all_twop_inputs @ weights[0].T) * exponential
    postsum = np.sum(presum, axis=(2,3))

    fourier_W = postsum

    print(f'fourier_W shape: {fourier_W.shape}')

    return fourier_W


def fourier_new_numpy_fft(runobject):
    epoch=runobject.model_epochs()[-1]
    P=runobject.trainargs.P
    model= load_model(runobject,epoch)[0]
    weights=[p for p in model.parameters() if p.dim()>1]
    all_twop_inputs=make_all_twop(runobject)
    all_psquared_inputs=transform_tensors(all_twop_inputs)
    
    fxy=all_twop_inputs@weights[0].T
    print(f'fxy shape: {fxy.shape}')
    fky=torch.fft.fft2(fxy,dim=(0,1))
    


    fky=fky.detach().numpy()
    fxy=fxy.detach().numpy()

    def norms_plot():
        fxy_normed=np.linalg.norm(fxy,axis=2)
        fky_normed=np.linalg.norm(fky,axis=2)

        fky_real_normed=np.linalg.norm(np.real(fky),axis=2)
        fky_imag_normed=np.linalg.norm(np.imag(fky),axis=2)



        fig=make_subplots(rows=1,cols=3,subplot_titles=['Fourier real','Fourier imag','Comp'])
        fig.add_trace(go.Histogram(x=fky_real_normed.flatten(),name='Fourier real'),row=1,col=1)
        fig.add_trace(go.Histogram(x=fky_imag_normed.flatten(),name='Fourier imag'),row=1,col=2)
        fig.add_trace(go.Histogram(x=fxy_normed.flatten(),name='Comp'),row=1,col=3)

        fig2=make_subplots(rows=1,cols=3,subplot_titles=['Fourier real','Fourier imag','Comp'])
        fig2.add_trace(go.Histogram(x=np.real(fky).flatten().flatten(),name='Fourier real'),row=1,col=1)
        fig2.add_trace(go.Histogram(x=np.imag(fky).flatten().flatten(),name='Fourier imag'),row=1,col=2)
        fig2.add_trace(go.Histogram(x=fxy.flatten().flatten(),name='Comp'),row=1,col=3)


        fig.show()
        fig2.show()
    
    def heatmap_plot():
        fky_real_normed=np.linalg.norm(np.real(fky),axis=2)
        fky_imag_normed=np.linalg.norm(np.imag(fky),axis=2)
        # print(f'fky imag sample {fky_imag_normed[:3,:3]}')
        # exit()
        fxy_normed=np.linalg.norm(fxy,axis=2)

        zmin = min(fky_real_normed.min(),fky_imag_normed.min(), fxy_normed.min())
        zmax = max(fky_real_normed.max(),fky_imag_normed.max(), fxy.max())

        fig=make_subplots(rows=1,cols=3,subplot_titles=['Fourier real','Fourier imag','Comp'])
        fig.add_trace(go.Heatmap(z=fky_real_normed,name='Fourier real'),row=1,col=1)
        fig.add_trace(go.Heatmap(z=fky_imag_normed,name='Fourier imag'),row=1,col=2)
        fig.add_trace(go.Heatmap(z=fxy_normed,name='Comp'),row=1,col=3)
        fig.show()
    
 


    
    
 


    return fky


#fourier_fft=fourier_new_numpy_fft(nongrok_object)

#test=fourier_new_numpy(nongrok_object)
#testfourier=fourier_test(2)
# root=f'/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/fourier'
# filename=f'{root}/fourier.npy'
#np.save(filename,test)


# loaded_ft=np.load(filename,allow_pickle=True)
# print(f'loaded_ft shape: {loaded_ft.shape}')

# loaded_ft_normed=np.linalg.norm(loaded_ft,axis=2)
# print(f'loaded_ft_normed shape: {loaded_ft_normed.shape}')
# twop=make_all_twop(nongrok_object,nongrok_object.trainargs.P)

def get_activations(model, x):
    activations = {}
    hooks = []

    def save_activation(name):
        """Hook function that saves the output of the layer to the activations dict."""
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    def register_hooks():
        """Registers hooks on specified layer types across the entire model."""
        for name, module in model.named_modules():
            # Check specifically for the layer types or names you are interested in
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.MaxPool2d, nn.ReLU)):
                # Adjust name to include full path for clarity, especially useful for layers within ModuleList
                full_name = f'{name} ({module.__class__.__name__})'
                print(f'Registering hook on {full_name}')
                hook = module.register_forward_hook(save_activation(full_name))
                hooks.append(hook)

    def remove_hooks():
        """Removes all hooks from the model."""
        for hook in hooks:
            hook.remove()
        print('All hooks removed.')

    register_hooks()
    # Forward pass to get outputs
    output = model(x)
    remove_hooks()  # Ensure hooks are removed after forward pass

    return activations, output

def remove_nans(tensor):
    # Create a boolean mask for non-NaN values
    mask = ~torch.isnan(tensor)
    
    # Use the mask to select only non-NaN values
    return tensor[mask]

def activations_fourier(runobject):
    P=runobject.trainargs.P
    #1. Make a P^2x2 tensor of kx,ky values - i.e. just 2*pi/P (nx,ny) for nx,ny between 0 and P-1
    kxs=2*torch.pi*torch.arange(P)/P
    ks=torch.cartesian_prod(torch.tensor(kxs),torch.tensor(kxs))
    #2. Make a P^2x2P tensor of fourier transformed input vectors. By writing out the fourier transfrom 
    ####of the (X,Y) vector explicitly you can see that it's just the fourier of x in x and the fourier of y
    ####in y. (pg 460-462 project notes in Remarkable). 
    ### Note that here the entries are complex. So I think the thing to do is to symmetrize to real frequencies
    ### by considering e^{ikx}+e^{-ikx} and e^{ikx}-e^{-ikx} as the components of the fourier transform. This just amounts
    #to using a different fourier basis. I think practically this will mean that instead of the each one-hot
    # half of the two-hot being e^{ikx} it'll be p/2 cos(kx)'s, interleaved with p/s sin(kx)'s. I guess this is what
    #the supreme toad did.
    fourier_vector_cos=torch.zeros(2*P)
    fourier_vector_sin=torch.zeros(2*P)
    kx,ky=ks[0,0],ks[0,1]
    
    
    #I'll just have the redundant frequencies. Can always change later
    fourier_vector_cos[torch.arange(P)]=torch.cos(kx*torch.arange(0,P)/P)
    fourier_vector_sin[torch.arange(P)]=torch.sin(kx*torch.arange(0,P))
    fourier_vector_cos[torch.arange(P,2*P)]=torch.cos(ky*torch.arange(P))
    fourier_vector_sin[torch.arange(P,2*P)]=torch.sin(ky*torch.arange(P))

    fourier_cos=torch.zeros(P*P,2*P)

    # Create a range tensor of shape (P,)
    arange_tensor = torch.arange(P)

    # Expand arange_tensor to shape (P^2, P) for broadcasting
    arange_expanded = arange_tensor.unsqueeze(0).expand(P * P, P)
    fourier_cos[:, :P] = torch.cos(ks[:, 0:1] * arange_expanded)
    fourier_cos[:, P:] = torch.cos(ks[:, 1:2] * arange_expanded)
    
    #for loop version of same code.
    # fourier_cos_2=torch.zeros(P*P,2*P)
    # for i in range(P * P):
    #     # Fill the first P columns with cosines of ks[:, 0] values
    #     fourier_cos_2[i, :P] = torch.cos(ks[i, 0] * torch.arange(P))
    #     # Fill the second P columns with cosines of ks[:, 1] values
    #     fourier_cos_2[i, P:] = torch.cos(ks[i, 1] * torch.arange(P))

    # print(torch.allclose(fourier_cos,fourier_cos_2))


    #It would be good to have a benchmark of what happens in the computational basis
    comp_basis=torch.zeros(P*P,2*P)
    comp_indices=ks/(2*torch.pi/P)
    comp_indices_int=comp_indices.int()
    comp_indices_int[:,1]=comp_indices_int[:,1]+P

    #row_indices = torch.arange(comp_indices_int.shape[1])
    #col_indices=comp_indices_int[range(P*P),row_indices]
    #comp_basis[row_indices,col_indices]=1

    
    comp_basis[torch.arange(P*P), comp_indices_int[torch.arange(P*P),0]] = 1
    comp_basis[torch.arange(P*P), comp_indices_int[torch.arange(P*P),1]] = 1

    
        
    
    #3. Run these through the model and extract the P^2x512 activations for each fourier frequency and each neuron
    test_model=load_model(runobject,runobject.model_epochs()[-1])[0]
    #Note that the hook applies after the layer is done. So it gives the activations at the end of the layer. 
    #Note also that to get the activations after ReLU you need to add ReLU to the list.
    with torch.no_grad():
        test_acts_f,output_f=get_activations(test_model,fourier_cos)#Note the output is already registered as the hook on the last layer
        test_acts_c,output_c=get_activations(test_model,comp_basis)



    
    #sample_neuron=random.randint(0,511)
    sample_neuron=0
    after_ReLU_f=test_acts_f['model.1 (ReLU)']
    after_ReLU_sample_f=after_ReLU_f[:,sample_neuron]

    after_ReLU_c=test_acts_c['model.1 (ReLU)']
    after_ReLU_sample_c=after_ReLU_c[:,sample_neuron]
    

    ks_reshape=ks.reshape(P,P,2)
    comp_indices_reshape=comp_indices_int.reshape(P,P,2)
    after_ReLU_f_reshape=after_ReLU_f.reshape(P,P,after_ReLU_f.shape[-1])
    after_ReLU_c_reshape=after_ReLU_c.reshape(P,P,after_ReLU_c.shape[-1])


    def activation_one_neuron():
        fig=make_subplots(rows=1,cols=2,subplot_titles=[f'Fourier - {sample_neuron} neuron',f'Comp - {sample_neuron} neuron'])
        fig.add_trace(go.Scatter(x=np.arange(after_ReLU_sample_f.shape[0]),y=after_ReLU_sample_f.detach().numpy(),name='Fourier'),row=1,col=1)
        fig.add_trace(go.Scatter(x=np.arange(after_ReLU_sample_c.shape[0]),y=after_ReLU_sample_c.detach().numpy(),name='Comp'),row=1,col=2)
        fig.update_yaxes(title_text=f'Activation on neuron {sample_neuron}')
        fig.update_xaxes(title_text='Fourier frequency (kx,ky)',row=1,col=1)
        fig.update_xaxes(title_text='Comp basis (x,y)',row=1,col=2)
        fig.update_layout(title_text=f'Activations after ReLU on randomly sampled neuron {sample_neuron}')
        fig.show()
    
    def activation_all_neurons():
        #print(f'reshape shape: {after_ReLU_f.reshape(P,P,after_ReLU_f.shape[-1]).shape}')
        eps=1e-8
        max_mean_f=torch.max(after_ReLU_f_reshape,dim=-1).values/(after_ReLU_f_reshape.mean(dim=-1)+eps)
        max_mean_c=torch.max(after_ReLU_c_reshape,dim=-1).values/(after_ReLU_c_reshape.mean(dim=-1)+eps)

        normed_max_mean_f=max_mean_f/torch.mean(max_mean_f)
        normed_max_mean_c=max_mean_c/torch.mean(max_mean_c)

        zmin = min(normed_max_mean_f.min().item(), normed_max_mean_c.min().item())
        zmax = max(normed_max_mean_f.max().item(), normed_max_mean_c.max().item())

        # Add heatmap traces with shared colorscale
        # fig.add_trace(go.Heatmap(z=z1, zmin=zmin, zmax=zmax, colorscale='Viridis'), row=1, col=1)
        # fig.add_trace(go.Heatmap(z=z2, zmin=zmin, zmax=zmax, colorscale='Viridis'), row=1, col=2)


        fig=make_subplots(rows=1,cols=2,subplot_titles=['Fourier','Comp'],shared_yaxes=True,shared_xaxes=True)
        fig.add_trace(go.Heatmap(z=normed_max_mean_f.detach().numpy(),zmin=zmin, zmax=zmax,colorscale='Viridis',name='Fourier'),row=1,col=1)
        fig.add_trace(go.Heatmap(z=normed_max_mean_c.detach().numpy(),zmin=zmin, zmax=zmax,colorscale='Viridis',name='Comp'),row=1,col=2)
        
        fig.update_xaxes(title_text='Fourier kx',row=1,col=1)
        fig.update_yaxes(title_text='Fourier ky',row=1,col=1)

        fig.update_xaxes(title_text='Computational x',row=1,col=2)
        fig.update_yaxes(title_text='Computational y',row=1,col=2)

        fig.update_layout(title_text='Max activation/mean activation for all neurons')



        fig.show()

    def count_within_std(tensor, threshold,times_range=1e-1):
        # Assuming tensor shape is (P, P, d)
        P, _, d = tensor.shape
        
        # Find max and std for each PxP slice
        max_values, _ = torch.max(tensor.reshape(-1, d), dim=0)
        #std_values = torch.std(tensor.reshape(-1, d), dim=0)
        std_values=times_range*(max_values-torch.min(tensor.reshape(-1,d),dim=0).values)
        
        # Create a mask for values within one std of max
        mask = (tensor >= (max_values - std_values)) & (tensor <= (max_values + std_values))
        
        # # Count True values in each slice
        # counts = torch.sum(mask, dim=(0, 1))
        
        # # Set count to zero where max doesn't exceed the threshold
        # counts = torch.where(max_values > threshold, counts, torch.zeros_like(counts))

            # Count True values in each slice
        counts = torch.sum(mask, dim=(0, 1)).float()  # Convert to float
        
        # Set count to NaN where max doesn't exceed the threshold
        counts = torch.where(max_values > threshold, counts, torch.full_like(counts, float('nan')))
        
        return counts
    
    def frequencies_per_neuron():
        #print(f'reshape shape: {after_ReLU_f.reshape(P,P,after_ReLU_f.shape[-1]).shape}')
        eps=1e-8
        times_range=10e-2


        fig=make_subplots(rows=1,cols=2,subplot_titles=['Fourier','Comp'],shared_yaxes=True,shared_xaxes=True,horizontal_spacing=0.08)
        within_sd_f=count_within_std(after_ReLU_f_reshape,1e-2,times_range)
        within_sd_c=count_within_std(after_ReLU_c_reshape,1e-2,times_range)

        within_sd_f=remove_nans(within_sd_f)
        within_sd_c=remove_nans(within_sd_c)


        fig.add_trace(go.Scatter(x=np.arange(within_sd_f.shape[0]),y=within_sd_f.detach().numpy(),name='Fourier'),row=1,col=1)
        fig.add_trace(go.Scatter(x=np.arange(within_sd_c.shape[0]),y=within_sd_c.detach().numpy(),name='Comp'),row=1,col=2)

        fig.update_yaxes(title_text='Number of frequencies')
        fig.update_xaxes(title_text='Neuron index')

        fig.update_layout(title_text=f'Number of frequencies within {100*times_range}% of range of maximum')

        fig.update_layout(
            yaxis=dict(title='Frequencies', showticklabels=True),
            yaxis2=dict(title='Frequencies', showticklabels=True),
            xaxis=dict(title='Index'),
            xaxis2=dict(title='Index'),
            # height=500,
            # width=900,
            showlegend=True
        )

        # Update y-axes to ensure they have the same range
        y_min = min(within_sd_f.min().item(), within_sd_c.min().item())
        y_max = max(within_sd_f.max().item(), within_sd_c.max().item())
        fig.update_yaxes(range=[y_min, y_max])
        fig.show()

    #activation_all_neurons()
    frequencies_per_neuron()
    exit()
        


    #4. Pick a neuron and correlate the activation with the fourier components. Hopefully you'll see that each 
    ####nueuron picks out a single (or a few) frequencies. 
    #If you wanted to, you could then consider linear combinations of frequencies and see if the more you have of a given
    #frequency, the more the neuron activates.

    #5. Then you can do the same plot as you did for Ising with neuron on the x-axis and the maximum corr coefficient on the y-axis

    #If this too doesn't work, then I'm not sure what to do for the modadd interpretability.
    
    return None
test=activations_fourier(grok_object)

exit()



# epoch=nongrok_object.model_epochs()[-1]
# P=nongrok_object.trainargs.P
# model= load_model(nongrok_object,epoch)[0]
# weights=[p for p in model.parameters() if p.dim()>1]
# comp_mat=twop@weights[0].T
# comp_normed=torch.linalg.norm(comp_mat,dim=2).detach().numpy()

# fig=make_subplots(rows=1,cols=2,subplot_titles=['Fourier','Comp'])
# fig.add_trace(go.Histogram(x=loaded_ft_normed.flatten()),row=1,col=1)
# fig.add_trace(go.Histogram(x=comp_normed.flatten()),row=1,col=2)
# fig.show()

# exit()

# with np.printoptions(precision=3, suppress=True, formatter={'float': '{:0.2f}'.format}, linewidth=100):
    
#     print(test)

# test=fourier_new(nongrok_object)
# exit()


def make_fourier_basis(object,p):
    #Basically stolen from Nanda. Does the symmetry argument he used hold here?

    fourier_basis = []
    fourier_basis.append(torch.ones(p)/np.sqrt(p))
    fourier_basis_names = ['Const']
    # Note that if p is even, we need to explicitly add a term for cos(kpi), ie
    # alternating +1 and -1
    for i in range(1, p//2 +1):
        fourier_basis.append(torch.cos(2*torch.pi*torch.arange(p)*i/p))
        fourier_basis.append(torch.sin(2*torch.pi*torch.arange(p)*i/p))
        fourier_basis[-2]/=fourier_basis[-2].norm()
        fourier_basis[-1]/=fourier_basis[-1].norm()
        fourier_basis_names.append(f'cos {i}')
        fourier_basis_names.append(f'sin {i}')
    fourier_basis = torch.stack(fourier_basis, dim=0)
    return fourier_basis, fourier_basis_names

def top_n_indices(tensor, n):
    # Flatten the tensor to a 1D tensor
    flat_tensor = tensor.view(-1)
    
    # Use topk to get the top n values and their indices in the flattened tensor
    top_n_values, top_n_flat_indices = torch.topk(flat_tensor, n)
    
    # Convert the flattened indices back to 2D indices
    top_n_indices_2d = torch.stack(torch.unravel_index(top_n_flat_indices, tensor.shape)).T
    
    return top_n_indices_2d



def fourier_output_matrix(runobject):
    epoch=runobject.model_epochs()[-1]
    P=runobject.trainargs.P
    model= load_model(runobject,epoch)[0]
    weights=[p for p in model.parameters() if p.dim()>1]
    fourier_1d=make_fourier_basis(runobject,P)[0]
    inverse_fourier_1d=torch.linalg.inv(fourier_1d)
    weights_2=weights[1]
    weights_2_fourier=fourier_1d.T@weights_2
    
    weight_2_fourier_normed=weights_2_fourier.pow(2).sum(1).pow(1/2)
    weight_2_normed=weights_2.pow(2).sum(1).pow(1/2)


          

    def matrix_heatmap(weights_2_fourier, weights_2):
        # Convert tensors to numpy arrays
        z1 = weights_2_fourier.detach().numpy()
        z2 = weights_2.detach().numpy()

        # Find the global min and max for the colorscale
        zmin = min(z1.min(), z2.min())
        zmax = max(z1.max(), z2.max())

        # Create subplots with shared y-axis
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True,shared_xaxes=True, subplot_titles=['Fourier', 'Comp'])

        # Add heatmap traces with shared colorscale
        fig.add_trace(go.Heatmap(z=z1, zmin=zmin, zmax=zmax, colorscale='Viridis'), row=1, col=1)
        fig.add_trace(go.Heatmap(z=z2, zmin=zmin, zmax=zmax, colorscale='Viridis'), row=1, col=2)

        # Show the figure
        fig.show()

    #matrix_heatmap(weights_2_fourier,weights_2)

    def histogram(weights_2_fourier, weights_2):
        # Convert tensors to numpy arrays
        x1 = weights_2_fourier.detach().numpy().flatten()
        x2 = weights_2.detach().numpy().flatten()

        # Create subplots with shared y-axis
        fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=['Fourier', 'Comp'])

        # Add histogram traces
        fig.add_trace(go.Histogram(x=x1, histnorm='probability'), row=1, col=1)
        fig.add_trace(go.Histogram(x=x2, histnorm='probability'), row=1, col=2)

        # Show the figure
        fig.show()
    histogram(weights_2_fourier,weights_2)
    

    def ranking():
        weight_2_fourier_normed_ranked=weight_2_fourier_normed[torch.argsort(weight_2_fourier_normed,descending=False)].detach().numpy()
        weight_2_normed_ranked=weight_2_normed[torch.argsort(weight_2_normed,descending=False)].detach().numpy()

        fig=make_subplots(rows=1,cols=1)
        fig.add_trace(go.Scatter(x=np.arange(0,len(weight_2_fourier_normed_ranked)),y=weight_2_fourier_normed_ranked/weight_2_fourier_normed_ranked[0],mode='lines',name='Fourier'),row=1,col=1)
        fig.add_trace(go.Scatter(x=np.arange(0,len(weight_2_normed_ranked)),y=weight_2_normed_ranked/weight_2_normed_ranked[0],mode='lines',name='Comp'),row=1,col=1)
        
        fig.show()

    

    def evaluate_comparison(target_tensor,fourier_transform,n):
        inverse_fourier_trans=torch.linalg.inv(fourier_transform)
        transformed_tensor=fourier_transform.T@target_tensor
        transformed_tensor_normed=transformed_tensor.pow(2).sum(1).pow(1/2)
        losses=[]
        transformed_mean=torch.mean(transformed_tensor)
        for m in range(n):
            top_n_indices=torch.topk(transformed_tensor_normed,m,largest=True).indices
            #print(top_n_indices)
            #print(transformed_tensor[top_n_indices].shape)
            remaining_tensor_mask=torch.ones(transformed_tensor.shape)*transformed_mean
            remaining_tensor_mask[top_n_indices,:]=1
            remaining_tensor=transformed_tensor*remaining_tensor_mask
            
            remaining_tensor_backtransformed=inverse_fourier_trans.T@remaining_tensor
            loss=torch.nn.MSELoss()(remaining_tensor_backtransformed,target_tensor)
            losses.append(loss.item())
        return losses
    


    test_fourier=evaluate_comparison(weights_2,fourier_1d,P+1)
    test_comp=evaluate_comparison(weights_2,torch.eye(fourier_1d.shape[0]),P+1)

    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Scatter(x=np.arange(0,len(test_fourier)),y=test_fourier,mode='lines',name='Fourier'),row=1,col=1)
    fig.add_trace(go.Scatter(x=np.arange(0,len(test_comp)),y=test_comp,mode='lines',name='Comp'),row=1,col=1)
    fig.show()
    exit()

    def plot_pca():
        fig=make_subplots(rows=2,cols=2)

        u,s,v=torch.svd(weights_2_fourier)

        u_f,s_f,v_f=torch.svd(weights_2_fourier)
        proj_matrix_f=torch.matmul(weights_2_fourier, torch.tensor(v_f[:, :2]))

        u_c,s_c,v_c=torch.svd(weights_2)
        proj_matrix_c=torch.matmul(weights_2, torch.tensor(v_c[:, :2]))
        
        proj_matrix_c=proj_matrix_c.detach().numpy()
        proj_matrix_f=proj_matrix_f.detach().numpy()

        fig.add_trace(go.Scatter(x=proj_matrix_c[:,0],y=proj_matrix_c[:,1],mode='markers',name='Comp'),row=1,col=1)
        fig.add_trace(go.Scatter(x=proj_matrix_f[:,0],y=proj_matrix_f[:,1],mode='markers',name='Fourier'),row=1,col=2)

        fig.add_trace(go.Scatter(x=np.arange(0,len(s_c)),y=s_c.detach().numpy(),mode='lines',name='Comp SVD'),row=2,col=1)
        fig.add_trace(go.Scatter(x=np.arange(0,len(s_f)),y=s_f.detach().numpy(),mode='lines',name='Fourier SVD'),row=2,col=1)

        fig.add_trace(go.Scatter(x=np.arange(0,len(weight_2_fourier_normed)),y=weight_2_fourier_normed.detach().numpy()/torch.max(weight_2_fourier_normed).item(),mode='lines',name='Fourier Normed'),row=2,col=2)
        fig.add_trace(go.Scatter(x=np.arange(0,len(weight_2_normed)),y=weight_2_normed.detach().numpy()/torch.max(weight_2_normed).item(),mode='lines',name='Comp Normed'),row=2,col=2)


        fig.show()


    # all_twop_inputs=make_all_twop(runobject)
    # all_psquared_inputs=transform_tensors(all_twop_inputs)

    

    

# test_2=fourier_output_matrix(grok_object)
# exit()

    # kxs=torch.zeros(P,P)
    # for i in range(P):
    #     kxs[i,i]=i
    # kxs=((2*torch.pi)/P)*kxs
    # kys=kxs.clone()
    
    
    
    
    







def make_fourier_basis(object,p,arange_p=p):
    #Basically stolen from Nanda. Does the symmetry argument he used hold here?

    fourier_basis = []
    fourier_basis.append(torch.ones(p)/np.sqrt(p))
    fourier_basis_names = ['Const']
    # Note that if p is even, we need to explicitly add a term for cos(kpi), ie
    # alternating +1 and -1
    for i in range(1, p//2 +1):
        fourier_basis.append(torch.cos(2*torch.pi*torch.arange(arange_p)*i/p))
        fourier_basis.append(torch.sin(2*torch.pi*torch.arange(arange_p)*i/p))
        fourier_basis[-2]/=fourier_basis[-2].norm()
        fourier_basis[-1]/=fourier_basis[-1].norm()
        fourier_basis_names.append(f'cos {i}')
        fourier_basis_names.append(f'sin {i}')
    fourier_basis = torch.stack(fourier_basis, dim=0)
    return fourier_basis, fourier_basis_names

def make_fourier_basis2(object,p):
    #Basically stolen from Nanda. Does the symmetry argument he used hold here?

    fourier_basis_cos = []
    fourier_basis_cos.append(torch.ones(p)/np.sqrt(p))
    fourier_basis_sin=[]
    fourier_basis_sin.append(torch.zeros(p)/np.sqrt(p))
    fourier_basis_cos_names = ['Const, p=0']
    fourier_basis_sin_names=['Const, p=0']
    # Note that if p is even, we need to explicitly add a term for cos(kpi), ie
    # alternating +1 and -1
    for i in range(1, p):
        fourier_basis_cos.append(torch.cos(2*torch.pi*torch.arange(p)*i/p))
        fourier_basis_sin.append(torch.sin(2*torch.pi*torch.arange(p)*i/p))
        fourier_basis_cos[-1]/=fourier_basis_cos[-1].norm()
        fourier_basis_sin[-1]/=fourier_basis_sin[-1].norm()
        fourier_basis_cos_names.append(f'cos {i}')
        fourier_basis_sin_names.append(f'sin {i}')
    fourier_basis_cos = torch.stack(fourier_basis_cos, dim=0)
    fourier_basis_sin = torch.stack(fourier_basis_sin, dim=0)
    return fourier_basis_cos, fourier_basis_cos_names,fourier_basis_sin,fourier_basis_sin_names



# fourier_basis, fourier_basis_names=make_fourier_basis(nongrok_object,2*nongrok_object.trainargs.P)


# # print(f'fourier_basis shape: {fourier_basis.shape}')
# # fig=make_subplots(rows=1,cols=1)
# # fig.add_trace(go.Heatmap(z=fourier_basis@fourier_basis.T),row=1,col=1)
# # fig.show()

# epoch=nongrok_object.model_epochs()[-1]
# model= load_model(nongrok_object,epoch)[0]
# trainloader=nongrok_object.train_loader
# for batch in trainloader:
#     inputs,labels=batch
#     test_input=inputs[0]
#     break





# weights=[p for p in model.parameters() if p.dim()>1]

# fourier_transformed_input_weights= weights[0]@fourier_basis.T
# print(f'fourier transformed input weights shape: {fourier_transformed_input_weights.shape}')

# fourier_input_norms=fourier_transformed_input_weights.pow(2).sum(0)

def transform_input(twop_vectors):
    print(twop_vectors.shape)
    p=twop_vectors.shape[0]//2
    psquared=torch.zeros(p,p)
    indices=torch.nonzero(twop_vectors)
    sortedindices,indices_of_indices=indices.sort()
    psquared[sortedindices[0],sortedindices[1]%p]=1
    
    #expected_psquared=torch.nonzero(twop_vectors)-torch.tensor([[0],[p]])
    # print(f'twop vectors nonzero {torch.nonzero(twop_vectors)}')
    # print(f'psquared vectors nonzero {torch.nonzero(psquared)}')
    # print(f'Expected psquared vector {expected_psquared}')
    

    return None



def transform_tensors_inverse(psquared_vectors):
    N,P,P=psquared_vectors.shape
    sample_index,first_index,second_index=torch.nonzero(psquared_vectors,as_tuple=True)
    second_index_added=second_index+P
    twop_vectors=torch.zeros(N,2*P)
    twop_vectors[sample_index,first_index]=1
    twop_vectors[sample_index,second_index_added]=1

    backtonpp=torch.zeros(P,P,2*P)
    backtonpp[first_index,second_index,first_index]=1
    backtonpp[first_index,second_index,second_index_added]=1



    
    return twop_vectors,backtonpp
# fourier_basis=make_fourier_basis(nongrok_object,nongrok_object.trainargs.P)[0]

def transform_tensors_inverse2(psquared_vectors):
    N,P,P=psquared_vectors.shape
    twoPtensor=torch.zeros(N,2*P)
    for N in range(N):
        for i in range(P):
            for j in range(P):
                first_half=psquared_vectors[N,:,j]
                second_half=psquared_vectors[N,i,:]
                twopentry=torch.cat([first_half,second_half],dim=0)
                twoPtensor[N]=twopentry
    print(f'twoP sample: {torch.nonzero(twoPtensor[0])}')
    print(f'psquared sample: {torch.nonzero(psquared_vectors[0])}')


def transform_tensors_inverse2(psquared_vectors):
    N, P, _ = psquared_vectors.shape
    print(N,P)
    exit()
    
    # Prepare an empty tensor for the results
    twoPtensor = torch.zeros(N, 2 * P)

    # Use broadcasting and advanced indexing to replace the loops
    first_half = psquared_vectors.transpose(1, 2).reshape(N, -1)  # Shape: (N, P * P)
    second_half = psquared_vectors.reshape(N, -1)                 # Shape: (N, P * P)
    
    # Stack the first and second half along the last dimension
    twoPtensor = torch.cat([first_half, second_half], dim=-1)     # Shape: (N, 2 * P)

    return twoPtensor

def transform_tensors_inverse2_vectorized(psquared_vectors):
    N, P, _ = psquared_vectors.shape
    
    # Create the first half by repeating each row P times
    first_half = psquared_vectors.repeat_interleave(P, dim=1)
    
    # Create the second half by repeating the whole tensor P times and reshaping
    second_half = psquared_vectors.repeat(1, P, 1).reshape(N, -1)
    
    # Concatenate the two halves
    twoPtensor = torch.cat([first_half, second_half], dim=1)
    
    return twoPtensor

twopvectors=make_all_twop(nongrok_object)
psquaredvectors=transform_tensors(twopvectors)
test=transform_tensors_inverse2_vectorized(psquaredvectors)
print(f'test shape: {test.shape}')
exit()

print(f'twop sample: {torch.nonzero(twopvectors[0])}')
print(f'psquared sample: {torch.nonzero(psquaredvectors[0])}')
print(f'test sample: {torch.nonzero(test[0])}')

exit()

        
        

# fourier_basis_2=make_fourier_basis(nongrok_object,2*nongrok_object.trainargs.P)[0]

# transformed_tensor=transform_tensors(inputs)
#transformed_tensors_fourier=fourier_basis@transformed_tensor@fourier_basis.T
#print(transformed_tensors_fourier.shape)

# all_twop_inputs=[]
# for i in range(0,nongrok_object.trainargs.P):
#     for j in range(0,nongrok_object.trainargs.P):
#         twop_input=torch.zeros(2*nongrok_object.trainargs.P)
#         twop_input[i]=1
#         twop_input[j+nongrok_object.trainargs.P]=1
#         all_twop_inputs.append(twop_input)
# all_twop_inputs_1=torch.stack(all_twop_inputs)




def batch_matrix_operation(A, F):
    """
    Apply F^T A F to each PxP matrix in the NxPxP tensor A.
    
    Args:
    A: Tensor of shape (N, P, P)
    F: Matrix of shape (P, P)
    
    Returns:
    result: Tensor of shape (N, P, P)
    """
    # Compute F^T
    F_transpose = F.t()
    
    # Perform the matrix multiplications
    # (N, P, P) @ (P, P) -> (N, P, P)
    temp = torch.matmul(A, F)
    
    # (P, P) @ (N, P, P) -> (N, P, P)
    result = torch.matmul(F_transpose, temp)
    
    return result

def convert_output_NPP(input_tensor,indexing_tensor):
    N, O = input_tensor.shape
    P=int(np.sqrt(N))
    output_tensor = torch.zeros(P,P,O)
    indexing_tensor_indices=torch.nonzero(indexing_tensor)
    output_tensor[indexing_tensor_indices[:,1], indexing_tensor_indices[:,2]] = input_tensor[indexing_tensor_indices[:,0]]
    
    

    # Create a tensor of shape (N, P, P)

    
    return output_tensor

# def make_all_twop2(object):


# all_twop_inputs_2=make_all_twop(nongrok_object)
# print(f'all_twop_inputs_2 shape: {all_twop_inputs_2.shape}')

# all_psquared_inputs=transform_tensors(all_twop_inputs_2)
# print(f'all_psquared_inputs shape: {all_psquared_inputs.shape}')

# print(fourier_basis.shape)



# #can change back to fourier_basis from fourier_basis_new
# fourier_all_psquared=fourier_basis @ all_psquared_inputs@ fourier_basis.T
# print(f'fourier_all_psquared shape: {fourier_all_psquared.shape}')

# transformed_fourier= torch.cat([fourier_all_psquared[:, 0, :], fourier_all_psquared[:, 1, :]], dim=1)
# print(f'fourier shape: {fourier_all_psquared.shape}')
# print(f'transformed_fourier shape: {transformed_fourier.shape}')

# print(f'fourier shape: {fourier_all_psquared.shape}')
# all_fourier_inputs=transform_tensors(fourier_all_psquared)
# print(f'all_fourier_inputs shape: {all_fourier_inputs.shape}')


#print(f'fourier_all_psquared shape: {fourier_all_psquared.shape}')


# input_matrix_outputs=all_twop_inputs_2@(weights[0].T)
# fourier_matrix_outputs=transformed_fourier@(weights[0].T)
# #At this point I need to reshape to make it a PxP matrix.
# #fourier_matrix_outputs.reshape(nongrok_object.trainargs.P,nongrok_object.trainargs.P,fourier_matrix_outputs.shape[-1])
# print('fourier matrix outputs shape: ',fourier_matrix_outputs.shape)
# fourier_matrix_outputs=convert_output_NPP(fourier_matrix_outputs,all_psquared_inputs)


# normed_output=input_matrix_outputs.pow(2).sum(-1)
# fourier_normed_output=fourier_matrix_outputs.pow(2).sum(-1)
# print('fourier normed output shape: ',fourier_normed_output.shape)

# print(input_matrix_outputs.shape)
# print(normed_output.shape)
# print(fourier_normed_output.shape)


def zero_below_top_n(tensor, n):
    N, P, _ = tensor.shape
    # Flatten each PxP tensor to get shape (N, P*P)
    flat_tensor = tensor.view(N, -1)
    
    # Get the indices of the top n elements in each row
    top_n_values, top_n_indices = torch.topk(flat_tensor, n, dim=1)
    
    # Create a mask that zeroes out all elements not in the top n
    mask = torch.zeros_like(flat_tensor).scatter_(1, top_n_indices, 1)
    
    # Apply the mask
    flat_tensor = flat_tensor * mask
    
    # Reshape back to NxPxP
    return flat_tensor.view(N, P, P)

def top_n_indices(tensor, n):
    # Flatten the tensor to a 1D tensor
    flat_tensor = tensor.view(-1)
    
    # Use topk to get the top n values and their indices in the flattened tensor
    top_n_values, top_n_flat_indices = torch.topk(flat_tensor, n)
    
    # Convert the flattened indices back to 2D indices
    top_n_indices_2d = torch.stack(torch.unravel_index(top_n_flat_indices, tensor.shape)).T
    
    return top_n_indices_2d

def evaluate_original(runobject):
    fourier_1d=make_fourier_basis(runobject,runobject.trainargs.P)[0]
    inverse_fourier_1d=torch.linalg.inv(fourier_1d)

    epoch=runobject.model_epochs()[-1]
    model= load_model(runobject,epoch)[0]
    weights=[p for p in model.parameters() if p.dim()>1]
    all_twop_inputs=make_all_twop(runobject)
    all_psquared_inputs=transform_tensors(all_twop_inputs)
    
    
    # print(f'all two p sample {torch.nonzero(all_twop_inputs[0,1])}')
    # print(f'all two p sample {torch.nonzero(all_twop_inputs[1,0])}')
    
    # print(f'all psquared sample {torch.nonzero(all_psquared_inputs[0])}')

    all_psquared_back=transform_tensors_inverse(all_psquared_inputs)[1]
    # print(f'all psquared back sample {torch.nonzero(all_psquared_back[0,1])}')
    # print(f'all psquared back sample {torch.nonzero(all_psquared_back[1,0])}')
    # print(f'all psquared back sample shape {all_psquared_back.shape}')
    # print(f'inverse works? {torch.allclose(all_psquared_back,all_twop_inputs)}')


    exit()    

    fourier_inputs=fourier_1d.T@all_psquared_inputs@fourier_1d
    
    
    
    transformed_fourier_inputs=torch.cat([fourier_inputs[:, 0, :], fourier_inputs[:, 1, :]], dim=1)
    transformed_comp_inputs=torch.cat([all_psquared_inputs[:, 0, :], all_psquared_inputs[:, 1, :]], dim=1)
    sample=transformed_comp_inputs[0]
    print(f'transformed comp inputs shape: {sample.shape}')
    
    print(f'transformed comp inputs nonzero 1 {torch.nonzero(sample)}')
    print(f'sample nonzero 2 {sample[:5]}')
    exit()
    
    transformed_fourier_weights=transformed_fourier_inputs@weights[0].T
    transformed_fourier_weights=convert_output_NPP(transformed_fourier_weights,all_psquared_inputs)

    transformed_comp_weights=transformed_comp_inputs@weights[0].T
    transformed_comp_weights=convert_output_NPP(transformed_comp_weights,all_psquared_inputs)

    #inv_fourier=inverse_fourier_1d.T@fourier_inputs@inverse_fourier_1d
    #print(f'inv fourier gives right answer? {torch.allclose(inv_fourier,all_psquared_inputs,atol=1e-5)}')
    
    def top_n_comparison(ppd_tensor,inverse_transform,target_tensor,ns):
        #print(f'input pdd tensor shape: {ppd_tensor.shape}')
        hidden_normed_tensor=(ppd_tensor.pow(2).sum(-1)).pow(1/2)
        cross_entropies=[]
        for n in range(1,ns):
            #print(f'hidden normed tensor shape: {hidden_normed_tensor.shape}')
            topn_indices=top_n_indices(hidden_normed_tensor,n)
            #print(f'topn indices shape: {topn_indices.shape}')
            topn_indices_mask=torch.zeros(ppd_tensor.shape)
            topn_indices_mask[topn_indices[:,0],topn_indices[:,1],:]=1
            # print(f'topn indices mask shape: {topn_indices_mask.shape}')
            # print(f'topn indices mask nonzero: {torch.nonzero(topn_indices_mask[:,:,0])}')
            # print(f'top n indices: {topn_indices}')
            effective_tensor=ppd_tensor*topn_indices_mask
            with torch.no_grad():
                cross_entropy=nn.MSELoss()(effective_tensor,ppd_tensor)
                cross_entropies.append(cross_entropy.item())
            

        
        

        

        


        return cross_entropies

    test=top_n_comparison(transformed_fourier_weights,inverse_fourier_1d,transformed_comp_weights,1000)
    test2=top_n_comparison(transformed_comp_weights,inverse_fourier_1d,transformed_comp_weights,1000)
    
    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Scatter(x=np.arange(1,1000),y=test,mode='lines',name='Fourier'),row=1,col=1)
    fig.add_trace(go.Scatter(x=np.arange(1,1000),y=test2,mode='lines',name='Comp'),row=1,col=1)
    fig.show()
    exit()

    print(f'fourier reconstruction cross-entropy: \n {test}')
    print(f'comp reconstruction cross-entropy: \n {test2}')
    exit()

    transformed_all_fourier_inputs=torch.cat([fourier_inputs[:, 0, :], fourier_inputs[:, 1, :]], dim=1)
    topn_fourier=zero_below_top_n(all_psquared_inputs,10)
    transformed_topn_inputs=torch.cat([topn_fourier[:, 0, :], topn_fourier[:, 1, :]], dim=1)
    transformed_weights=transformed_topn_inputs@weights[0].T
    transformed_weights=transformed_inputs@weights[0]

# test=evaluate_original(nongrok_object)
# exit()

#Neel nanda's functions

nanda_fourier_basis,nanda_fourier_basis_names=make_fourier_basis(nongrok_object,nongrok_object.trainargs.P)
def fft1d(tensor):
    # Converts a tensor with dimension p into the Fourier basis
    return tensor @ nanda_fourier_basis.T

def fourier_2d_basis_term(x_index, y_index):
    # Returns the 2D Fourier basis term corresponding to the outer product of
    # the x_index th component in the x direction and y_index th component in the
    # y direction
    # Returns a 1D vector of length p^2
    return (nanda_fourier_basis[x_index][:, None] * nanda_fourier_basis[y_index][None, :]).flatten()

def fft2d(mat,p):
    # Converts a pxpx... or batch x ... tensor into the 2D Fourier basis.
    # Output has the same shape as the original
    shape = mat.shape
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=p, y=p)
    fourier_mat = torch.einsum('xyz,fx,Fy->fFz', mat, nanda_fourier_basis, nanda_fourier_basis)
    return fourier_mat.reshape(shape)

def fft2d_inverse(mat,p):
    # Converts a pxpx... or batch x ... tensor into the 2D Fourier basis.
    # Output has the same shape as the original
    shape = mat.shape
    mat = einops.rearrange(mat, '(x y) ... -> x y (...)', x=p, y=p)
    fourier_basis_inv=torch.linalg.inv(nanda_fourier_basis)
    fourier_mat = torch.einsum('xyz,fx,Fy->fFz', mat, fourier_basis_inv, fourier_basis_inv)
    return fourier_mat.reshape(shape)

def analyse_fourier_2d(tensor, p,top_k=10):
    # Processes a (p,p) or (p*p) tensor in the 2D Fourier Basis, showing the
    # top_k terms and how large a fraction of the variance they explain
    values, indices = tensor.flatten().pow(2).sort(descending=True)
    rows = []
    total = values.sum().item()
    for i in range(top_k):
        rows.append([tensor.flatten()[indices[i]].item(),
                     values[i].item()/total,
                     values[:i+1].sum().item()/total,
                     nanda_fourier_basis_names[indices[i].item()//p],
                     nanda_fourier_basis_names[indices[i]%p]])
    #display(pd.DataFrame(rows, columns=['Coefficient', 'Frac explained', 'Cumulative frac explained', 'x', 'y']))

def get_2d_fourier_component(tensor, x, y):
    # Takes in a batch x ... tensor and projects it onto the 2D Fourier Component
    # (x, y)
    vec = fourier_2d_basis_term(x, y).flatten()
    return vec[:, None] @ (vec[None, :] @ tensor)

def get_component_cos_xpy(tensor, freq, collapse_dim=False):
    # Gets the component corresponding to cos(freq*(x+y)) in the 2D Fourier basis
    # This is equivalent to the matrix cos((x+y)*freq*2pi/p)
    cosx_cosy_direction = fourier_2d_basis_term(2*freq-1, 2*freq-1).flatten()
    sinx_siny_direction = fourier_2d_basis_term(2*freq, 2*freq).flatten()
    # Divide by sqrt(2) to ensure it remains normalised
    cos_xpy_direction = (cosx_cosy_direction - sinx_siny_direction)/np.sqrt(2)
    # Collapse_dim says whether to project back into R^(p*p) space or not
    if collapse_dim:
        return (cos_xpy_direction @ tensor)
    else:
        return cos_xpy_direction[:, None] @ (cos_xpy_direction[None, :] @ tensor)

def get_component_sin_xpy(tensor, freq, collapse_dim=False):
    # Gets the component corresponding to sin((x+y)*freq*2pi/p) in the 2D Fourier basis
    sinx_cosy_direction = fourier_2d_basis_term(2*freq, 2*freq-1).flatten()
    cosx_siny_direction = fourier_2d_basis_term(2*freq-1, 2*freq).flatten()
    sin_xpy_direction = (sinx_cosy_direction + cosx_siny_direction)/np.sqrt(2)
    if collapse_dim:
        return (sin_xpy_direction @ tensor)
    else:
        return sin_xpy_direction[:, None] @ (sin_xpy_direction[None, :] @ tensor)


def comp_and_fourier_norms(runobject):
    epoch=runobject.model_epochs()[-1]
    model= load_model(runobject,epoch)[0]
    weights=[p for p in model.parameters() if p.dim()>1]


    all_twop_inputs=make_all_twop(runobject)
    all_psquared_inputs=transform_tensors(all_twop_inputs)
    

    fourier_basis=make_fourier_basis(runobject,runobject.trainargs.P)[0]
    



    fourier_all_psquared=fourier_basis.T @ all_psquared_inputs@ fourier_basis

    #transformed_fourier= torch.cat([fourier_all_psquared[:, 0, :], fourier_all_psquared[:, 1, :]], dim=1)
    transformed_fourier=transform_tensors_inverse(fourier_all_psquared)[1]
    print(f'transformed fourier shape: {transformed_fourier.shape}')
    
    print(torch.allclose(transformed_fourier,all_twop_inputs,atol=1e-2))
    exit()

    input_matrix_outputs=all_twop_inputs@(weights[0].T)
    fourier_matrix_outputs=transformed_fourier@(weights[0].T)

    #fourier_matrix_outputs=convert_output_NPP(fourier_matrix_outputs,all_psquared_inputs)
    input_svd=torch.linalg.svdvals(input_matrix_outputs)
    fourier_svd=torch.linalg.svdvals(fourier_matrix_outputs)
    print(f'transformed fourier sample: {torch.nonzero(transformed_fourier[0,0])}')
    print(f'input sample: {torch.nonzero(transformed_fourier[0,0])}')
    exit()
    print(f'same svd: {torch.allclose(input_matrix_outputs,fourier_matrix_outputs,atol=1e-3)}')
    exit()

    fourier_basis_cos,_,fourier_basis_sin,_ = make_fourier_basis2(runobject,runobject.trainargs.P)
    f2_cos=fourier_basis_cos @ all_psquared_inputs@ fourier_basis_cos.T
    tf2_cos= torch.cat([f2_cos[:, 0, :], f2_cos[:, 1, :]], dim=1)
    fmo_cos=tf2_cos@(weights[0].T)
    fmo_cos=convert_output_NPP(fmo_cos,all_psquared_inputs)



    f2_sin=fourier_basis_sin @ all_psquared_inputs@ fourier_basis_sin.T
    tf2_sin= torch.cat([f2_sin[:, 0, :], f2_sin[:, 1, :]], dim=1)
    fmo_sin=tf2_sin@(weights[0].T)
    fmo_sin=convert_output_NPP(fmo_sin,all_psquared_inputs)
    



    
    fourier_matrix_outputs_layer2=fourier_basis.T@weights[1]
    output_layer2=weights[1]
    

    normed_output=(input_matrix_outputs.pow(2).sum(-1)).pow(1/2)
    fourier_normed_output=(fourier_matrix_outputs.pow(2).sum(-1)).pow(1/2)
    fourier_normed_output_layer2=(fourier_matrix_outputs_layer2.pow(2).sum(0)).pow(1/2)
    normed_output_layer2=weights[1].pow(2).sum(-1).pow(1/2)


    
    fmo_cos_normed_output=(fmo_cos.pow(2).sum(-1)).pow(1/2)
    fmo_sin_normed_output=(fmo_sin.pow(2).sum(-1)).pow(1/2)

    

    return normed_output,fourier_normed_output,fmo_cos_normed_output,fmo_sin_normed_output,output_layer2,fourier_matrix_outputs_layer2

# testcomp=comp_and_fourier_norms(nongrok_object)
# exit()

def fourier_pca(grokobject,nongrokobject=None):
    objects=1
    comp_output=[]
    fourier_output=[]

    if nongrok_object is not None:
        objects+=1
        nongrok_comp_output,nongrok_fourier_output,nongrok_fmo_cos,nongrok_fmo_sin,nongrok_comp_output_layer2,nongrok_fourier_output_layer2=comp_and_fourier_norms(nongrok_object)
        comp_output.append(nongrok_comp_output)
        fourier_output.append(nongrok_fourier_output)

    grok_comp_output,grok_fourier_output,grok_fmo_cos,grok_fmo_sin,grok_comp_output_layer2,grok_fourier_output_layer2=comp_and_fourier_norms(grokobject)
    comp_output.append(grok_comp_output)
    fourier_output.append(grok_fourier_output)



    fig=make_subplots(rows=2,cols=3,subplot_titles=['Computational basis PCA - Learn','Computational basis PCA - Grok','Computational basis singular values','Fourier basis PCA - Learn','Fourier basis PCA - Grok','Fourier basis singular values'])
    
    u_nongrok_comp,s_nongrok_comp,v_nongrok_comp=torch.svd(nongrok_comp_output)
    nongrok_proj_matrix_comp=torch.matmul(nongrok_comp_output, torch.tensor(v_nongrok_comp[:, :2]))

    u_grok_comp,s_grok_comp,v_grok_comp=torch.svd(grok_comp_output)
    grok_proj_matrix_comp=torch.matmul(grok_comp_output, torch.tensor(v_grok_comp[:, :2]))

    u_nongrok_fourier,s_nongrok_fourier,v_nongrok_fourier=torch.svd(nongrok_fourier_output)
    nongrok_proj_matrix_fourier=torch.matmul(nongrok_fourier_output, torch.tensor(v_nongrok_fourier[:, :2]))

    u_grok_fourier,s_grok_fourier,v_grok_fourier=torch.svd(grok_fourier_output)
    grok_proj_matrix_fourier=torch.matmul(grok_fourier_output, torch.tensor(v_grok_fourier[:, :2]))



    fig.add_trace(go.Scatter(x=nongrok_proj_matrix_comp[:,0].detach().numpy(), y=nongrok_proj_matrix_comp[:,1].detach().numpy(), name='Learn', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter(x=grok_proj_matrix_comp[:,0].detach().numpy(), y=grok_proj_matrix_comp[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='red'),showlegend=True), row=1, col=2)

    fig.add_trace(go.Scatter(x=nongrok_proj_matrix_fourier[:,0].detach().numpy(), y=nongrok_proj_matrix_fourier[:,1].detach().numpy(), name='Learn', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=2, col=1)
    fig.add_trace(go.Scatter(x=grok_proj_matrix_fourier[:,0].detach().numpy(), y=grok_proj_matrix_fourier[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='red'),showlegend=True), row=2, col=2)
    

    fig.add_trace(go.Scatter(x=np.arange(0,len(s_nongrok_comp)),y=s_nongrok_comp.detach().numpy(),name='Learn',mode='lines',line=dict(color='blue')),row=1,col=3)
    fig.add_trace(go.Scatter(x=np.arange(0,len(s_grok_comp)),y=s_grok_comp.detach().numpy(),name='Grok',mode='lines',line=dict(color='red')),row=1,col=3)

    

    fig.add_trace(go.Scatter(x=np.arange(0,len(s_nongrok_fourier)),y=s_nongrok_fourier.detach().numpy(),name='Learn',mode='lines',line=dict(color='blue')),row=2,col=3)
    fig.add_trace(go.Scatter(x=np.arange(0,len(s_grok_fourier)),y=s_grok_fourier.detach().numpy(),name='Grok',mode='lines',line=dict(color='red')),row=2,col=3)
    
    


    fig.update_yaxes(title_text='Singular value',col=3)
    fig.update_xaxes(title_text='Rank',col=3)

    fig.update_layout(title_text='PCA of norms')
    fig.update_xaxes(title_text='PCA 1 of norms - comp',row=1)
    fig.update_yaxes(title_text='PCA 2 of norms - comp',row=1)

    fig.update_xaxes(title_text='PCA 1 of norms - fourier',row=2)
    fig.update_yaxes(title_text='PCA 2 of norms - fourier',row=2)

    return fig

testfig=fourier_pca(grok_object,nongrok_object).show()
exit()


def matrix_heatmap(grokobject,nongrokobject=None):
    objects=1
    comp_output=[]
    fourier_output=[]

    if nongrok_object is not None:
        objects+=1
        nongrok_comp_output,nongrok_fourier_output,nongrok_fmo_cos,nongrok_fmo_sin,nongrok_comp_output_layer2,nongrok_fourier_output_layer2=comp_and_fourier_norms(nongrok_object)
        comp_output.append(nongrok_comp_output)
        fourier_output.append(nongrok_fourier_output)

    grok_comp_output,grok_fourier_output,grok_fmo_cos,grok_fmo_sin,grok_comp_output_layer2,grok_fourier_output_layer2=comp_and_fourier_norms(grokobject)
    comp_output.append(grok_comp_output)
    fourier_output.append(grok_fourier_output)



    #layer 1 `rollout' fourier:
    epoch=grokobject.model_epochs()[-1]
    model= load_model(grokobject,epoch)[0]
    weights=[p for p in model.parameters() if p.dim()>1]
    fourier_basis=make_fourier_basis(grokobject,2*grokobject.trainargs.P)[0]
    fourier_matrix_outputs_layer1=(weights[0])@fourier_basis.T
    grok_fourier_layer1=(fourier_matrix_outputs_layer1.pow(2).sum(0)).pow(1/2)
    grok_comp_layer1=(weights[0].pow(2).sum(0)).pow(1/2)



    
    nongrok_model= load_model(nongrokobject,epoch)[0]
    nongrok_weights=[p for p in nongrok_model.parameters() if p.dim()>1]
    
    nongrok_fourier_matrix_outputs_layer1=(nongrok_weights[0])@fourier_basis.T
    nongrok_fourier_layer1=(nongrok_fourier_matrix_outputs_layer1.pow(2).sum(0)).pow(1/2)
    nongrok_comp_layer1=(nongrok_weights[0].pow(2).sum(0)).pow(1/2)

    
    

    # Assuming objects, comp_output, and fourier_output are defined
    fig = make_subplots(
        rows=2, cols=objects,
        subplot_titles=['Computational basis norms - Learn ','Computational basis norms - Grok', 'Fourier basis norms - Learn', 'Fourier basis norms - Grok'][:2*objects],
        shared_yaxes=True  # Ensure y-axes are shared within each row
    )

    for object_index in range(objects):
        normed_output = comp_output[object_index]
        fourier_normed_output = fourier_output[object_index] 
        
        # Add the heatmap for the first row (computational basis norms)
        fig.add_trace(go.Heatmap(
            z=normed_output.detach().numpy(),
            x=np.arange(0, normed_output.shape[0]),
            y=np.arange(0, normed_output.shape[1]),
            coloraxis='coloraxis1',  # Link to the first color axis
            showscale=False  # Hide individual color scale
        ), row=1, col=1 + object_index)

        # Add the heatmap for the second row (Fourier basis norms)
        fig.add_trace(go.Heatmap(
            z=fourier_normed_output.detach().numpy(),
            x=np.arange(0, fourier_normed_output.shape[0]),
            y=np.arange(0, fourier_normed_output.shape[1]),
            coloraxis='coloraxis2',  # Link to the second color axis
            showscale=False  # Hide individual color scale
        ), row=2, col=1 + object_index)

    # Add shared color scales for each row
    fig.update_layout(
        coloraxis1=dict(colorscale='Viridis'),  # Define the color scale for the first row
        coloraxis2=dict(colorscale='Viridis'),  # Define the color scale for the second row
    )

    # Add the color bars for each row
    fig.update_layout(
        coloraxis1_colorbar=dict(
            title="Computational Norms",  # Title for the first row color bar
            #x=1.1,
            y=0.83,  # Positioning the color bar centrally for the first row
            len=0.45  # Length of the color bar (fraction of plot height)
        ),
        coloraxis2_colorbar=dict(
            title="Fourier Norms",  # Title for the second row color bar
            #x=1.1,
            y=0.2,  # Positioning the color bar centrally for the second row
            len=0.45  # Length of the color bar (fraction of plot height)
        )
    )

    # Show the figure

    fig1=make_subplots(rows=1,cols=2,subplot_titles=['Computational basis norms - Learn','Computational basis norms - Grok','Fourier basis norms - Learn','Fourier basis norms - Grok'])
    
    # fig2.add_trace(go.Scatter(x=np.arange(0,nongrok_comp_output_layer2.shape[0]),y=nongrok_comp_output_layer2.detach().numpy()/np.max(nongrok_comp_output_layer2.detach().numpy()),name='Learn - Comp',mode='lines',line=dict(color='blue')),row=1,col=1)
    # fig2.add_trace(go.Scatter(x=np.arange(0,grok_comp_output_layer2.shape[0]),y=grok_comp_output_layer2.detach().numpy()/np.max(grok_comp_output_layer2.detach().numpy()),name='Grok - Comp',mode='lines',line=dict(color='red')),row=1,col=2)
    
    # fig2.add_trace(go.Scatter(x=np.arange(0,nongrok_fourier_output_layer2.shape[0]),y=nongrok_fourier_output_layer2.detach().numpy()/np.max(nongrok_fourier_output_layer2.detach().numpy()),name='Learn - Fourier',mode='lines',line=dict(color='blue')),row=2,col=1)
    # fig2.add_trace(go.Scatter(x=np.arange(0,grok_fourier_output_layer2.shape[0]),y=grok_fourier_output_layer2.detach().numpy()/np.max(grok_fourier_output_layer2.detach().numpy()),name='Grok - Fourier',mode='lines',line=dict(color='red')),row=2,col=2)
    
    fig1.add_trace(go.Scatter(x=np.arange(0,nongrok_comp_layer1.shape[0]),y=nongrok_comp_layer1.detach().numpy()/(np.max(nongrok_comp_layer1.detach().numpy())),name='Learn - Comp',mode='lines',line=dict(color='blue')),row=1,col=1)
    fig1.add_trace(go.Scatter(x=np.arange(0,nongrok_fourier_layer1.shape[0]),y=nongrok_fourier_layer1.detach().numpy()/(np.max(nongrok_fourier_layer1.detach().numpy())),name='Learn - Fourier',mode='lines',line=dict(color='orange')),row=1,col=1)

    fig1.add_trace(go.Scatter(x=np.arange(0,grok_comp_layer1.shape[0]),y=grok_comp_layer1.detach().numpy()/(torch.max(grok_comp_layer1).item()),name='Grok - Comp',mode='lines',line=dict(color='red')),row=1,col=2)
    fig1.add_trace(go.Scatter(x=np.arange(0,grok_comp_layer1.shape[0]),y=grok_fourier_layer1.detach().numpy()/torch.max(grok_fourier_layer1).item(),name='Grok - Fourier',mode='lines',line=dict(color='black')),row=1,col=2)

    fig1.update_layout(title_text='Layer 1 Fourier and Computational norms - "rollout"')
    fig1.update_xaxes(title_text='Component',row=1)
    #fig2.update_xaxes(title_text='Component - Fourier',row=2)
    fig1.update_yaxes(title_text='Norm, normalized by max value')
    #fig2.update_yaxes(range=[0,1])



    fig11=make_subplots(rows=2,cols=2,subplot_titles=['Computational basis norms - Learn','Computational basis norms - Grok','Fourier basis norms - Learn','Fourier basis norms - Grok'])
    
    grok_flat_comp=torch.flatten(grok_comp_output)
    grok_flat_fourier=torch.flatten(grok_fourier_output)
    nongrok_flat_comp=torch.flatten(nongrok_comp_output)
    nongrok_flat_fourier=torch.flatten(nongrok_fourier_output)


    fig11.add_trace(go.Scatter(x=np.arange(0,len(nongrok_flat_comp.detach().numpy())),y=nongrok_flat_comp.detach().numpy()/torch.max(nongrok_flat_comp).item(),name='Learn - Comp',mode='markers',marker=dict(color='orange')),row=1,col=1)
    fig11.add_trace(go.Scatter(x=np.arange(0,len(nongrok_flat_comp.detach().numpy())),y=nongrok_flat_comp.detach().numpy()/torch.max(nongrok_flat_comp).item(),name='Learn - Fourier',mode='markers',marker=dict(color='blue')),row=1,col=2)

    fig11.add_trace(go.Scatter(x=np.arange(0,len(grok_flat_comp.detach().numpy())),y=grok_flat_fourier.detach().numpy()/torch.max(grok_flat_fourier).item(),name='Grok - Comp',mode='markers',marker=dict(color='black')),row=2,col=1)
    fig11.add_trace(go.Scatter(x=np.arange(0,len(grok_flat_comp.detach().numpy())),y=grok_flat_fourier.detach().numpy()/torch.max(grok_flat_fourier).item(),name='Grok - Fourier',mode='markers',marker=dict(color='red')),row=2,col=2)
    
    fig11.update_layout(title_text='Norms layer 1 - flattened')
    fig11.update_xaxes(title_text='Component')
    fig11.update_yaxes(title_text='Norm, normalized by max value')
    
    


    #############
    grok_comp_norms_layer2=(grok_comp_output_layer2.pow(2).sum(-1)).pow(1/2)
    nongrok_comp_norms_layer2=(nongrok_comp_output_layer2.pow(2).sum(-1)).pow(1/2)

    grok_fourier_norms_layer2=(grok_fourier_output_layer2.pow(2).sum(-1)).pow(1/2)
    nongrok_fourier_norms_layer2=(nongrok_fourier_output_layer2.pow(2).sum(-1)).pow(1/2)

    

    


    fig2=make_subplots(rows=1,cols=2,subplot_titles=['Computational basis norms - Learn','Computational basis norms - Grok','Fourier basis norms - Learn','Fourier basis norms - Grok'])
    
    # fig2.add_trace(go.Scatter(x=np.arange(0,nongrok_comp_output_layer2.shape[0]),y=nongrok_comp_output_layer2.detach().numpy()/np.max(nongrok_comp_output_layer2.detach().numpy()),name='Learn - Comp',mode='lines',line=dict(color='blue')),row=1,col=1)
    # fig2.add_trace(go.Scatter(x=np.arange(0,grok_comp_output_layer2.shape[0]),y=grok_comp_output_layer2.detach().numpy()/np.max(grok_comp_output_layer2.detach().numpy()),name='Grok - Comp',mode='lines',line=dict(color='red')),row=1,col=2)
    
    # fig2.add_trace(go.Scatter(x=np.arange(0,nongrok_fourier_output_layer2.shape[0]),y=nongrok_fourier_output_layer2.detach().numpy()/np.max(nongrok_fourier_output_layer2.detach().numpy()),name='Learn - Fourier',mode='lines',line=dict(color='blue')),row=2,col=1)
    # fig2.add_trace(go.Scatter(x=np.arange(0,grok_fourier_output_layer2.shape[0]),y=grok_fourier_output_layer2.detach().numpy()/np.max(grok_fourier_output_layer2.detach().numpy()),name='Grok - Fourier',mode='lines',line=dict(color='red')),row=2,col=2)
    
    fig2.add_trace(go.Scatter(x=np.arange(0,nongrok_comp_norms_layer2.shape[0]),y=nongrok_comp_norms_layer2.detach().numpy(),name='Learn - Comp',mode='lines',line=dict(color='orange')),row=1,col=1)
    fig2.add_trace(go.Scatter(x=np.arange(0,nongrok_fourier_norms_layer2.shape[0]),y=nongrok_fourier_norms_layer2.detach().numpy(),name='Learn - Fourier',mode='lines',line=dict(color='blue')),row=1,col=1)

    fig2.add_trace(go.Scatter(x=np.arange(0,grok_comp_norms_layer2.shape[0]),y=grok_comp_norms_layer2.detach().numpy(),name='Grok - Comp',mode='lines',line=dict(color='black')),row=1,col=2)
    fig2.add_trace(go.Scatter(x=np.arange(0,grok_fourier_norms_layer2.shape[0]),y=grok_fourier_norms_layer2.detach().numpy(),name='Grok - Fourier',mode='lines',line=dict(color='red')),row=1,col=2)

    fig2.update_layout(title_text='Layer 2 Fourier and Computational norms')
    fig2.update_xaxes(title_text='Component',row=1)
    #fig2.update_xaxes(title_text='Component - Fourier',row=2)
    fig2.update_yaxes(title_text='Norm')
    #fig2.update_yaxes(range=[0,1])



    fig2norm=make_subplots(rows=1,cols=2,subplot_titles=['Computational basis norms - Learn','Computational basis norms - Grok','Fourier basis norms - Learn','Fourier basis norms - Grok'])
    
    max_non_grok_comp=torch.max(nongrok_comp_norms_layer2).item()
    max_non_grok_fourier=torch.max(nongrok_fourier_norms_layer2).item()
    max_grok_comp=torch.max(grok_comp_norms_layer2).item()
    max_grok_fourier=torch.max(grok_fourier_norms_layer2).item()

    # fig2.add_trace(go.Scatter(x=np.arange(0,nongrok_comp_output_layer2.shape[0]),y=nongrok_comp_output_layer2.detach().numpy()/np.max(nongrok_comp_output_layer2.detach().numpy()),name='Learn - Comp',mode='lines',line=dict(color='blue')),row=1,col=1)
    # fig2.add_trace(go.Scatter(x=np.arange(0,grok_comp_output_layer2.shape[0]),y=grok_comp_output_layer2.detach().numpy()/np.max(grok_comp_output_layer2.detach().numpy()),name='Grok - Comp',mode='lines',line=dict(color='red')),row=1,col=2)
    
    # fig2.add_trace(go.Scatter(x=np.arange(0,nongrok_fourier_output_layer2.shape[0]),y=nongrok_fourier_output_layer2.detach().numpy()/np.max(nongrok_fourier_output_layer2.detach().numpy()),name='Learn - Fourier',mode='lines',line=dict(color='blue')),row=2,col=1)
    # fig2.add_trace(go.Scatter(x=np.arange(0,grok_fourier_output_layer2.shape[0]),y=grok_fourier_output_layer2.detach().numpy()/np.max(grok_fourier_output_layer2.detach().numpy()),name='Grok - Fourier',mode='lines',line=dict(color='red')),row=2,col=2)
    
    fig2norm.add_trace(go.Scatter(x=np.arange(0,nongrok_comp_norms_layer2.shape[0]),y=nongrok_comp_norms_layer2.detach().numpy()/(max_non_grok_comp),name='Learn - Comp',mode='lines',line=dict(color='blue')),row=1,col=1)
    fig2norm.add_trace(go.Scatter(x=np.arange(0,nongrok_fourier_norms_layer2.shape[0]),y=nongrok_fourier_norms_layer2.detach().numpy()/(max_non_grok_fourier),name='Learn - Fourier',mode='lines',line=dict(color='orange')),row=1,col=1)

    fig2norm.add_trace(go.Scatter(x=np.arange(0,grok_comp_norms_layer2.shape[0]),y=grok_comp_norms_layer2.detach().numpy()/(max_grok_comp),name='Grok - Comp',mode='lines',line=dict(color='red')),row=1,col=2)
    fig2norm.add_trace(go.Scatter(x=np.arange(0,grok_fourier_norms_layer2.shape[0]),y=grok_fourier_norms_layer2.detach().numpy()/(max_grok_fourier),name='Grok - Fourier',mode='lines',line=dict(color='black')),row=1,col=2)

    fig2norm.update_layout(title_text='Layer 2 Fourier and Computational norms')
    fig2norm.update_xaxes(title_text='Component',row=1)
    #fig2.update_xaxes(title_text='Component - Fourier',row=2)
    fig2norm.update_yaxes(title_text='Norm, normalized by max')
    #fig2.update_yaxes(range=[0,1])


    fig4=make_subplots(rows=2,cols=3,subplot_titles=['Computational basis PCA - Learn','Computational basis PCA - Grok','Computational basis singular values','Fourier basis PCA - Learn','Fourier basis PCA - Grok','Fourier basis singular values'])
    
    u_nongrok_comp,s_nongrok_comp,v_nongrok_comp=torch.svd(comp_output[0])
    nongrok_proj_matrix_comp=torch.matmul(comp_output[0], torch.tensor(v_nongrok_comp[:, :2]))

    u_grok_comp,s_grok_comp,v_grok_comp=torch.svd(comp_output[1])
    grok_proj_matrix_comp=torch.matmul(comp_output[0], torch.tensor(v_grok_comp[:, :2]))

    u_nongrok_fourier_cos,s_nongrok_fourier_cos,v_nongrok_fourier_cos=torch.svd(nongrok_fmo_cos)
    nongrok_proj_matrix_fourier_cos=torch.matmul(nongrok_fmo_cos, torch.tensor(v_nongrok_fourier_cos[:, :2]))

    u_grok_fourier_cos,s_grok_fourier_cos,v_grok_fourier_cos=torch.svd(grok_fmo_cos)
    grok_proj_matrix_fourier_cos=torch.matmul(grok_fmo_cos, torch.tensor(v_grok_fourier_cos[:, :2]))

    u_nongrok_fourier_sin,s_nongrok_fourier_sin,v_nongrok_fourier_sin=torch.svd(nongrok_fmo_sin)
    nongrok_proj_matrix_fourier_sin=torch.matmul(nongrok_fmo_sin, torch.tensor(v_nongrok_fourier_sin[:, :2]))

    u_grok_fourier_sin,s_grok_fourier_sin,v_grok_fourier_sin=torch.svd(grok_fmo_sin)
    grok_proj_matrix_fourier_sin=torch.matmul(grok_fmo_sin, torch.tensor(v_grok_fourier_sin[:, :2]))


    fig4.add_trace(go.Scatter(x=nongrok_proj_matrix_comp[:,0].detach().numpy(), y=nongrok_proj_matrix_comp[:,1].detach().numpy(), name='Learn', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=1, col=1)
    fig4.add_trace(go.Scatter(x=grok_proj_matrix_comp[:,0].detach().numpy(), y=grok_proj_matrix_comp[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='red'),showlegend=True), row=1, col=2)

    fig4.add_trace(go.Scatter(x=nongrok_proj_matrix_fourier_cos[:,0].detach().numpy(), y=nongrok_proj_matrix_fourier_cos[:,1].detach().numpy(), name='Learn', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=2, col=1)
    fig4.add_trace(go.Scatter(x=nongrok_proj_matrix_fourier_sin[:,0].detach().numpy(), y=nongrok_proj_matrix_fourier_sin[:,1].detach().numpy(), name='Learn', mode='markers',marker=dict(size=10,color='blue'),showlegend=False), row=2, col=1)

    fig4.add_trace(go.Scatter(x=grok_proj_matrix_fourier_cos[:,0].detach().numpy(), y=grok_proj_matrix_fourier_cos[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='red'),showlegend=True), row=2, col=2)
    fig4.add_trace(go.Scatter(x=grok_proj_matrix_fourier_sin[:,0].detach().numpy(), y=grok_proj_matrix_fourier_sin[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='red'),showlegend=True), row=2, col=2)

    fig4.add_trace(go.Scatter(x=np.arange(0,len(s_nongrok_comp)),y=s_nongrok_comp.detach().numpy(),name='Learn',mode='lines',line=dict(color='blue')),row=1,col=3)
    fig4.add_trace(go.Scatter(x=np.arange(0,len(s_grok_comp)),y=s_grok_comp.detach().numpy(),name='Grok',mode='lines',line=dict(color='red')),row=1,col=3)

    s_nongrok_fourier=torch.sort(torch.cat([s_nongrok_fourier_cos,s_nongrok_fourier_sin]),descending=True)[0]
    s_grok_fourier=torch.sort(torch.cat([s_grok_fourier_cos,s_grok_fourier_sin]),descending=True)[0]

    fig4.add_trace(go.Scatter(x=np.arange(0,len(s_nongrok_fourier)),y=s_nongrok_fourier.detach().numpy(),name='Learn',mode='lines',line=dict(color='blue')),row=2,col=3)
    fig4.add_trace(go.Scatter(x=np.arange(0,len(s_grok_fourier)),y=s_grok_fourier.detach().numpy(),name='Grok',mode='lines',line=dict(color='red')),row=2,col=3)
    
    


    fig4.update_yaxes(title_text='Singular value',col=3)
    fig4.update_xaxes(title_text='Rank',col=3)

    fig4.update_layout(title_text='PCA of norms')
    fig4.update_xaxes(title_text='PCA 1 of norms - comp',row=1)
    fig4.update_yaxes(title_text='PCA 2 of norms - comp',row=1)

    fig4.update_xaxes(title_text='PCA 1 of norms - fourier',row=2)
    fig4.update_yaxes(title_text='PCA 2 of norms - fourier',row=2)


    fig6=make_subplots(rows=2,cols=2,subplot_titles=['Computational basis PCA - Learn','Computational basis PCA - Grok','Fourier basis PCA - Learn','Fourier basis PCA - Grok'])
    
    u_nongrok_comp,s_nongrok_comp,v_nongrok_comp=torch.svd(comp_output[0])
    nongrok_proj_matrix_comp=torch.matmul(comp_output[0], torch.tensor(v_nongrok_comp[:, :2]))

    u_grok_comp,s_grok_comp,v_grok_comp=torch.svd(comp_output[1])
    grok_proj_matrix_comp=torch.matmul(comp_output[0], torch.tensor(v_grok_comp[:, :2]))

    u_nongrok_fourier,s_nongrok_fourier,v_nongrok_fourier=torch.svd(nongrok_fmo_sin)
    nongrok_proj_matrix_fourier=torch.matmul(nongrok_fmo_sin, torch.tensor(v_nongrok_fourier[:, :2]))

    u_grok_fourier,s_grok_fourier,v_grok_fourier=torch.svd(grok_fmo_sin)
    grok_proj_matrix_fourier=torch.matmul(grok_fmo_sin, torch.tensor(v_grok_fourier[:, :2]))

    fig6.add_trace(go.Scatter(x=nongrok_proj_matrix_comp[:,0].detach().numpy(), y=nongrok_proj_matrix_comp[:,1].detach().numpy(), name='Learn', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=1, col=1)
    fig6.add_trace(go.Scatter(x=grok_proj_matrix_comp[:,0].detach().numpy(), y=grok_proj_matrix_comp[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='red'),showlegend=True), row=1, col=2)

    fig6.add_trace(go.Scatter(x=nongrok_proj_matrix_fourier[:,0].detach().numpy(), y=nongrok_proj_matrix_fourier[:,1].detach().numpy(), name='Learn', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=2, col=1)
    fig6.add_trace(go.Scatter(x=grok_proj_matrix_fourier[:,0].detach().numpy(), y=grok_proj_matrix_fourier[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='red'),showlegend=True), row=2, col=2)

    fig6.update_layout(title_text='PCA of norms')
    fig6.update_xaxes(title_text='PCA 1 of norms - comp',row=1)
    fig6.update_yaxes(title_text='PCA 2 of norms - comp',row=1)

    fig6.update_xaxes(title_text='PCA 1 of norms - fourier',row=2)
    fig6.update_yaxes(title_text='PCA 2 of norms - fourier',row=2)



    fig5=make_subplots(rows=3,cols=2,subplot_titles=['Computational basis PCA - Learn','Computational basis PCA - Grok','Fourier basis PCA - Learn','Fourier basis PCA - Grok'])
    



    u_nongrok_comp,s_nongrok_comp,v_nongrok_comp=torch.svd(nongrok_comp_output_layer2)
    nongrok_proj_matrix_comp=torch.matmul(nongrok_comp_output_layer2, torch.tensor(v_nongrok_comp[:, :2]))

    u_grok_comp,s_grok_comp,v_grok_comp=torch.svd(grok_comp_output_layer2)
    grok_proj_matrix_comp=torch.matmul(grok_comp_output_layer2, torch.tensor(v_grok_comp[:, :2]))

    u_nongrok_fourier,s_nongrok_fourier,v_nongrok_fourier=torch.svd(nongrok_fourier_output_layer2)
    nongrok_proj_matrix_fourier=torch.matmul(nongrok_fourier_output_layer2, torch.tensor(v_nongrok_fourier[:, :2]))

    u_grok_fourier,s_grok_fourier,v_grok_fourier=torch.svd(grok_fourier_output_layer2)
    grok_proj_matrix_fourier=torch.matmul(grok_fourier_output_layer2, torch.tensor(v_grok_fourier[:, :2]))

    fig5.add_trace(go.Scatter(x=nongrok_proj_matrix_comp[:,0].detach().numpy(), y=nongrok_proj_matrix_comp[:,1].detach().numpy(), name='Learn', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=1, col=1)
    fig5.add_trace(go.Scatter(x=grok_proj_matrix_comp[:,0].detach().numpy(), y=grok_proj_matrix_comp[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='red'),showlegend=True), row=1, col=2)

    fig5.add_trace(go.Scatter(x=nongrok_proj_matrix_fourier[:,0].detach().numpy(), y=nongrok_proj_matrix_fourier[:,1].detach().numpy(), name='Learn', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=2, col=1)
    fig5.add_trace(go.Scatter(x=grok_proj_matrix_fourier[:,0].detach().numpy(), y=grok_proj_matrix_fourier[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=2, col=2)


    fig5.update_yaxes(title_text='Singular value',row=3)
    fig5.update_xaxes(title_text='Rank',row=3)

    fig5.update_layout(title_text='PCA of norms')
    fig5.update_xaxes(title_text='PCA 1 of norms - comp',row=1)
    fig5.update_yaxes(title_text='PCA 2 of norms - comp',row=1)

    fig5.update_xaxes(title_text='PCA 1 of norms - fourier',row=2)
    fig5.update_yaxes(title_text='PCA 2 of norms - fourier',row=2)
    



    

    
    

    
    return fig,fig1,fig11,fig2,fig2norm,fig4,fig5,fig6

fig,fig1,fig11,fig2,fig2norm,fig4,fig5,fig6=matrix_heatmap(grok_object,nongrok_object)
#fig.show()#heatmap
# fig1.show()
# fig11.show()
# fig2.show()
# fig2norm.show()
# fig4.show()
# fig5.show()
# fig6.show()

grok_object.cosine_sim(nongrok_object).show()
grok_object.traincurves_and_iprs(nongrok_object).show()

def norm_histogram_plot(grokobject,nongrokobject=None):
    objects=1
    comp_output=[]
    fourier_output=[]
    # print(f'grok trainargs:{vars(grokobject.trainargs)}')
    # print(f'nongrok trainargs:{vars(nongrokobject.trainargs)}')
    # exit()
    if nongrok_object is not None:
        objects+=1
        nongrok_comp_output,nongrok_fourier_output,a,b,c,d=comp_and_fourier_norms(nongrok_object)
        comp_output.append(nongrok_comp_output)
        fourier_output.append(nongrok_fourier_output)

    grok_comp_output,grok_fourier_output,a,b,c,d=comp_and_fourier_norms(grokobject)
    
    
    comp_output.append(grok_comp_output)
    fourier_output.append(grok_fourier_output)

    colors=['blue','red']
    names=['Learn','Grok']
    # Assuming objects, comp_output, and fourier_output are defined
    fig = make_subplots(
        rows=2, cols=objects,
        subplot_titles=['Computational basis norms - Learn ','Computational basis norms - Grok', 'Fourier basis norms - Learn', 'Fourier basis norms - Grok'][:2*objects],
        shared_yaxes=True  # Ensure y-axes are shared within each row
    )

    print(torch.equal(comp_output[0],comp_output[1]))
    

    for object_index in range(objects):
        normed_output = comp_output[object_index]
        fourier_normed_output = fourier_output[object_index] 
        
        # Add the heatmap for the first row (computational basis norms)
        fig.add_trace(go.Histogram(
            x=normed_output.detach().numpy().flatten(),
            marker=dict(color=colors[object_index]),showlegend=True,name=names[object_index]+f' computational basis'
        ), row=1, col=1 + object_index)

        # Add the heatmap for the second row (Fourier basis norms)
        fig.add_trace(go.Histogram(
            x=fourier_normed_output.detach().numpy().flatten(),
            marker=dict(color=colors[object_index]),showlegend=True,name=names[object_index]+f' fourier basis'
        ), row=2, col=1 + object_index)

    # Show the figure
    return fig
    
norm_histogram_plot(grok_object,nongrok_object).show()
exit()

def comp_and_fourier_plots(grokobject,nongrokobject=None):
    grok_comp_output,grok_fourier_output=comp_and_fourier_norms(grokobject)

    if nongrok_object is not None:
        nongrok_comp_output,nongrok_fourier_output=comp_and_fourier_norms(nongrokobject)
    

    
    return None

fig=make_subplots(rows=1,cols=1)
#fig.add_trace(go.Heatmap(z=input_matrix_outputs,x=np.arange(0,input_matrix_outputs.shape[0],y=np.arange(0,input_matrix_outputs.shape[1]))),row=1,col=1)
fig.add_trace(
    go.Heatmap(
        z=normed_output.detach().numpy(),
        x=np.arange(0, normed_output.shape[0]),
        y=np.arange(0, normed_output.shape[1])
    ),
    row=1,
    col=1
)
fig.update_layout(title_text='Untransformed number input')
fig.update_xaxes(title_text="First number ('x')")
fig.update_yaxes(title_text="Second number ('y')")

fig.show()

fig=make_subplots(rows=1,cols=1)
#fig.add_trace(go.Heatmap(z=input_matrix_outputs,x=np.arange(0,input_matrix_outputs.shape[0],y=np.arange(0,input_matrix_outputs.shape[1]))),row=1,col=1)
fig.add_trace(
    go.Heatmap(
        z=fourier_normed_output.detach().numpy(),
        x=np.arange(0, normed_output.shape[0]),
        y=np.arange(0, normed_output.shape[1])
    ),
    row=1,
    col=1
)
fig.update_layout(title_text='Fourier transformed input')
fig.update_xaxes(title_text="Fourier 'x' component")
fig.update_yaxes(title_text="Fourier 'y' component")

fig.show()


fig=make_subplots(rows=1,cols=2,subplot_titles=['Computational basis norms','Fourier basis norms'])
fig.add_trace(go.Histogram(x=normed_output.detach().numpy().flatten(),name='Untransformed'),row=1,col=1)
fig.add_trace(go.Histogram(x=fourier_normed_output.detach().numpy().flatten(),name='Fourier transformed'),row=1,col=2)
fig.update_layout(title_text='Output histograms')
fig.update_xaxes(title_text='Input component')
fig.update_yaxes(title_text='Frequency')
fig.show()









print(torch.equal(all_twop_inputs, all_twop_inputs_2))

#Ok! So the fourier transform setup is different. The correct thing is to split out the two-hot encoded vector and do a fourier transform on that.
#The quicker thing is just to `roll it out` and do a Fourier transform on a vector of length 2P.


# fig=make_subplots(rows=2,cols=1)
# fig.add_trace(go.Scatter(x=fourier_basis_names,y=fourier_input_norms.detach().numpy()),row=1,col=1)
# fig.show()

# epoch=nongrok_object.model_epochs()[-1]
# def pca_mat(object,epoch):
#     model= load_model(object,epoch)[0]
#     weights=[p for p in model.parameters() if p.dim()>1]
#     u_grok,s_grok,v_grok=torch.svd(weights[0])
#     grok_proj_matrix=torch.matmul(weights[0], torch.tensor(v_grok[:, :2]))

#     return grok_proj_matrix

# grok_proj_matrix=pca_mat(grok_object,epoch)
# nongrok_proj_matrix=pca_mat(nongrok_object,epoch)

# fig=make_subplots(rows=1,cols=2)
# fig.add_trace(go.Scatter(x=grok_proj_matrix[:,0].detach().numpy(), y=grok_proj_matrix[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='red'),showlegend=True), row=1, col=1)
# fig.add_trace(go.Scatter(x=nongrok_proj_matrix[:,0].detach().numpy(), y=nongrok_proj_matrix[:,1].detach().numpy(), name='Learn', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=1, col=2)
# fig.show()
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
