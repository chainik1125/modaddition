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
from mod_add_cluster.cluster_run_average import TrainArgs
from mod_add_cluster.cluster_run_average import CNN
from mod_add_cluster.cluster_run_average import CNN_nobias
from contextlib import contextmanager
import functools
from mod_add_cluster.cluster_run_average import ModularArithmeticDataset
from mod_add_cluster.cluster_run_average import Square, MLP
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


#cosine_plots
# nongrok_foldername="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/test_runs/hiddenlayer_[70, 35]_desc_mnist_wm_1.0"
# foldername="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/test_runs/hiddenlayer_[70, 35]_desc_mnist_wm_10.0"


# grok_object=open_files_in_leaf_directories(foldername)[0]



# nongrok_object=open_files_in_leaf_directories(nongrok_foldername)[0]
# methods = [func for func in dir(nongrok_object) if callable(getattr(nongrok_object, func)) and not func.startswith("__")]

# grok_object.cosine_sim(nongrok_object).show()


#Linear plots
nongrok_foldername="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/test_runs/hiddenlayer_[512]_desc_modadd_wm_0.1"
foldername="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/test_runs/hiddenlayer_[512]_desc_modadd_wm_15.0"


grok_object=open_files_in_leaf_directories(foldername)[0]

nongrok_object=open_files_in_leaf_directories(nongrok_foldername)[0]

print(f'trainargs grok: {vars(grok_object.trainargs)}')
print(f'trainargs nongrok: {vars(nongrok_object.trainargs)}')



#methods = [func for func in dir(nongrok_object) if callable(getattr(nongrok_object, func)) and not func.startswith("__")]

#grok_object.linear_decomposition_plot(nongrok_object).show()
# grok_object.cosine_sim(nongrok_object).show()
# grok_object.traincurves_and_iprs(nongrok_object).show()
# grok_object.weights_histogram_epochs2(nongrok_object).show()


#Let's try to do the Fourier transform

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

fourier_basis, fourier_basis_names=make_fourier_basis(nongrok_object,2*nongrok_object.trainargs.P)
# print(f'fourier_basis shape: {fourier_basis.shape}')
# fig=make_subplots(rows=1,cols=1)
# fig.add_trace(go.Heatmap(z=fourier_basis@fourier_basis.T),row=1,col=1)
# fig.show()

epoch=nongrok_object.model_epochs()[-1]
model= load_model(nongrok_object,epoch)[0]
trainloader=nongrok_object.train_loader
for batch in trainloader:
    inputs,labels=batch
    test_input=inputs[0]
    break





weights=[p for p in model.parameters() if p.dim()>1]

fourier_transformed_input_weights= weights[0]@fourier_basis.T
fourier_input_norms=fourier_transformed_input_weights.pow(2).sum(0)

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

def transform_tensors(twop_vectors):
    if twop_vectors.shape[0]==97:
        P=twop_vectors.shape[0]
        twop_vectors = twop_vectors.reshape(-1, 2*P)
        N=twop_vectors.shape[0]
        
    else:
        N, twoP = twop_vectors.shape
        P = twoP // 2
    
    

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


fourier_basis=make_fourier_basis(nongrok_object,nongrok_object.trainargs.P)[0]
fourier_basis_2=make_fourier_basis(nongrok_object,2*nongrok_object.trainargs.P)[0]

transformed_tensor=transform_tensors(inputs)
#transformed_tensors_fourier=fourier_basis@transformed_tensor@fourier_basis.T
#print(transformed_tensors_fourier.shape)

all_twop_inputs=[]
for i in range(0,nongrok_object.trainargs.P):
    for j in range(0,nongrok_object.trainargs.P):
        twop_input=torch.zeros(2*nongrok_object.trainargs.P)
        twop_input[i]=1
        twop_input[j+nongrok_object.trainargs.P]=1
        all_twop_inputs.append(twop_input)
all_twop_inputs_1=torch.stack(all_twop_inputs)


def make_all_twop(nongrok_object):
    P = nongrok_object.trainargs.P

    # Create a tensor of shape (P, P, 2P) filled with zeros
    all_twop_inputs = torch.zeros(P, P, 2*P)

    # Set the appropriate indices to 1
    all_twop_inputs[torch.arange(P), :, torch.arange(P)] = 1
    all_twop_inputs[:, torch.arange(P), torch.arange(P, 2*P)] = 1
    
    

    # Reshape the tensor to (P*P, 2P) from (PxPx2P)

    #all_twop_inputs = all_twop_inputs.reshape(-1, 2*P)

    return all_twop_inputs

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


all_twop_inputs_2=make_all_twop(nongrok_object)
print(f'all_twop_inputs_2 shape: {all_twop_inputs_2.shape}')

all_psquared_inputs=transform_tensors(all_twop_inputs_2)
print(f'all_psquared_inputs shape: {all_psquared_inputs.shape}')



fourier_all_psquared=fourier_basis.T @ all_psquared_inputs@ fourier_basis
transformed_fourier= torch.cat([fourier_all_psquared[:, 0, :], fourier_all_psquared[:, 1, :]], dim=1)
print(f'fourier shape: {fourier_all_psquared.shape}')
print(f'transformed_fourier shape: {transformed_fourier.shape}')

print(f'fourier shape: {fourier_all_psquared.shape}')
# all_fourier_inputs=transform_tensors(fourier_all_psquared)
# print(f'all_fourier_inputs shape: {all_fourier_inputs.shape}')


#print(f'fourier_all_psquared shape: {fourier_all_psquared.shape}')


input_matrix_outputs=all_twop_inputs_2@(weights[0].T)
fourier_matrix_outputs=transformed_fourier@(weights[0].T)
#At this point I need to reshape to make it a PxP matrix.
#fourier_matrix_outputs.reshape(nongrok_object.trainargs.P,nongrok_object.trainargs.P,fourier_matrix_outputs.shape[-1])
print('fourier matrix outputs shape: ',fourier_matrix_outputs.shape)
fourier_matrix_outputs=convert_output_NPP(fourier_matrix_outputs,all_psquared_inputs)


normed_output=input_matrix_outputs.pow(2).sum(-1)
fourier_normed_output=fourier_matrix_outputs.pow(2).sum(-1)
print('fourier normed output shape: ',fourier_normed_output.shape)

print(input_matrix_outputs.shape)
print(normed_output.shape)
print(fourier_normed_output.shape)

def comp_and_fourier_norms(runobject):
    epoch=runobject.model_epochs()[-1]
    model= load_model(runobject,epoch)[0]
    weights=[p for p in model.parameters() if p.dim()>1]


    all_twop_inputs=make_all_twop(runobject)
    all_psquared_inputs=transform_tensors(all_twop_inputs)
    fourier_basis=make_fourier_basis(runobject,runobject.trainargs.P)[0]

    fourier_all_psquared=fourier_basis.T @ all_psquared_inputs@ fourier_basis
    transformed_fourier= torch.cat([fourier_all_psquared[:, 0, :], fourier_all_psquared[:, 1, :]], dim=1)

    input_matrix_outputs=all_twop_inputs_2@(weights[0].T)
    fourier_matrix_outputs=transformed_fourier@(weights[0].T)
    fourier_matrix_outputs=convert_output_NPP(fourier_matrix_outputs,all_psquared_inputs)

    fourier_matrix_outputs_layer2=(weights[1].T)@fourier_basis


    normed_output=(input_matrix_outputs.pow(2).sum(-1)).pow(1/2)
    fourier_normed_output=(fourier_matrix_outputs.pow(2).sum(-1)).pow(1/2)
    fourier_normed_output_layer2=(fourier_matrix_outputs_layer2.pow(2).sum(0)).pow(1/2)
    normed_output_layer2=weights[1].pow(2).sum(-1).pow(1/2)
    

    return normed_output,fourier_normed_output,normed_output_layer2,fourier_normed_output_layer2

# def matrix_heatmap(grokobject,nongrokbject=None):
#     objects=1
#     comp_ouput=[]
#     fourier_output=[]

#     if nongrok_object is not None:
#         objects+=1
#         nongrok_comp_output,nongrok_fourier_output=comp_and_fourier_norms(nongrok_object)
#         comp_ouput.append(nongrok_comp_output)
#         fourier_output.append(nongrok_fourier_output)

#     grok_comp_output,grok_fourier_output=comp_and_fourier_norms(grokobject)
#     comp_ouput.append(grok_comp_output)
#     fourier_output.append(grok_fourier_output)

    
#     fig=make_subplots(rows=2,cols=objects,subplot_titles=['Computational basis norms','Fourier basis norms'][:objects])
#     for object_index in range(objects):
#         normed_output=comp_ouput[object_index]
#         fourier_normed_output=fourier_output[object_index] 
#         fig.add_trace(go.Heatmap(
#             z=normed_output.detach().numpy(),
#             x=np.arange(0, normed_output.shape[0]),
#             y=np.arange(0, normed_output.shape[1])),
#         row=1,col=1+object_index)

#         fig.add_trace(go.Heatmap(
#             z=normed_output.detach().numpy(),
#             x=np.arange(0, fourier_normed_output.shape[0]),
#             y=np.arange(0, fourier_normed_output.shape[1])),
#         row=2,col=1+object_index)
    
#     return fig


def matrix_heatmap(grokobject,nongrokobject=None):
    objects=1
    comp_output=[]
    fourier_output=[]

    if nongrok_object is not None:
        objects+=1
        nongrok_comp_output,nongrok_fourier_output,nongrok_comp_output_layer2,nongrok_fourier_output_layer2=comp_and_fourier_norms(nongrok_object)
        comp_output.append(nongrok_comp_output)
        fourier_output.append(nongrok_fourier_output)

    grok_comp_output,grok_fourier_output,grok_comp_output_layer2,grok_fourier_output_layer2=comp_and_fourier_norms(grokobject)
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
    nongrok_fourier_basis=make_fourier_basis(nongrokobject,2*nongrokobject.trainargs.P)[0]
    nongrok_fourier_matrix_outputs_layer1=(weights[0])@fourier_basis.T
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
    
    fig1.add_trace(go.Scatter(x=np.arange(0,nongrok_comp_layer1.shape[0]),y=nongrok_comp_layer1.detach().numpy(),name='Learn - Comp',mode='lines',line=dict(color='blue')),row=1,col=1)
    fig1.add_trace(go.Scatter(x=np.arange(0,nongrok_fourier_layer1.shape[0]),y=nongrok_fourier_layer1.detach().numpy(),name='Learn - Fourier',mode='lines',line=dict(color='orange')),row=1,col=1)

    fig1.add_trace(go.Scatter(x=np.arange(0,grok_comp_layer1.shape[0]),y=grok_comp_layer1.detach().numpy(),name='Grok - Comp',mode='lines',line=dict(color='red')),row=1,col=2)
    fig1.add_trace(go.Scatter(x=np.arange(0,grok_comp_layer1.shape[0]),y=grok_fourier_layer1.detach().numpy(),name='Grok - Fourier',mode='lines',line=dict(color='black')),row=1,col=2)

    fig1.update_layout(title_text='Layer 1 Fourier and Computational norms')
    fig1.update_xaxes(title_text='Component',row=1)
    #fig2.update_xaxes(title_text='Component - Fourier',row=2)
    fig1.update_yaxes(title_text='Norm')
    #fig2.update_yaxes(range=[0,1])




    fig2=make_subplots(rows=1,cols=2,subplot_titles=['Computational basis norms - Learn','Computational basis norms - Grok','Fourier basis norms - Learn','Fourier basis norms - Grok'])
    
    # fig2.add_trace(go.Scatter(x=np.arange(0,nongrok_comp_output_layer2.shape[0]),y=nongrok_comp_output_layer2.detach().numpy()/np.max(nongrok_comp_output_layer2.detach().numpy()),name='Learn - Comp',mode='lines',line=dict(color='blue')),row=1,col=1)
    # fig2.add_trace(go.Scatter(x=np.arange(0,grok_comp_output_layer2.shape[0]),y=grok_comp_output_layer2.detach().numpy()/np.max(grok_comp_output_layer2.detach().numpy()),name='Grok - Comp',mode='lines',line=dict(color='red')),row=1,col=2)
    
    # fig2.add_trace(go.Scatter(x=np.arange(0,nongrok_fourier_output_layer2.shape[0]),y=nongrok_fourier_output_layer2.detach().numpy()/np.max(nongrok_fourier_output_layer2.detach().numpy()),name='Learn - Fourier',mode='lines',line=dict(color='blue')),row=2,col=1)
    # fig2.add_trace(go.Scatter(x=np.arange(0,grok_fourier_output_layer2.shape[0]),y=grok_fourier_output_layer2.detach().numpy()/np.max(grok_fourier_output_layer2.detach().numpy()),name='Grok - Fourier',mode='lines',line=dict(color='red')),row=2,col=2)
    
    fig2.add_trace(go.Scatter(x=np.arange(0,nongrok_comp_output_layer2.shape[0]),y=nongrok_comp_output_layer2.detach().numpy(),name='Learn - Comp',mode='lines',line=dict(color='blue')),row=1,col=1)
    fig2.add_trace(go.Scatter(x=np.arange(0,nongrok_fourier_output_layer2.shape[0]),y=nongrok_fourier_output_layer2.detach().numpy(),name='Learn - Fourier',mode='lines',line=dict(color='orange')),row=1,col=1)

    fig2.add_trace(go.Scatter(x=np.arange(0,grok_comp_output_layer2.shape[0]),y=grok_comp_output_layer2.detach().numpy(),name='Grok - Comp',mode='lines',line=dict(color='red')),row=1,col=2)
    fig2.add_trace(go.Scatter(x=np.arange(0,grok_comp_output_layer2.shape[0]),y=grok_fourier_output_layer2.detach().numpy(),name='Grok - Fourier',mode='lines',line=dict(color='black')),row=1,col=2)

    fig2.update_layout(title_text='Layer 2 Fourier and Computational norms')
    fig2.update_xaxes(title_text='Component',row=1)
    #fig2.update_xaxes(title_text='Component - Fourier',row=2)
    fig2.update_yaxes(title_text='Norm')
    #fig2.update_yaxes(range=[0,1])


    fig4=make_subplots(rows=2,cols=2,subplot_titles=['Computational basis PCA - Learn','Computational basis PCA - Grok','Fourier basis PCA - Learn','Fourier basis PCA - Grok'])
    
    u_nongrok_comp,s_nongrok_comp,v_nongrok_comp=torch.svd(comp_output[0])
    nongrok_proj_matrix_comp=torch.matmul(comp_output[0], torch.tensor(v_nongrok_comp[:, :2]))

    u_grok_comp,s_grok_comp,v_grok_comp=torch.svd(comp_output[1])
    grok_proj_matrix_comp=torch.matmul(comp_output[0], torch.tensor(v_grok_comp[:, :2]))

    u_nongrok_fourier,s_nongrok_fourier,v_nongrok_fourier=torch.svd(fourier_output[0])
    nongrok_proj_matrix_fourier=torch.matmul(fourier_output[0], torch.tensor(v_nongrok_fourier[:, :2]))

    u_grok_fourier,s_grok_fourier,v_grok_fourier=torch.svd(fourier_output[1])
    grok_proj_matrix_fourier=torch.matmul(fourier_output[1], torch.tensor(v_grok_fourier[:, :2]))

    fig4.add_trace(go.Scatter(x=nongrok_proj_matrix_comp[:,0].detach().numpy(), y=nongrok_proj_matrix_comp[:,1].detach().numpy(), name='Learn', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=1, col=1)
    fig4.add_trace(go.Scatter(x=grok_proj_matrix_comp[:,0].detach().numpy(), y=grok_proj_matrix_comp[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='red'),showlegend=True), row=1, col=2)

    fig4.add_trace(go.Scatter(x=nongrok_proj_matrix_fourier[:,0].detach().numpy(), y=nongrok_proj_matrix_fourier[:,1].detach().numpy(), name='Learn', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=2, col=1)
    fig4.add_trace(go.Scatter(x=grok_proj_matrix_fourier[:,0].detach().numpy(), y=grok_proj_matrix_fourier[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=2, col=2)

    fig4.update_layout(title_text='PCA of norms')
    fig4.update_xaxes(title_text='PCA 1 of norms - comp',row=1)
    fig4.update_yaxes(title_text='PCA 2 of norms - comp',row=1)

    fig4.update_xaxes(title_text='PCA 1 of norms - fourier',row=2)
    fig4.update_yaxes(title_text='PCA 2 of norms - fourier',row=2)

    



    

    
    

    
    return fig,fig1,fig2,fig4

fig1,fig2,fig3,fig4=matrix_heatmap(grok_object,nongrok_object)
fig1.show()
fig2.show()
fig3.show()
fig4.show()



def norm_histogram_plot(grokobject,nongrokobject=None):
    objects=1
    comp_output=[]
    fourier_output=[]
    # print(f'grok trainargs:{vars(grokobject.trainargs)}')
    # print(f'nongrok trainargs:{vars(nongrokobject.trainargs)}')
    # exit()
    if nongrok_object is not None:
        objects+=1
        nongrok_comp_output,nongrok_fourier_output,a,b=comp_and_fourier_norms(nongrok_object)
        comp_output.append(nongrok_comp_output)
        fourier_output.append(nongrok_fourier_output)

    grok_comp_output,grok_fourier_output,a,b=comp_and_fourier_norms(grokobject)
    print(torch.equal(grok_comp_output,nongrok_comp_output))
    
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







exit()

print(torch.equal(all_twop_inputs, all_twop_inputs_2))
exit()
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
