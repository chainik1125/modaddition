
#export PYTHONPATH="${PYTHONPATH}:/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/Code"
import os
import sys
import dill

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
from cluster.cluster_run_average import TrainArgs
from cluster.cluster_run_average import CNN
from cluster.cluster_run_average import CNN_nobias
from contextlib import contextmanager
import functools
from cluster.cluster_run_average import ModularArithmeticDataset
from cluster.cluster_run_average import Square, MLP
import glob




dtype=torch.float32


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def hook_fn2(module,input,output,activation_list):
    activation_list.append(output)

def get_model_neuron_activations(epochindex,pathtodata,image_number,image_set):
    model_activations=[]
    temp_hook_dict={}

    #1. define the model:
    #1a. Get the saved_files
    onlyfiles = [f for f in os.path.listdir(pathtodata) if os.isfile(os.join(pathtodata, f)) if f!='init.pth']



    full_run_data = torch.load(pathtodata+f'/{onlyfiles[epochindex]}')

    onlyfiles=sorted(onlyfiles,key=get_epoch)


    full_run_data = torch.load(pathtodata+f'/{onlyfiles[epochindex]}')
    print(f'File name: {onlyfiles[epochindex]}')
    epoch=int(onlyfiles[epochindex].split('.')[0])


    test_model = CNN(input_dim=16, output_size=2, input_channels=1, conv_channels=conv_channels, hidden_widths=hiddenlayers,
                    activation=nn.ReLU(), optimizer=torch.optim.Adam,
                    learning_rate=0.001, weight_decay=0.08, multiplier=20.0, dropout_prob=0)

    test_model.to(device)
    test_model.load_state_dict(full_run_data['model'])
    #Define the hook
    def hook_fn(module, input, output):
        model_activations.append(output)

    count=0
    for layer in test_model.modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d) or isinstance(layer, torch.nn.modules.Linear):
            count+=1
            handle = layer.register_forward_hook(hook_fn)
            #print(type(layer).__name__)
            temp_hook_dict[count]=handle


    # 2. Forward pass the example image through the model
    with torch.no_grad():
        #print(example_image.shape)
        output = test_model(image_set)


    # Detach the hook
    for hook in temp_hook_dict.values():
        hook.remove()


    # 3. Get the activations from the list  for the specified layer


    #print([len(x) for x in model_activations])

    detached_activations=[]
    #print(model_activations[1].shape)
    for act in model_activations:
        #print(f'len(act): {len(act)}')
        if image_number==None:#averages over all images
            flat=torch.flatten(sum([act[x] for x in range(len(act))])/len(act))
        elif image_number=='dist':
            flat=act
        elif image_number=='var':
            mean=torch.flatten(sum([act[x] for x in range(len(act))])/len(act))
            x2=torch.flatten(sum([act[x]**2 for x in range(len(act))])/len(act))
            # print(len(act))
            # print(act[0].shape)
            # print(act[0])
            # print(act[0]**2)
            # print(len(mean))
            # print(len(x2))
            flat=(x2-mean**2)**(1/2)
            flat=torch.flatten(flat)
            #print(flat)
        else:
            flat=torch.flatten(act[image_number])
        #print(type(flat))
        #print(flat.shape)
        if image_number=='dist':
            detached_act=flat.detach().cpu().numpy()
        else:
            detached_act=flat.detach().cpu().numpy()
        detached_activations.append(detached_act)

    model_activations.clear() #Need this because otherwise the hook will repopulate the list!

    #detached_activations_array=np.stack(detached_activations,axis=0)#Convert to an array to keep the vectorization-->You don't want to do this because each layer has a different shape

    return detached_activations,epoch








def generate_test_set(dataset,size):
    ising=True
    if ising:
        data=dataset
        random.shuffle(data)#This randomizes the selection
        L=data[0][0].shape[0]
        # split data into input (array) and labels (phase and temp)
        inputs, phase_labels, temp_labels = zip(*data)
        # for now ignore temp labels
        my_X = torch.Tensor(np.array(inputs)).to(dtype) # transform to torch tensor of FLOATS
        my_y = torch.Tensor(np.array(phase_labels)).to(torch.long) # transform to torch tensor of INTEGERS
        my_y_temp = torch.Tensor(np.array(temp_labels)).to(dtype)
        #print(my_X.dtype, my_y.dtype)
        #print("Created Ising Dataset")

        train_size, test_size, batch_size = 100, size, 100
        a, b = train_size, test_size
        test_data = TensorDataset(my_X[:b], my_y[:b]) # test
        scramble_snapshot=False

        # load data in batches for reduced memory usage in learning
        test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False)

        for b, (X_test, y_test) in enumerate(test_loader):
            if scramble_snapshot:
                X_test_a=np.array(X_test)
                X_test_perc=np.zeros((test_size,L,L))
                for t in range(test_size):
                    preshuff=X_test_a[t,:,:].flatten()
                    np.random.shuffle(preshuff)
                    X_test_perc[t,:,:]=np.reshape(preshuff,(L,L))
                X_test=torch.Tensor(X_test_perc).to(dtype)

    return X_test.view(test_size,1,L,L),y_test



def generate_test_set_modadd(dataset,size,run_object):
    ising=False
    modadd=True
    if ising:
        data=dataset
        random.shuffle(data)#This randomizes the selection
        L=data[0][0].shape[0]
        # split data into input (array) and labels (phase and temp)
        inputs, phase_labels, temp_labels = zip(*data)
        # for now ignore temp labels
        my_X = torch.Tensor(np.array(inputs)).to(dtype) # transform to torch tensor of FLOATS
        my_y = torch.Tensor(np.array(phase_labels)).to(torch.long) # transform to torch tensor of INTEGERS
        my_y_temp = torch.Tensor(np.array(temp_labels)).to(dtype)
        #print(my_X.dtype, my_y.dtype)
        #print("Created Ising Dataset")

        train_size, test_size, batch_size = 100, size, 100
        a, b = train_size, test_size
        test_data = TensorDataset(my_X[:b], my_y[:b]) # test
        scramble_snapshot=False

        # load data in batches for reduced memory usage in learning
        test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False)

        for b, (X_test, y_test) in enumerate(test_loader):
            if scramble_snapshot:
                X_test_a=np.array(X_test)
                X_test_perc=np.zeros((test_size,L,L))
                for t in range(test_size):
                    preshuff=X_test_a[t,:,:].flatten()
                    np.random.shuffle(preshuff)
                    X_test_perc[t,:,:]=np.reshape(preshuff,(L,L))
                X_test=torch.Tensor(X_test_perc).to(dtype)
        return X_test.view(test_size,1,L,L),y_test
    if modadd:

        ma_dataset = ModularArithmeticDataset(P=run_object.trainargs.P, seed=run_object.trainargs.data_seed,loss_criterion="MSE")
        train_dataset, test_dataset = torch.utils.data.random_split(ma_dataset, [run_object.trainargs.train_fraction, 1-run_object.trainargs.train_fraction])
        train_loader = DataLoader(train_dataset, batch_size=run_object.trainargs.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=run_object.trainargs.batch_size, shuffle=True)
        return test_loader,test_dataset
    

#OK let's treat this like we would in the model. The first thing I need to do is load an object, and then extract, or generate if need be




def manual_config_CNN(run_object):
    config_dict={name:None for name,param in inspect.signature(CNN.__init__).parameters.items() if name != 'self'}
    #print(config_dict)
    first_batch = next(iter(run_object.test_loader))
    images, labels = first_batch
    input_dim=images[0].shape[0]
    print(input_dim)
    
    
    output_size=len(torch.unique(labels))
    if len(images[0].shape)==2:
        input_channels=1
    else:
        input_channels=images.shape[0]
    #print(input_channels)
    conv_channels=run_object.trainargs.conv_channels
    hidden_widths=run_object.trainargs.hiddenlayers
    activation=run_object.trainargs.activation()
    optimizer=run_object.trainargs.optimizer
    learning_rate=run_object.trainargs.lr
    weight_decay=run_object.trainargs.weight_decay
    multiplier=run_object.trainargs.weight_multiplier
    dropout_prob=run_object.trainargs.dropout_prob
    print(locals()['input_dim'])
    config_dict2={}
    for key in config_dict.keys():
        config_dict2[key]=locals()[key]
    return config_dict2





#################################################
#################################################
#################################################
#################################################




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
            if isinstance(module, (nn.Conv2d, nn.Linear,nn.MaxPool2d)):
                # Adjust name to include full path for clarity, especially useful for layers within ModuleList
                full_name = f'{name} ({module.__class__.__name__})'
                print(f'Registering hook on {full_name}')
                hook = module.register_forward_hook(save_activation(full_name))
                hooks.append(hook)
            #Have to manually add the hook for pooling layer 5
        # full_name = f'conv_layers.2 ({model.conv_layers[2].__class__.__name__})'
        # hook=model.conv_layers[2].register_forward_hook(save_activation(full_name))
        # full_name = f'conv_layers.5 ({model.conv_layers[5].__class__.__name__})'
        # hook=model.conv_layers[5].register_forward_hook(save_activation(full_name))

        #     # Explicitly check if this is the layer you are particularly interested in
        # if name == "conv_layers.5":
        #     print(f"Special hook registered on {full_name}")

    def remove_hooks():
        """Removes all hooks from the model."""
        for hook in hooks:
            hook.remove()
        print('All hooks removed.')

    register_hooks()
    # Forward pass to get outputs
    output = model(x)

    return activations, output, remove_hooks





# model=single_run.modelclass(**single_run.modelconfig)

# test_statedict=single_run.models[single_run.model_epochs()[-1]]['model']
# result=model.load_state_dict(test_statedict,strict=False)

# print("Missing keys:", result.missing_keys)
# print("Unexpected keys:", result.unexpected_keys)


# test=generate_test_set(dataset,1000)
# with torch.no_grad():
#         y_pred=model(test[0].to(device))
        
        



# activations, output, cleanup = get_activations(model,test[0])
# var_sorted_activations={key:activations[key].var(dim=0) for key in activations.keys()}
# for key in var_sorted_activations.keys():
#     print(key,var_sorted_activations[key].shape)


#single_run.make_activations_histogram2(non_grokked_object=single_run_ng,epoch=99900,fig=None,sortby='all',dataset=dataset).show()
#single_run.activations_epochs(single_run_ng,sortby='all',dataset=dataset).show()



#plotting







#Interp




#Single neuron properties

#magnetization

def magnetization(spin_grid):
    return sum(sum(spin_grid))

#1. Energy vs. activation

def compute_energy(spin_grid, J=1):
    """
    Compute the energy of a 2D Ising model.

    Parameters:
    spin_grid (2D numpy array): The lattice of spins, each element should be +1 or -1.
    J (float): Interaction energy. Defaults to 1.

    Returns:
    float: Total energy of the configuration.
    """
    energy = 0
    rows, cols = spin_grid.shape

    for i in range(rows):
        for j in range(cols):
            # Periodic boundary conditions
            right_neighbor = spin_grid[i, (j + 1) % cols]
            bottom_neighbor = spin_grid[(i + 1) % rows, j]

            # Sum over nearest neighbors
            energy += -J * spin_grid[i, j] * (right_neighbor + bottom_neighbor)

    return energy/(2*J*spin_grid.shape[0]*spin_grid.shape[1])




def compute_energy(spin_grid, J=1):
    """
    Compute the energy of a 2D Ising model.

    Parameters:
    spin_grid (2D numpy array): The lattice of spins, each element should be +1 or -1.
    J (float): Interaction energy. Defaults to 1.

    Returns:
    float: Total energy of the configuration.
    """
    energy = 0
    rows, cols = spin_grid.shape

    for i in range(rows):
        for j in range(cols):
            # Periodic boundary conditions
            right_neighbor = spin_grid[i, (j + 1) % cols]
            bottom_neighbor = spin_grid[(i + 1) % rows, j]

            # Sum over nearest neighbors
            energy += -J * spin_grid[i, j] * (right_neighbor + bottom_neighbor)

    return energy/(2*J*spin_grid.shape[0]*spin_grid.shape[1])


def compute_energy_parallel(spin_grid, J=1):
    """
    Compute the energy of a 2D Ising model in a parallelized manner using numpy.

    Parameters:
    spin_grid (2D numpy array): The lattice of spins, each element should be +1 or -1.
    J (float): Interaction energy. Defaults to 1.

    Returns:
    float: Total energy of the configuration.
    """
    rows, cols = spin_grid.shape
    # Use numpy roll to shift the array for periodic boundary conditions
    right_neighbors = np.roll(spin_grid, shift=-1, axis=1)
    bottom_neighbors = np.roll(spin_grid, shift=-1, axis=0)

    # Calculate interactions for right and bottom neighbors
    interaction_right = spin_grid * right_neighbors
    interaction_bottom = spin_grid * bottom_neighbors

    # Sum up the interactions and multiply by -J
    total_interaction = -J * (interaction_right + interaction_bottom)

    # Calculate the total energy and normalize
    energy = total_interaction.sum()
    normalized_energy = energy / (2 * J * rows * cols)

    return normalized_energy


def compute_energy_torch(spin_grid, J=1):
    """
    Compute the energy of a 2D Ising model using PyTorch for parallelization.

    Parameters:
    spin_grid (2D torch tensor): The lattice of spins, each element should be +1 or -1.
    J (float): Interaction energy. Defaults to 1.

    Returns:
    float: Total energy of the configuration.
    """
    # Ensure spin_grid is a torch tensor
    if not isinstance(spin_grid, torch.Tensor):
        spin_grid = torch.tensor(spin_grid, dtype=torch.float32)

    rows, cols = spin_grid.shape
    # Use torch.roll to shift the tensor for periodic boundary conditions
    right_neighbors = torch.roll(spin_grid, shifts=-1, dims=1)
    bottom_neighbors = torch.roll(spin_grid, shifts=-1, dims=0)

    # Calculate interactions for right and bottom neighbors
    interaction_right = spin_grid * right_neighbors
    interaction_bottom = spin_grid * bottom_neighbors

    # Sum up the interactions and multiply by -J
    total_interaction = -J * (interaction_right + interaction_bottom)

    # Calculate the total energy and normalize
    energy = total_interaction.sum()
    normalized_energy = energy / (2 * J * rows * cols)

    return normalized_energy.item()  # Convert to Python float for compatibility



#Let me try the networkx path

import networkx as nx
import itertools



#1. First need to find the neigbours of a given point

def find_neighbours(image, position):
    neighbour_positions=[]
    linear_size=image.shape[0]

    neighbour_positions=[((position[0]+i)%linear_size,(position[1]+j)%linear_size) for i,j in itertools.product([-1,0,1],repeat=2)]
    value=image[position[0],position[1]]
    matches=[neighbour_positions[i] for i in range(len(neighbour_positions)) if value==image[neighbour_positions[i][0],neighbour_positions[i][1]]]
    return matches


#2. Then find the connectivity matrix
def form_connectivity_matrix(image):
    dim=1
    for i in image.shape:
        dim=i*dim
        #print(f'dim {dim}')
    cm=np.zeros((dim,dim))
    linear_dim=image.shape[0]
    #print(f'linear dim {linear_dim}')
    for i in range(cm.shape[0]):
        #first convert i to a coordinate
        imagepos=[int(i/16),i%16]
        neighbours=find_neighbours(image=image,position=imagepos)
        for n in neighbours:
            #first you need to convert the neighbour positions into array indices
            neighbourindex=n[0]*16+n[1]
            cm[i,neighbourindex]=1
            cm[neighbourindex,i]=1
    #In principle you should go through each row and column but I think just one should be sufficient.
    for i in range(cm.shape[0]):
        cm[i,i]=0
    return cm


#3. Create a networkx graph from that matrix
def create_graph(c_matrix):
    linear_dim=c_matrix.shape[0]
    G=nx.Graph()
    G.add_nodes_from([i for i in range(linear_dim)])
    edges=np.transpose(np.nonzero(c_matrix))
    G.add_edges_from(edges)
    return G
#4. Use networkx to extract the connected component
def get_ccs(image):
    conn_matrix=form_connectivity_matrix(image)
    graph=create_graph(conn_matrix)
    ccs=max(nx.connected_components(graph), key=len)
    largest_cc = len(max(nx.connected_components(graph), key=len))
    return largest_cc


#Ok so now let's try it.

#I need to:
#1 Get a dictionary of neuron indices by whatever the sort is
#2. Form the imagesxfeatures tensor - i.e. 1000x3 (energy,mag,largest_cc)
#3. Form the  imagesxneuronsxfeatures tensor
#4. Then I can extract the imagesxfeature tensor at the value of the index called by the dictionary - from there I can easily yank out the correlation for any neuron index!


def get_acts_dict(single_run,dataset,epoch,sortby):
    model=single_run.modelclass(**single_run.modelconfig)

    test_statedict=single_run.models[epoch]['model']
    result=model.load_state_dict(test_statedict,strict=False)
    if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
        print("Missing keys:", result.missing_keys)
        print("Unexpected keys:", result.unexpected_keys)
    
    # with torch.no_grad():
    #         y_pred=model(dataset[0].to(device))

    activations_grok, output, cleanup = get_activations(model,dataset[0])#Might be good to compare grokked and c
    cleanup()
    if sortby=='var':
        sorted_activations_grok={key:activations_grok[key].var(dim=0) for key in activations_grok.keys()}
    if sortby=='mean':
        sorted_activations_grok={key:activations_grok[key].mean(dim=0) for key in activations_grok.keys()}
    if sortby=='absmean':
        sorted_activations_grok={key:activations_grok[key].abs().mean(dim=0) for key in activations_grok.keys()}
    if sortby=='all':
        sorted_activations_grok=activations_grok
    activations_rankings={key:get_act_ranks(sorted_activations_grok[key]) for key in sorted_activations_grok}
    

    return sorted_activations_grok,activations_rankings

def get_act_ranks(tensor):

    flat_tensor = tensor.flatten()

    # Get the indices of the top elements in the flattened tensor
    _, flat_indices = flat_tensor.topk(flat_tensor.numel())

    # Convert flat indices back to 2D indices
    indices = torch.stack(torch.unravel_index(flat_indices, tensor.shape)).T 

    return indices

# sorted_dic,acts_grok=get_acts_dict(single_run=single_run,dataset=test,epoch=99900,sortby='var')
# print([len(x) for x in acts_grok.values()])
# print(acts_grok.keys())

# print([list(sorted_dic.values())[0][tuple(x)] for x in indices[:10]])

def compute_energy_torch_batch(spin_grids, J=1):
    """
    Compute the energy of a batch of 2D Ising models using PyTorch for single-channel images.

    Parameters:
    spin_grids (4D torch tensor): A batch of lattice of spins, each image has a single channel. 
                                  The tensor shape should be [batch_size, channels, height, width].
                                  Typically, channels will be 1 for grayscale images.
    J (float): Interaction energy. Defaults to 1.

    Returns:
    torch.Tensor: Tensor of energies for each image in the batch.
    """
    # Ensure spin_grids is a torch tensor
    if not isinstance(spin_grids, torch.Tensor):
        spin_grids = torch.tensor(spin_grids, dtype=torch.float32)

    batch_size, channels, rows, cols = spin_grids.shape
    assert channels == 1, "This function expects single-channel images."

    # Remove channel dimension since it's 1 (squeeze or indexing can be used)
    spin_grids = spin_grids.squeeze(1)

    # Use torch.roll to shift the tensor for periodic boundary conditions
    right_neighbors = torch.roll(spin_grids, shifts=-1, dims=2)
    bottom_neighbors = torch.roll(spin_grids, shifts=-1, dims=1)

    # Calculate interactions for right and bottom neighbors
    interaction_right = spin_grids * right_neighbors
    interaction_bottom = spin_grids * bottom_neighbors

    # Sum up the interactions and multiply by -J
    total_interaction = -J * (interaction_right + interaction_bottom)

    # Calculate the total energy and normalize
    energy = total_interaction.sum(dim=(1, 2))  # Sum over each image individually
    normalized_energy = energy / (2 * J * rows * cols)

    return normalized_energy

# print(test[0].shape)

#Check on the energy function
# test_en=compute_energy_torch_batch(spin_grids=test[0],J=1)
# print(test_en.shape)
# print(test_en[:10])

# samples=3
# indices=[random.randint(0, len(test[0])) for _ in range(samples)]
# fig=make_subplots(rows=1,cols=samples,subplot_titles=[f'Tensor {indices[x]}, En: {test_en[indices[x]]}' for x in range(len(indices))])

# # print(test[0][0][0])
# # exit()

# for index in range(samples):
#     print(test[0][indices[index]])
#     fig.add_trace(go.Heatmap(z=test[0][indices[index]][0].detach().numpy()),row=1,col=index+1)
# fig.show()


#Let's try to do the connected components part

from scipy.ndimage import label

def round_to_significant_figures(tensor, sig_figs=2):
    # Avoid log of zero by replacing zero with the smallest non-zero float number
    tensor = tensor.clone()  # Avoid modifying the original tensor
    tensor[tensor == 0] = torch.finfo(tensor.dtype).tiny

    # Get the order of magnitude of each element in the tensor
    magnitudes = torch.floor(torch.log10(torch.abs(tensor)))

    # Scale elements so the sig_figs digit is before the decimal point
    scale = 10.0 ** (sig_figs - 1 - magnitudes)

    # Round to the nearest integer and unscale
    tensor = torch.round(tensor * scale) / scale

    return tensor




def sigfig2(tensor, sig_figs=2):
    # Avoid log of zero by replacing zero with the smallest non-zero float number
    tensor = tensor.clone()  # Clone to avoid modifying the original tensor
    non_zero_mask = tensor != 0
    tensor[non_zero_mask] = torch.where(
        non_zero_mask,
        torch.round(tensor[non_zero_mask] * (10.0 ** (sig_figs - 1 - torch.floor(torch.log10(torch.abs(tensor[non_zero_mask])))))) / 
        (10.0 ** (sig_figs - 1 - torch.floor(torch.log10(torch.abs(tensor[non_zero_mask]))))),
        tensor[non_zero_mask]
    )

    # Convert tensor to string to avoid trailing zeros
    result_as_strings = [f"{x:.{sig_figs}g}" for x in tensor.tolist()]

    return result_as_strings




def sigfigs3(tensor, sig_figs=2, lower_bound=0.001, upper_bound=1000):
    # Clone to avoid modifying the original tensor
    tensor = tensor.clone()
    non_zero_mask = tensor != 0

    # Apply rounding
    tensor[non_zero_mask] = torch.where(
        non_zero_mask,
        torch.round(tensor[non_zero_mask] * (10.0 ** (sig_figs - 1 - torch.floor(torch.log10(torch.abs(tensor[non_zero_mask])))))) /
        (10.0 ** (sig_figs - 1 - torch.floor(torch.log10(torch.abs(tensor[non_zero_mask]))))),
        tensor[non_zero_mask]
    )

    # Convert each element to a string based on its magnitude
    result_as_strings = []
    for x in tensor.tolist():
        if x == 0:
            result_as_strings.append(f"{0:.{sig_figs}g}")
        elif abs(x) < lower_bound or abs(x) >= upper_bound:
            # Use scientific notation
            result_as_strings.append(f"{x:.{sig_figs}e}")
        else:
            # Use normal fixed-point notation
            result_as_strings.append(f"{x:.{sig_figs}f}")

    return result_as_strings


def largest_component_sizes(tensor):
    # Initialize lists to hold the sizes of the largest components for each type
    max_sizes_ones = []
    max_sizes_neg_ones = []

    # Iterate through each image in the batch
    for i in range(tensor.size(0)):
        # Convert the PyTorch tensor slice to a NumPy array
        matrix = tensor[i, 0].numpy()

        # Use scipy.ndimage.label to find connected components for +1's
        labeled_array_ones, num_features_ones = label(matrix == 1, structure=np.ones((3, 3)))
        size_max_ones = max(np.sum(labeled_array_ones == j) for j in range(1, num_features_ones + 1)) if num_features_ones > 0 else 0
        max_sizes_ones.append(size_max_ones)

        # Use scipy.ndimage.label to find connected components for -1's
        labeled_array_neg_ones, num_features_neg_ones = label(matrix == -1, structure=np.ones((3, 3)))
        size_max_neg_ones = max(np.sum(labeled_array_neg_ones == j) for j in range(1, num_features_neg_ones + 1)) if num_features_neg_ones > 0 else 0
        max_sizes_neg_ones.append(size_max_neg_ones)
    maxsize = np.maximum(np.array(max_sizes_ones), np.array(max_sizes_neg_ones)) #Makes it more efficient as you don't have to loop over
    
    # Combine the sizes into a single tensor with shape (1000, 2)
    # Each row contains sizes of the largest connected components for +1's and -1's respectively
    #return torch.tensor(list(zip(max_sizes_ones, max_sizes_neg_ones)))
    return torch.tensor(maxsize)
    # Convert the list of max sizes to a PyTorch tensor
    return torch.tensor(max_sizes)

# test_cc=largest_component_sizes(test[0])
# print(test_cc.shape)

def test_image_func(image_tensor,func,func_name,samples,feature_tensor):
    if func!=None and feature_tensor==None:
        result_tensor=func(image_tensor)
    else:
        result_tensor=feature_tensor
    indices=[random.randint(0, len(test[0])) for _ in range(samples)]
    fig=make_subplots(rows=1,cols=samples,subplot_titles=[f'Tensor {indices[x]}, {", ".join([f"{name}:{value}" for name, value in zip(func_name, sigfigs3(result_tensor[indices[x]],sig_figs=2))])}' for x in range(len(indices))])#{func_name} {", ".join(sigfigs3(result_tensor[indices[x]],sig_figs=2))}



    for index in range(samples):
        print(test[0][indices[index]])
        fig.add_trace(go.Heatmap(z=test[0][indices[index]][0].detach().numpy()),row=1,col=index+1)
    fig.show()



#OK great, everything seems to work so now we can construct the tensor 
def construct_features_tensor(images_tensor,feature_funcs):
    first=True
    for func in feature_funcs:
        if first:
            features_tensor=func(images_tensor)
            features_tensor=features_tensor.unsqueeze(1)
            first=False
        else:
            features_tensor=torch.cat((features_tensor,func(images_tensor).unsqueeze(1)),dim=1)
        
    return features_tensor

def magnetization(tensor):
    return tensor.sum(dim=tuple(range(1, tensor.dim())))





#test_features=construct_features_tensor(test[0],[functools.partial(compute_energy_torch_batch,J=1),largest_component_sizes,magnetization])
#print(f'features tensor shape {test_features.shape}')
#test_image_func(test[0],func=None,func_name=['En','CC','Mag'],samples=3,feature_tensor=test_features)

#OK great, everything seems to work. Now let's try to write a function to do the correlation plots
# neuron_index=0
# activations, output, cleanup = get_activations(model,test[0])
# sorted_dic,acts_grok=get_acts_dict(single_run=single_run,dataset=test,epoch=99900,sortby='var')#second output is the indices tensor
# print(sorted_dic['conv_layers.0 (Conv2d)'].shape)
# print(acts_grok['conv_layers.0 (Conv2d)'].shape)
# neuron_index=acts_grok['conv_layers.0 (Conv2d)'][neuron_index]
# #Now need to index the tensor
# print(tuple(neuron_index.tolist()))
# print(activations['conv_layers.0 (Conv2d)'].shape)


# print(activations['conv_layers.0 (Conv2d)'][(slice(None),)+tuple(neuron_index.tolist())].shape)








def correlation_plots(features_tensor,feature_names,sorted_activations_tensor,activations_tensor,neuron_index,layer_name):
    feature_dim=features_tensor.shape[1]
    fig=make_subplots(rows=1,cols=feature_dim,subplot_titles=feature_names)
    features_act_pairs=[]
    activations_index=sorted_activations_tensor[layer_name][neuron_index]
    
    
    activations_data=activations_tensor[layer_name][(slice(None),)+tuple(activations_index.tolist())]
    for feature in range(feature_dim):
        fig.add_trace(go.Scatter(x=features_tensor[:,feature].detach().numpy(),y=activations_data.detach().numpy(),mode='markers'),row=1,col=1+feature)
    return fig


#features_ten=construct_features_tensor(test[0],[functools.partial(compute_energy_torch_batch,J=1),largest_component_sizes,magnetization])
#correlation_plots(features_tensor=features_ten,feature_names=['Energy','Connected Component','Magnetization'],sorted_activations_tensor=acts_grok,activations_tensor=activations,neuron_index=0,layer_name='conv_layers.3 (Conv2d)').show()

def correlation_one_epoch(single_run,single_run_ng,sortby,epoch,neuron_index,images_tensor,feature_funcs,fig):
    model_grok=single_run.modelclass(**single_run.modelconfig)
    grok_state_dic=single_run.models[epoch]['model']
    result=model_grok.load_state_dict(grok_state_dic,strict=False)
    if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
        print(result.missing_keys,result.unexpected_keys)

    features_tensor=construct_features_tensor(images_tensor=images_tensor,feature_funcs=feature_funcs)
    activations_grok, output_grok, cleanup_grok = get_activations(model_grok,images_tensor)
    sorted_activations_grok,acts_indices_grok=get_acts_dict(single_run=single_run,dataset=test,epoch=epoch,sortby=sortby)

    #removing 
    sorted_activations_grok.pop('conv_layers.2 (MaxPool2d)')
    if 'conv_layers.5 (MaxPool2d)' in sorted_activations_grok:
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    #sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    sorted_activations_grok.pop('fc_layers.3 (Linear)')
    activations_grok.pop('conv_layers.2 (MaxPool2d)')
    if 'conv_layers.5 (MaxPool2d)' in sorted_activations_grok:
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    #activations_grok.pop('conv_layers.5 (MaxPool2d)')
    activations_grok.pop('fc_layers.3 (Linear)')
    acts_indices_grok.pop('conv_layers.2 (MaxPool2d)')
    if 'conv_layers.5 (MaxPool2d)' in sorted_activations_grok:
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    #acts_indices_grok.pop('conv_layers.5 (MaxPool2d)')
    acts_indices_grok.pop('fc_layers.3 (Linear)')


    model_nogrok=single_run.modelclass(**single_run_ng.modelconfig)
    nogrok_state_dic=single_run_ng.models[epoch]['model']
    result=model_nogrok.load_state_dict(nogrok_state_dic,strict=False)
    if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
        print(result.missing_keys,result.unexpected_keys)

    activations_nogrok, output_nogrok, cleanup_nogrok = get_activations(model_nogrok,images_tensor)
    sorted_activations_nogrok,acts_indices_nogrok=get_acts_dict(single_run=single_run_ng,dataset=test,epoch=epoch,sortby=sortby)

    #removing 
    sorted_activations_nogrok.pop('conv_layers.2 (MaxPool2d)')
    if 'conv_layers.5 (MaxPool2d)' in sorted_activations_grok:
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    #sorted_activations_nogrok.pop('conv_layers.5 (MaxPool2d)')
    sorted_activations_nogrok.pop('fc_layers.3 (Linear)')
    activations_nogrok.pop('conv_layers.2 (MaxPool2d)')
    if 'conv_layers.5 (MaxPool2d)' in sorted_activations_grok:
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    #activations_nogrok.pop('conv_layers.5 (MaxPool2d)')
    activations_nogrok.pop('fc_layers.3 (Linear)')
    acts_indices_nogrok.pop('conv_layers.2 (MaxPool2d)')
    if 'conv_layers.5 (MaxPool2d)' in sorted_activations_grok:
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    #acts_indices_nogrok.pop('conv_layers.5 (MaxPool2d)')
    acts_indices_nogrok.pop('fc_layers.3 (Linear)')

    #Now you need to populate the layers
    feature_dim=features_tensor.shape[1]
    if fig==None:
        fig=make_subplots(rows=max(4,2*feature_dim),cols=1+len(sorted_activations_grok.keys()),subplot_titles=['Grok loss']+[f'ENERGY, layer{key}' for key in sorted_activations_grok.keys()]+['No grok loss']+[f'ENERGY, layer {key}' for key in sorted_activations_grok.keys()]+['Grok accuracy']+[f'MAG, Layer {key}' for key in sorted_activations_grok.keys()]+['No grok accuracy']+[f'MAG, Layer {key}' for key in sorted_activations_grok.keys()])#(len(grok_weights)+4)//2

    #titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_nogrok))]

    fig.add_trace(go.Scatter(x=list(range(len(single_run.train_losses))),y=single_run.train_losses,marker=dict(color='black'),showlegend=True,name='Train'),row=1,col=1)
    fig.add_trace(go.Scatter(x=list(range(len(single_run.test_losses))),y=single_run.test_losses,marker=dict(color='orange'),showlegend=True,name='Test'),row=1,col=1)
    fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(single_run.train_losses), max(single_run.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

    fig.add_trace(go.Scatter(x=list(range(len(single_run_ng.train_losses))),y=single_run_ng.train_losses,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=2,col=1)
    fig.add_trace(go.Scatter(x=list(range(len(single_run_ng.test_losses))),y=single_run_ng.test_losses,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=2,col=1)
    fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(single_run_ng.train_losses), max(single_run_ng.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)



    fig.add_trace(go.Scatter(x=list(range(len(single_run.test_accuracies))),y=single_run.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=3,col=1)
    fig.add_trace(go.Scatter(x=list(range(len(single_run.train_accuracies))),y=single_run.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=3,col=1)
    fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(single_run.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=3, col=1)

    fig.add_trace(go.Scatter(x=list(range(len(single_run_ng.test_accuracies))),y=single_run_ng.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=4,col=1)
    fig.add_trace(go.Scatter(x=list(range(len(single_run_ng.train_accuracies))),y=single_run_ng.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=4,col=1)
    fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(single_run_ng.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=4, col=1)

    if epoch==single_run.model_epochs()[0]:
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", type='log',row=1, col=2)

        fig.update_xaxes(title_text="Epoch", row=1, col=3)
        fig.update_yaxes(title_text="Accuracy", row=1, col=3)
        fig.update_xaxes(title_text="Epoch", row=1, col=4)
        fig.update_yaxes(title_text="Accuracy", row=1, col=4)
    count=0
    
    showleg=True
    for key in sorted_activations_grok.keys():
        activations_index=acts_indices_grok[key][neuron_index]
        activations_for_neuron=activations_grok[key][(slice(None),)+tuple(activations_index.tolist())]

        activations_index_nogrok=acts_indices_nogrok[key][neuron_index]
        activations_for_neuron_nogrok=activations_nogrok[key][(slice(None),)+tuple(activations_index_nogrok.tolist())]
        for feature in range(feature_dim):
            fig.add_trace(go.Scatter(x=features_tensor[:,feature],y=activations_for_neuron,mode='markers',marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1+2*feature,col=count+2)
            fig.add_trace(go.Scatter(x=features_tensor[:,feature],y=activations_for_neuron_nogrok,mode='markers',marker=dict(color='blue'),showlegend=showleg,name='No grok'),row=1+2*feature+1,col=count+2)
            #fig.add_trace(go.Scatter(x=sorted_gw,y=ccdf_gw,marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1,col=count+3)
            #fig.add_trace(go.Scatter(x=sorted_ngw,y=ccdf_ngw,marker=dict(color='blue'),showlegend=showleg,name='No Grok'),row=1,col=count+3)
            showleg=False
            #fig.update_yaxes(type='log', row=1, col=count+3)
            #fig.update_yaxes(type='log', row=2, col=count+3)
            #fig.update_xaxes(type='log', row=1, col=count+3)
            #fig.update_xaxes(type='log', row=2, col=count+3)
            # fig.add_trace(go.Histogram(x=[epoch], name=f'Grok_Epoch_{epoch}'),row=1,col=i+2)  # Placeholder histogram
            # fig.add_trace(go.Histogram(x=[epoch + 0.5], name=f'NoGrok_Epoch_{epoch}'),row=2,col=i+2)  # Placeholder histogram
        if epoch==single_run.model_epochs()[0]:
            fig.update_xaxes(title_text="Weight", row=1, col=count+1)
            fig.update_xaxes(title_text="Weight", row=2, col=count+1)
        count+=1
    return fig

features_list=[functools.partial(compute_energy_torch_batch,J=1),magnetization]#largest_component_sizes
#correlation_one_epoch(single_run=single_run,single_run_ng=single_run_ng,sortby='var',epoch=99900,neuron_index=0,images_tensor=test[0],feature_funcs=features_list,fig=None).show()
#single_run.correlation_one_epoch(single_run_ng=single_run_ng,sortby='var',epoch=99900,neuron_index=0,images_tensor=test[0],feature_funcs=features_list,fig=None,dataset=test).show()
#single_run.correlation_epochs(non_grokked_object=single_run_ng,sortby='var',epoch=99900,neuron_index=0,images_tensor=test[0],feature_funcs=features_list,dataset=test).show()


#Let's see if we can do some pruning. Let's just try to do PCA-based last layer pruning.
#1. Load the model
#2. access last layer weights
#3. do svd
#4. Test the accuracy
def set_lowest_k_to_zero(arr, k):
    """
    Sets the lowest k elements of the given array to zero.

    Parameters:
        arr (np.ndarray): The input array of any shape.
        k (int): The number of lowest elements to set to zero.
    
    Returns:
        np.ndarray: The modified array with the lowest k elements set to zero.
    """
    # Flatten the array to make it easier to find the lowest k elements
    flattened_array = arr.flatten()
    
    # Get the indices of the lowest k elements
    indices_of_lowest_k = np.argsort(flattened_array)[:k]
    
    # Set these elements to zero
    flattened_array[indices_of_lowest_k] = 0
    
    # Reshape the array back to its original shape and return
    return flattened_array.reshape(arr.shape)

# epoch=99900
# cut=1
# model=single_run.modelclass(**single_run.modelconfig)
# test_statedict=single_run.models[epoch]['model']
# result=model.load_state_dict(test_statedict,strict=False)
# print(test_statedict.keys())

# if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
#     print("Missing keys:", result.missing_keys)
#     print("Unexpected keys:", result.unexpected_keys)
# #Need something that will give back model accuracy and loss
# #layer_weights={key:test_statedict[key] for key in test_statedict.keys() if 'weight' in key}
# last_layer=test_statedict['conv_layers.3.weight']#'conv_layers.3.weight'
# print(last_layer.shape)
# u,s,v=np.linalg.svd(last_layer.detach().numpy(),full_matrices=False)
# rec=np.matmul(u * s[..., None, :], v)


# #u,s,v=np.linalg.svd(last_layer,full_matrices=False)
# print(np.allclose(last_layer.detach().numpy(), rec))

# # cut=8

# # #Let's implement the cut
# # #_, indices = torch.topk(s, k=cut, largest=True)
# # s_cut=set_lowest_k_to_zero(s,k=cut)
# # print(f'Cut {cut}' )
# # print(s_cut)
# # exit()
# cut_state_dict=test_statedict
# cut_state_dict['conv_layers.3.weight']=torch.tensor(np.matmul(u * s[..., None, :], v))
# cut_model=single_run.modelclass(**single_run.modelconfig)
# result=cut_model.load_state_dict(cut_state_dict,strict=False)
# if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
#     print("Missing keys:", result.missing_keys)
#     print("Unexpected keys:", result.unexpected_keys)
# #Nice! Now I just need to do the loss and accuracy test!

# #print(acc_loss_test(cut_model))

def acc_loss_test(model):
    X_test,y_test=generate_test_set(dataset=dataset,size=1000)
    with torch.no_grad():
        # Apply the model
        y_val = model(X_test.to(device))
        # Tally the number of correct predictions
        predicted = torch.max(y_val.data, 1)[1]
        test_correct=0
        test_correct += (predicted == y_test.to(device)).sum().item()
        test_loss = criterion(y_val, y_test.to(device))
        test_accuracy=test_correct/len(X_test)
    return test_accuracy,test_loss






def pca_pruning_last_layer(epoch,grokked_object,non_grokked_object,layer_name):
    grokked_accuracy=[]
    grokked_loss=[]
    sharezerogrokked=[]
    non_grokked_accuracy=[]
    non_grokked_loss=[]
    number_pruned=[]
    sharezerogrokked=[]
    sharezeronongrokked=[]
    def get_svd(run_object):
        try:
    # Attempt to clear the dictionary if it exists
            del model_run
            model_run=run_object.modelclass(**run_object.modelconfig)
        except NameError:
    # If run_state_dic is not defined, you can optionally define it here, or pass
            model_run=run_object.modelclass(**run_object.modelconfig)
    #     try:
    # # Attempt to clear the dictionary if it exists
    #         del run_state_dic
    #         run_state_dic=copy.deepcopy(run_object.models[epoch]['model'])
    #     except NameError:
    #         run_state_dic=copy.deepcopy(run_object.models[epoch]['model'])
        if 'run_state_dic' in locals():
            del run_state_dic  # Ensure old dictionary is removed if exists
            run_state_dic = copy.deepcopy(run_object.models[epoch]['model'])
        else:
            run_state_dic = copy.deepcopy(run_object.models[epoch]['model'])

        result=model_run.load_state_dict(run_state_dic,strict=False)
        
        if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
            print("Missing keys:", result.missing_keys)
            print("Unexpected keys:", result.unexpected_keys)

        
        run_last_layer=run_state_dic[layer_name]
        u,s,v=np.linalg.svd(run_last_layer.detach().numpy(), full_matrices=False)
        #check
        rec=np.matmul(u * s[..., None, :], v)
        #u,s,v=np.linalg.svd(last_layer,full_matrices=False)
        print(f'Reconstruction correct? {np.allclose(run_last_layer.detach().numpy(), rec)}')
        return model_run,run_last_layer,u,s,v
    model_grokked,layer_g,u_g,s_g,v_g=get_svd(grokked_object)
    model_nongrokked,layer_ng,u_ng,s_ng,v_ng=get_svd(non_grokked_object)
    print(f'model non-grokked {acc_loss_test(model_nongrokked)}')
    print(f'model grokked {acc_loss_test(model_grokked)}')
    
    def pruned_accloss(model,layer_weight,u,s,v,svs_cut,state_dict,layer_name):        
        # _, indices = torch.topk(s, k=svs_cut, largest=False)
        # s[indices]=0
        s=set_lowest_k_to_zero(arr=s,k=svs_cut)
        new_layer_matrix=np.matmul(u * s[..., None, :], v)
        state_dict[layer_name]=torch.tensor(new_layer_matrix)
        result=model.load_state_dict(state_dict,strict=False)
        if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
            print("Missing keys:", result.missing_keys)
            print("Unexpected keys:", result.unexpected_keys)

        
        acc,loss=acc_loss_test(model)
        
        nonzero=np.mean(np.isclose(new_layer_matrix, 0))#/np.sum(np.isclose(torch.zeros(layer_weight.shape),layer_weight))
        
        

        return loss,acc,nonzero

    if 'conv' in layer_name:
        rangeind=layer_g.numel() + 1
    elif 'fc_layer' in layer_name:
        rangeind=len(s_g)+1
    for sv in tqdm(range(rangeind)):
        loss_g,acc_g,nonzero_g=pruned_accloss(model=model_grokked,layer_name=layer_name,
        layer_weight=layer_g,u=u_g,s=s_g,v=v_g,svs_cut=sv,state_dict=copy.deepcopy(grokked_object.models[epoch]['model']))
        
        loss_ng,acc_ng,nonzero_ng=pruned_accloss(model=model_nongrokked,layer_name=layer_name,
        layer_weight=layer_ng,u=u_ng,s=s_ng,v=v_ng,svs_cut=sv,state_dict=copy.deepcopy(non_grokked_object.models[epoch]['model']))
        
        grokked_accuracy.append(acc_g)
        grokked_loss.append(loss_g.item())
        sharezerogrokked.append(nonzero_g)
        number_pruned.append(sv)

        non_grokked_accuracy.append(acc_ng)
        non_grokked_loss.append(loss_ng.item())
        sharezeronongrokked.append(nonzero_ng)

    return grokked_accuracy, grokked_loss, sharezerogrokked, non_grokked_accuracy, non_grokked_loss, sharezeronongrokked, number_pruned








#OK let's combine the pruning functions into one function

def pca_pruning(grokked_run,non_grokked_run,layer_name,epoch,fig):
    acc_g,loss_g,zero_g,acc_ng,loss_ng,zero_ng,number_pruned=pca_pruning_last_layer(epoch=epoch,grokked_object=grokked_run,non_grokked_object=non_grokked_run,layer_name=layer_name)
    if fig==None:
        fig=make_subplots(rows=2,cols=2)
    fig.add_trace(go.Scatter(x=number_pruned,y=acc_g,marker=dict(color='red'),showlegend=True,name='Grok'),row=1,col=1)
    fig.add_trace(go.Scatter(x=number_pruned,y=acc_ng,marker=dict(color='blue'),showlegend=True,name='No grok'),row=1,col=1)
    fig.update_xaxes(title_text="SVD's pruned", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    fig.add_trace(go.Scatter(x=number_pruned,y=loss_g,marker=dict(color='red'),showlegend=False,name='Grok'),row=1,col=2)
    fig.add_trace(go.Scatter(x=number_pruned,y=loss_ng,marker=dict(color='blue'),showlegend=False,name='No grok'),row=1,col=2)
    fig.update_xaxes(title_text="SVD's pruned", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=2)
    fig.add_trace(go.Scatter(x=zero_g,y=acc_g,marker=dict(color='red'),showlegend=True,name='Grok'),row=2,col=1)
    fig.add_trace(go.Scatter(x=zero_ng,y=acc_ng,marker=dict(color='blue'),showlegend=True,name='No grok'),row=2,col=1)
    fig.update_xaxes(title_text="Share of zero weights", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    fig.add_trace(go.Scatter(x=number_pruned,y=zero_g,marker=dict(color='red'),showlegend=True,name='Grok'),row=2,col=2)
    fig.add_trace(go.Scatter(x=number_pruned,y=zero_ng,marker=dict(color='blue'),showlegend=True,name='No grok'),row=2,col=2)
    fig.update_xaxes(title_text="SVD's pruned", row=2, col=2)
    fig.update_yaxes(title_text="Share of zero weights", row=2, col=2)
    
    return fig


#Let's try to do the magnitude based pruning!
# model=single_run.modelclass(**single_run.modelconfig)
# model_dic=copy.deepcopy(single_run.models[epoch]['model'])
# result=model.load_state_dict(model_dic,strict=False)
# if len(result.missing_keys)>0 or len(result.unexpected_keys)>0:
#     print("Missing keys:", result.missing_keys)
#     print("Unexpected keys:", result.unexpected_keys)
# for name, module in model.named_modules():
#     # Check specifically for the layer types or names you are interested in
#     if isinstance(module, (nn.Conv2d, nn.Linear)):
#         # Adjust name to include full path for clarity, especially useful for layers within ModuleList
#         full_name = f'{name} ({module.__class__.__name__})'
#         if name != 'fc_layers.3':
#             prune.l1_unstructured(module, name='weight', amount=0.9)

import torch.nn.utils.prune as prune
def prune_share(runobj, layer_name, pruning_percent):
    model,model_dic=load_model(runobj)
    # Access the specific layer by name
    module = dict(model.named_modules())[layer_name]
    
    # Apply pruning to the specific layer
    prune.l1_unstructured(module, name='weight', amount=pruning_percent)
    
    # Optionally, make the pruning permanent
    prune.remove(module, 'weight')
    
    return model


def save_original_weights(model, layers_to_prune):
    """
    Saves the original weights of the specified layers in the model.
    
    Parameters:
    - model: a PyTorch model
    - layers_to_prune: list of str, names of the layers to save weights from
    
    Returns:
    - original_weights: dict, keys are layer names and values are weights
    """
    original_weights = {}
    for name, module in model.named_modules():
        if name in layers_to_prune:
            original_weights[name] = copy.deepcopy(module.weight.data)
    return original_weights

def reset_to_original_weights(model, original_weights):
    """
    Resets the model's weights to their original state.
    
    Parameters:
    - model: a PyTorch model
    - original_weights: dict, original weights to reset to
    """
    for name, module in model.named_modules():
        if name in original_weights:
            module.weight.data = original_weights[name].clone()
def iterative_prune_from_original(model, original_weights, layers_to_prune, pruning_percents,pruning_type):
    """
    Iteratively prunes the model from the original weights and evaluates accuracy after each level of pruning.
    
    Parameters:
    - model: a PyTorch model
    - original_weights: dict, original weights of the model
    - layers_to_prune: list of layer names to prune
    - dataset: dataset to evaluate the model on
    - pruning_percents: list of floats, percentages to prune
    
    Returns:
    - accuracies: list of tuples (float, float), pruning percentage and accuracy
    """
    accuracies = []
    losses=[]
    parameters_to_prune=[]
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module,nn.Conv2d):
            print(module)
            # if isinstance(module,nn.Linear):
            #     if module.in_features==100 and module.out_features==2:
            #         continue
            parameters_to_prune.append((module,'weight'))
    
    #parameters_to_prune=[('conv_layers.0.weight','weight'),('conv_layers.3.weight','weight'),('fc_layers.0.weight','weight')]
    for percent in tqdm(pruning_percents,position=1,leave=False,disable=True):
        reset_to_original_weights(model, original_weights)
        if pruning_type=='local':
            for layer_name in layers_to_prune:
                module = dict(model.named_modules())[layer_name]
                prune.l1_unstructured(module, name='weight', amount=percent)
                prune.remove(module, 'weight')
        elif pruning_type=='global':
                prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=percent,)
        elif pruning_type=='all_layers':
            for layer_name in layers_to_prune:
                module = dict(model.named_modules())[layer_name]
                prune.l1_unstructured(module, name='weight', amount=percent)
                prune.remove(module, 'weight')

        accuracy,loss = acc_loss_test(model)  # Assuming acc_los is defined elsewhere
        accuracies.append(accuracy)
        losses.append(loss.item())
    reset_to_original_weights(model,original_weights)
    return accuracies,losses


def iterative_prune_from_original_mod(model, original_weights, layers_to_prune, pruning_percents,pruning_type):
    """
    Iteratively prunes the model from the original weights and evaluates accuracy after each level of pruning.
    
    Parameters:
    - model: a PyTorch model
    - original_weights: dict, original weights of the model
    - layers_to_prune: list of layer names to prune
    - dataset: dataset to evaluate the model on
    - pruning_percents: list of floats, percentages to prune
    
    Returns:
    - accuracies: list of tuples (float, float), pruning percentage and accuracy
    """
    accuracies = []
    losses=[]
    parameters_to_prune=[]
    # for name,module in model.named_modules():
    #     print(name)
    #     print(module)
    # exit()
    for _, module in model.named_modules():
        if isinstance(module, nn.Linear) or isinstance(module,nn.Conv2d):
            # if isinstance(module,nn.Linear):
            #     if module.in_features==100 and module.out_features==2:
            #         continue
            parameters_to_prune.append((module,'weight'))
    #parameters_to_prune=[('conv_layers.0.weight','weight'),('conv_layers.3.weight','weight'),('fc_layers.0.weight','weight')]
    for percent in tqdm(pruning_percents,position=1,leave=False,disable=False):
        reset_to_original_weights(model, original_weights)
        if pruning_type=='local':
            for layer_name in layers_to_prune:
                module = dict(model.named_modules())[layer_name]
                prune.l1_unstructured(module, name='weight', amount=percent)
                prune.remove(module, 'weight')
        elif pruning_type=='global':
                prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=percent,)
        elif pruning_type=='all_layers':
            for layer_name in layers_to_prune:
                module = dict(model.named_modules())[layer_name]
                prune.l1_unstructured(module, name='weight', amount=percent)
                prune.remove(module, 'weight')

        accuracy,loss = acc_loss_test_mod(model)  # Assuming acc_los is defined elsewhere
        accuracies.append(accuracy)
        losses.append(loss)
    reset_to_original_weights(model,original_weights)
    return accuracies,losses




def load_model(runobject,epoch):
    learning_rate=runobject.trainargs.lr
    weight_decay=runobject.trainargs.weight_decay
    model=runobject.modelclass(**runobject.modelconfig)
    model=runobject.modelinstance
    model_dic=copy.deepcopy(runobject.models[epoch]['model'])
    result=model.load_state_dict(model_dic,strict=False)
    if len(result.missing_keys)>0 or len(result.unexpected_keys)>0:
        print("Missing keys:", result.missing_keys)
        print("Unexpected keys:", result.unexpected_keys)
    return model, model_dic

def magnitude_prune(grokked_object,non_grokked_object,pruning_percents,layers_pruned,fig,epoch):
    original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
    original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)
    original_weights_g = save_original_weights(original_model_g, layers_pruned)
    original_weights_ng = save_original_weights(original_model_ng, layers_pruned)

    accs_g,losses_g=iterative_prune_from_original(original_model_g, original_weights_g, layers_pruned, pruning_percents,pruning_type='local')
    accs_ng,losses_ng=iterative_prune_from_original(original_model_ng, original_weights_ng, layers_pruned, pruning_percents,pruning_type='local')

    #Let's put the PCA in there for comparison
    acc_g,loss_g,zero_g,acc_ng,loss_ng,zero_ng,number_pruned=pca_pruning_last_layer(epoch=epoch,grokked_object=grokked_object,non_grokked_object=non_grokked_object,layer_name=layers_pruned[0]+'.weight')


    if fig==None:
        fig=make_subplots(rows=2,cols=2,subplot_titles=['Grok - Magnitude','No grok - magnitude','Grok - PCA','No grok - PCA'])
    fig.add_trace(go.Scatter(x=pruning_percents,y=accs_g,marker=dict(color='red'),name='Grok',showlegend=True),row=1,col=1)
    fig.add_trace(go.Scatter(x=pruning_percents,y=accs_ng,marker=dict(color='blue'),name='Grok',showlegend=True),row=1,col=1)
    fig.add_trace(go.Scatter(x=pruning_percents,y=losses_g,marker=dict(color='red'),name='Grok',showlegend=False),row=1,col=2)
    fig.add_trace(go.Scatter(x=pruning_percents,y=losses_ng,marker=dict(color='blue'),name='Grok',showlegend=False),row=1,col=2)
    fig.update_xaxes(title_text="Percent pruned", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_xaxes(title_text="Percent pruned", row=1, col=2)
    fig.update_yaxes(title_text="Loss",type='log', row=1, col=2)

    fig.add_trace(go.Scatter(x=zero_g,y=acc_g,marker=dict(color='red'),name='Grok',showlegend=False),row=2,col=1)
    fig.add_trace(go.Scatter(x=zero_ng,y=acc_ng,marker=dict(color='blue'),name='Grok',showlegend=False),row=2,col=1)
    fig.add_trace(go.Scatter(x=zero_g,y=loss_g,marker=dict(color='red'),name='Grok',showlegend=False),row=2,col=2)
    fig.add_trace(go.Scatter(x=zero_ng,y=loss_ng,marker=dict(color='blue'),name='Grok',showlegend=False),row=2,col=2)

    fig.update_xaxes(title_text="Percent pruned", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    fig.update_xaxes(title_text="Percent pruned", row=2, col=2)
    fig.update_yaxes(title_text="Loss",type='log', row=2, col=2)



    return fig

def magnitude_prune_prod(grokked_object,non_grokked_object,pruning_percents,layers_pruned,fig,epoch):
    original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
    original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)
    original_weights_g = save_original_weights(original_model_g, layers_pruned)
    original_weights_ng = save_original_weights(original_model_ng, layers_pruned)


    
    #accs_losses_g_global=iterative_prune_from_original(original_model_g, original_weights_g, layers_pruned, pruning_percents,'global')
    #accs_losses_ng_global=iterative_prune_from_original(original_model_ng, original_weights_ng, layers_pruned, pruning_percents,'global')

    original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
    original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)

    accs_losses_g_al=iterative_prune_from_original(original_model_g, original_weights_g, layers_pruned, pruning_percents,'all_layers')
    accs_losses_ng_al=iterative_prune_from_original(original_model_ng, original_weights_ng, layers_pruned, pruning_percents,'all_layers')

    original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
    original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)

    accs_losses_g=[iterative_prune_from_original(original_model_g, original_weights_g, [i], pruning_percents,'local') for i in layers_pruned]
    accs_losses_ng=[iterative_prune_from_original(original_model_ng, original_weights_ng, [i], pruning_percents,'local') for i in layers_pruned]

    
    #Let's put the PCA in there for comparison


    if fig==None:
        fig=make_subplots(rows=2,cols=1+len(accs_losses_g),subplot_titles=[r'$\text{(a) Pruning - whole network}$',r'$\text{(b) Pruning - first convolutional layer}$',r'$\text{(c) Pruning - second convolutional layer}$',r'$\text{(d) Pruning - fully connected layer}$',r'$\text{(e) Pruning - whole network}$',r'$\text{(f) Pruning - first convolutional layer}$',r'$\text{(g) Pruning - second convolutional layer}$',r'$\text{(h) Pruning - fully connected layer}$'])

    fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_g_al[0],marker=dict(color='red'),name='Grokking',showlegend=True),row=1,col=1)
    fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_ng_al[0],marker=dict(color='blue'),name='Learning',showlegend=True),row=1,col=1)
    fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_g_al[1],marker=dict(color='red'),name='Grokking',showlegend=False),row=2,col=1)
    fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_ng_al[1],marker=dict(color='blue'),name='Learning',showlegend=False),row=2,col=1)    

    #fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_g_global[0],marker=dict(color='red'),name='Grokking',showlegend=True),row=1,col=2)
    #fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_ng_global[0],marker=dict(color='blue'),name='Learning',showlegend=True),row=1,col=2)
    #fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_g_global[1],marker=dict(color='red'),name='Grokking',showlegend=False),row=2,col=2)
    #fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_ng_global[1],marker=dict(color='blue'),name='Learning',showlegend=False),row=2,col=2)
    for i in range(len(accs_losses_g)):
        fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_g[i][0],marker=dict(color='red'),name='Grokking',showlegend=False),row=1,col=2+i)
        fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_ng[i][0],marker=dict(color='blue'),name='Learning',showlegend=False),row=1,col=2+i)
        showleg=False
        fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_g[i][1],marker=dict(color='red'),name='Grok',showlegend=False),row=2,col=2+i)
        fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_ng[i][1],marker=dict(color='blue'),name='Grok',showlegend=False),row=2,col=2+i)
        fig.update_xaxes(title_text=r"$\text{Percent pruned}$",row=1,col=1)
        fig.update_yaxes(title_text=r"$\text{Accuracy}$", row=1, col=1)
        fig.update_xaxes(title_text=r"$\text{Percent pruned}$", row=2, col=1)
        fig.update_yaxes(title_text=r"$\text{Loss}$",type='log', row=2, col=1)
        fig.update_xaxes(title_text=r"$\text{Percent pruned}$",row=1,col=2+i)
        fig.update_yaxes(title_text=r"$\text{Accuracy}$", row=1, col=2+i)
        fig.update_xaxes(title_text=r"$\text{Percent pruned}$", row=2, col=2+i)
        fig.update_yaxes(title_text=r"$\text{Loss}$",type='log', row=2, col=2+i)

    # fig.add_trace(go.Scatter(x=zero_g,y=acc_g,marker=dict(color='red'),name='Grok',showlegend=False),row=2,col=1)
    # fig.add_trace(go.Scatter(x=zero_ng,y=acc_ng,marker=dict(color='blue'),name='Grok',showlegend=False),row=2,col=1)
    # fig.add_trace(go.Scatter(x=zero_g,y=loss_g,marker=dict(color='red'),name='Grok',showlegend=False),row=2,col=2)
    # fig.add_trace(go.Scatter(x=zero_ng,y=loss_ng,marker=dict(color='blue'),name='Grok',showlegend=False),row=2,col=2)

    # fig.update_xaxes(title_text="Percent pruned", row=2, col=1)
    # fig.update_yaxes(title_text="Accuracy", row=2, col=1)
    # fig.update_xaxes(title_text="Percent pruned", row=2, col=2)
    # fig.update_yaxes(title_text="Loss",type='log', row=2, col=2)



    return fig


#function to check that the run parameters the same in an averaged run
def check_all_variables_same(file_list):
    return None
def group_runs_to_dic(runfolder):
    file_list=[os.path.join(runfolder, f) for f in os.listdir(runfolder) if os.path.isfile(os.path.join(runfolder, f))]
    seed_dic={}
    for filename in file_list:
        seed=int(filename.split('seed_')[1].split('_')[0])
        grok=filename.split('grok_')[1][0]
        with open(filename, 'rb') as in_strm:
            runobj = torch.load(in_strm,map_location=device)
        if (seed in seed_dic.keys())==False:
            seed_dic[seed]=[None,None]
        
        if grok=='T':
            seed_dic[seed][0]=runobj
        elif grok=='F':
            seed_dic[seed][1]=runobj
    
    return seed_dic

def group_runs_to_dic_ram(runfolder):
    file_list=[os.path.join(runfolder, f) for f in os.listdir(runfolder) if os.path.isfile(os.path.join(runfolder, f))]
    seed_dic={}
    for filename in file_list:
        seed=int(filename.split('seed_')[1].split('_')[0])
        grok=filename.split('grok_')[1][0]
        if (seed in seed_dic.keys())==False:
            seed_dic[seed]=[None,None]
        if grok=='T':
            seed_dic[seed][0]=filename
        elif grok=='F':
            seed_dic[seed][1]=filename
    
    return seed_dic

def find_closest_pairs_vectorized(list1, list2):
    arr1 = np.array(list1)
    arr2 = np.array(list2)
    
    x1, y1 = arr1[:, 0], arr1[:, 1]
    x2, y2 = arr2[:, 0], arr2[:, 1]
    
    y1 = y1[:, np.newaxis]  # Reshape y1 to a column vector for broadcasting
    differences = np.abs(y1 - y2)  # Compute the absolute differences
    
    closest_indices = np.argmin(differences, axis=1)  # Find indices of the closest y2
    
    closest_x2 = x2[closest_indices]
    closest_y2 = y2[closest_indices]
    
    result = np.column_stack((x1, y1.flatten(), closest_x2, closest_y2))
    
    return result

def find_max_x2_with_flat_result(x1_y1, x2_y2):
    # Convert lists of pairs into numpy arrays with the appropriate type
    x1, y1 = np.array(x1_y1, dtype=float).T
    x2, y2 = np.array(x2_y2, dtype=float).T

    # Prepare the output list to store the result tuples
    result = []

    # Vectorized comparison of each y2 with all y1s
    for current_x2, current_y2 in zip(x2, y2):
        # Filter y1 to find where y1 > current_y2
        valid_indices = y1 > current_y2

        # If there's at least one y1 greater than current_y2, find the maximum x1
        if np.any(valid_indices):
            max_x1_index = np.argmax(x1[valid_indices])  # Get the index of the maximum x1 that meets the condition
            corresponding_y1_index = np.where(valid_indices)[0][max_x1_index]  # Map it back to the original index
            max_x1 = x1[corresponding_y1_index]
            max_y1 = y1[corresponding_y1_index]

            result.append([max_x1, max_y1, current_x2, current_y2])
        else:
            # If no valid x1 is found, append NaNs or a similar placeholder
            result.append([np.nan, np.nan, current_x2, current_y2])

    return np.array(result)

# Example usage
# x1_y1 = [(10, 20), (15, 25), (5, 15)]
# x2_y2 = [(8, 17), (20, 18), (25, 30)]
# result = find_min_x2_with_pairs(x1_y1, x2_y2)
# print(result)




def find_compression_factor(grokked_pruning_acc,non_grokked_pruning_acc,percents,start,stop):
    grok_curve=[(grokked_pruning_acc[i],percents[i]) for i in range(len(percents))]
    #print(f' grok_curve {grok_curve}')
    non_grok_curve=[(non_grokked_pruning_acc[i],percents[i]) for i in range(len(percents))]
    #print(f' no grok curve {non_grok_curve}')
    grok_curve=[grok_curve[i] for i in range(len(percents)) if (grok_curve[i][0]<start and grok_curve[i][0]>stop)]
    non_grok_curve=[non_grok_curve[i] for i in range(len(percents)) if (non_grok_curve[i][0]<start and non_grok_curve[i][0]>stop)]
    if len(grok_curve)==0 or len(non_grok_curve)==0:
        compression_factors=np.array([])
    else:
        closest_sizes=find_closest_pairs_vectorized(grok_curve,non_grok_curve)
        compression_factors=(1-closest_sizes[:,1])/(1-closest_sizes[:,-1])
    print(f' compression factors {compression_factors}')
        #accuracies=np.stack(closest_sizes[:,0],closest_sizes[:,1])
        #print(accuracies)
        
    
    target_length=len(percents)
    compression_factors=np.pad(compression_factors, (0, max(0, target_length - len(compression_factors))), constant_values=np.nan)
    
    return compression_factors

    
def find_compression_factor2(grokked_pruning_acc,non_grokked_pruning_acc,percents,start,stop):
    grok_curve=[(percents[i],grokked_pruning_acc[i]) for i in range(len(percents))]
    #print(f' grok_curve {grok_curve}')
    non_grok_curve=[(percents[i],non_grokked_pruning_acc[i]) for i in range(len(percents))]
    #print(f' no grok curve {non_grok_curve}')
    #grok_curve=[grok_curve[i] for i in range(len(percents)) if (grok_curve[i][1]<start and grok_curve[i][1]>stop)]
    non_grok_curve=[non_grok_curve[i] for i in range(len(percents)) if (non_grok_curve[i][1]<start and non_grok_curve[i][1]>stop)]
    if len(grok_curve)==0 or len(non_grok_curve)==0:
        compression_factors=np.array([])
    else:
        closest_sizes=np.array(find_max_x2_with_flat_result(grok_curve,non_grok_curve))
        
        print(f' closest sizes {closest_sizes}')

        closest_sizes=np.array(find_max_x2_with_flat_result(non_grok_curve,grok_curve))

        print(f' closest sizes reversed {closest_sizes}')
        
        compression_factors=(1-closest_sizes[:,1])/(1-closest_sizes[:,-1])
    print(f' compression factors {compression_factors}')
        #accuracies=np.stack(closest_sizes[:,0],closest_sizes[:,1])
        #print(accuracies)
        
    
    target_length=len(percents)
    compression_factors=np.pad(compression_factors, (0, max(0, target_length - len(compression_factors))), constant_values=np.nan)
    
    return compression_factors



def magnitude_prune_prod_avg(runfolder,pruning_percents,layers_pruned,fig,epoch):
    seed_dic=group_runs_to_dic_ram(runfolder)
    each_layer_grokked=[]
    each_layer_nongrokked=[]
    all_layer_grokked=[]
    all_layer_nongrokked=[]


    all_layers_compression_factor=[]
    each_layer_compression_factors=[]
  

    for seed in tqdm(seed_dic.keys(),desc="Seeds done",position=0,leave=True):#seed_dic.keys():
        with open(seed_dic[seed][0], 'rb') as in_strm:
            grokked_object = torch.load(in_strm,map_location=device)
        with open(seed_dic[seed][1], 'rb') as in_strm:
            non_grokked_object = torch.load(in_strm,map_location=device)

        if seed<3:
            grokked_object.plot_traincurves(non_grokked_object).show()
                
        original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
        original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)
        original_weights_g = save_original_weights(original_model_g, layers_pruned)
        original_weights_ng = save_original_weights(original_model_ng, layers_pruned)


        original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
        original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)

        accs_losses_g_al=iterative_prune_from_original(original_model_g, original_weights_g, layers_pruned, pruning_percents,'all_layers')
        accs_losses_ng_al=iterative_prune_from_original(original_model_ng, original_weights_ng, layers_pruned, pruning_percents,'all_layers')

        all_layers_compression_factor.append(find_compression_factor(grokked_pruning_acc=accs_losses_g_al[0],non_grokked_pruning_acc=accs_losses_ng_al[0],percents=pruning_percents,start=0.9,stop=0.7))
        
        

        all_layer_grokked.append(np.array(accs_losses_g_al))
        all_layer_nongrokked.append(np.array(accs_losses_ng_al))

        original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
        original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)

        accs_losses_g=[iterative_prune_from_original(original_model_g, original_weights_g, [i], pruning_percents,'local') for i in layers_pruned]
        accs_losses_ng=[iterative_prune_from_original(original_model_ng, original_weights_ng, [i], pruning_percents,'local') for i in layers_pruned]
        each_layer_grokked.append(np.array(accs_losses_g))
        each_layer_nongrokked.append(np.array(accs_losses_ng))
        layer_compressions=[]

        for layer_ind in range(len(accs_losses_g)):
            layer_compression=find_compression_factor(grokked_pruning_acc=accs_losses_g[layer_ind][0],non_grokked_pruning_acc=accs_losses_ng[layer_ind][0],percents=pruning_percents,start=0.9,stop=0.65)
            layer_compressions.append(layer_compression)
        each_layer_compression_factors.append(layer_compressions)
        
    

    avg_each_layer_grokked=np.mean(np.array(each_layer_grokked),axis=0)
    avg_each_layer_nongrokked=np.mean(np.array(each_layer_nongrokked),axis=0)
    avg_all_layer_grokked=np.mean(np.array(all_layer_grokked),axis=0)
    avg_all_layer_nongrokked=np.mean(np.array(all_layer_nongrokked),axis=0)

    # print(all_layers_compression_factor)
    
    
    all_layers_compression_factor_nonnan=np.ma.masked_invalid(np.array(all_layers_compression_factor))
    all_layers_compression_factor_nonnan[all_layers_compression_factor_nonnan>100]=np.nan
    max_all_layers_compression_factor=np.max(all_layers_compression_factor_nonnan,axis=1)

    med_all_layers_compression_factor=np.median(np.array(all_layers_compression_factor_nonnan),axis=1)
    
    max_all_layers_compression_factor=np.where(max_all_layers_compression_factor.mask, np.nan, max_all_layers_compression_factor.data)
    # print(max_all_layers_compression_factor)

    each_layer_compression_factor_nonan=np.ma.masked_invalid(np.array(each_layer_compression_factors))
    each_layer_compression_factor_nonan[each_layer_compression_factor_nonan>100]=np.nan
    # print(each_layer_compression_factor_nonan)
    max_each_layer=np.max(each_layer_compression_factor_nonan,axis=2)
    max_each_layer=np.where(max_each_layer.mask, np.nan, max_each_layer.data)
    
    


    if fig==None:
        fig=make_subplots(rows=2,cols=2+len(accs_losses_g),subplot_titles=[r'$\text{(a) Pruning - whole network}$',r'$\text{(b) Pruning - first convolutional layer}$',r'$\text{(c) Pruning - second convolutional layer}$',r'$\text{(d) Pruning - fully connected layer}$',r'$\text{(e) Pruning - whole network}$',r'$\text{(f) Pruning - first convolutional layer}$',r'$\text{(g) Pruning - second convolutional layer}$',r'$\text{(h) Pruning - fully connected layer}$'])
    # if fig==None:
    #     fig=make_subplots(rows=2,cols=2,subplot_titles=[r'$\text{(a) Pruning - whole network}$',r'$\text{(a) Pruning - whole network}$',r'$\text{(a) Pruning - whole network}$',r'$\text{(a) Pruning - Compressions}$'])
    fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_grokked[0],marker=dict(color='red'),name='Grokking',showlegend=True),row=1,col=1)
    fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_nongrokked[0],marker=dict(color='blue'),name='Learning',showlegend=True),row=1,col=1)
    fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_grokked[1],marker=dict(color='red'),name='Grokking',showlegend=False),row=2,col=1)
    fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_nongrokked[1],marker=dict(color='blue'),name='Learning',showlegend=False),row=2,col=1)    

    for i in range(len(accs_losses_g)):
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_each_layer_grokked[i][0],marker=dict(color='red'),name='Grokking',showlegend=False),row=1,col=2+i)
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_each_layer_nongrokked[i][0],marker=dict(color='blue'),name='Learning',showlegend=False),row=1,col=2+i)
        showleg=False
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_each_layer_grokked[i][1],marker=dict(color='red'),name='Grok',showlegend=False),row=2,col=2+i)
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_each_layer_nongrokked[i][1],marker=dict(color='blue'),name='Grok',showlegend=False),row=2,col=2+i)
        fig.update_xaxes(title_text=r"$\text{Percent pruned}$",row=1,col=1)
        fig.update_yaxes(title_text=r"$\text{Accuracy}$", row=1, col=1)
        fig.update_xaxes(title_text=r"$\text{Percent pruned}$", row=2, col=1)
        fig.update_yaxes(title_text=r"$\text{Loss}$",type='log', row=2, col=1)
        fig.update_xaxes(title_text=r"$\text{Percent pruned}$",row=1,col=2+i)
        fig.update_yaxes(title_text=r"$\text{Accuracy}$", row=1, col=2+i)
        fig.update_xaxes(title_text=r"$\text{Percent pruned}$", row=2, col=2+i)
        fig.update_yaxes(title_text=r"$\text{Loss}$",type='log', row=2, col=2+i)

    plotarray=np.where(np.isnan(max_all_layers_compression_factor), 0, max_all_layers_compression_factor)
    fig.add_trace(go.Bar(x=list(range(len(plotarray))),y=plotarray,marker=dict(color='orange'),name='Compression',showlegend=False),row=1,col=2+len(accs_losses_g))

    plotarrayeachlayer=np.where(np.isnan(max_each_layer), 0, max_each_layer)
    for layermax in range(len(plotarrayeachlayer[0])):
        fig.add_trace(go.Bar(x=list(range(len(plotarrayeachlayer))),y=[x[layermax] for x in plotarrayeachlayer],marker=dict(color='orange'),name='Compression',showlegend=False),row=2,col=2+len(accs_losses_g))
    
    #fig2=make_subplots(rows=1,cols=len(list(seed_dic.keys())))
    # for seed in seed_dic.keys():
    #     single_run=seed_dic[seed][0]
    #     single_run_ng=seed_dic[seed][1]
    #     single_run.plot_traincurves(single_run_ng).show()

    return fig

def magnitude_prune_epochs_avg(runfolder,pruning_percents,layers_pruned,fig,epochs):
    
    seed_dic=group_runs_to_dic_ram(runfolder)
    avg_all_layer_grokked_epochs=[]
    avg_all_layer_non_grokked_epochs=[]
    avg_all_layer_integral_epochs=[]
    grok_IPR_epochs=[]
    non_grok_IPR_epochs=[]
    
    avg_train_acc_grokked=[]
    avg_test_acc_grokked=[]
    avg_test_loss_grokked=[]
    avg_train_loss_grokked=[]

    avg_train_acc_non_grokked=[]
    avg_test_acc_non_grokked=[]
    avg_test_loss_non_grokked=[]
    avg_train_loss_non_grokked=[]


    first=True
    for epoch in epochs:
        

        all_layer_grokked=[]
        all_layer_nongrokked=[]
        all_layer_IPR_grok=[]
        all_layer_IPR_non_grok=[]
    
        for seed in tqdm(seed_dic.keys(),desc="Seeds done",position=0,leave=True):#seed_dic.keys():
            with open(seed_dic[seed][0], 'rb') as in_strm:
                grokked_object = torch.load(in_strm,map_location=device)
            with open(seed_dic[seed][1], 'rb') as in_strm:
                non_grokked_object = torch.load(in_strm,map_location=device)
            
            if first:
                avg_train_acc_grokked.append(np.array(grokked_object.train_accuracies))
                avg_test_acc_grokked.append(np.array(grokked_object.test_accuracies))
                avg_train_loss_grokked.append(np.array(grokked_object.train_losses))
                avg_test_loss_grokked.append(np.array(grokked_object.test_losses))

                avg_train_acc_non_grokked.append(np.array(non_grokked_object.train_accuracies))
                avg_test_acc_non_grokked.append(np.array(non_grokked_object.test_accuracies))
                avg_train_loss_non_grokked.append(np.array(non_grokked_object.train_losses))
                avg_test_loss_non_grokked.append(np.array(non_grokked_object.test_losses))
                

            # if seed<1:
            #     grokked_object.plot_traincurves(non_grokked_object).show()

                    
            original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
            original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)
            original_weights_g = save_original_weights(original_model_g, layers_pruned)
            original_weights_ng = save_original_weights(original_model_ng, layers_pruned)
            #IPR
            r=2
            grok_layer_weights=[original_model_dic_g[key+'.weight'] for key in layers_pruned]
            non_grok_layer_weights=[original_model_dic_ng[key+'.weight'] for key in layers_pruned]
            grok_weights=torch.cat([torch.flatten(x) for x in grok_layer_weights])
            grok_IPR=torch.sum(torch.abs(grok_weights)**(2*r))/(torch.sum(grok_weights**2)**r)
            non_grok_weights=torch.cat([torch.flatten(x) for x in non_grok_layer_weights])
            non_grok_IPR=torch.sum(torch.abs(non_grok_weights)**(2*r))/(torch.sum(non_grok_weights**2)**r)

            accs_losses_g_al=iterative_prune_from_original(original_model_g, original_weights_g, layers_pruned, pruning_percents,'all_layers')
            accs_losses_ng_al=iterative_prune_from_original(original_model_ng, original_weights_ng, layers_pruned, pruning_percents,'all_layers')

            all_layer_grokked.append(np.array(accs_losses_g_al))
            all_layer_nongrokked.append(np.array(accs_losses_ng_al))   
            all_layer_IPR_grok.append(grok_IPR.item())
            all_layer_IPR_non_grok.append(non_grok_IPR.item())    
        
        first=False
        avg_all_layer_grokked=np.mean(np.array(all_layer_grokked),axis=0)
        avg_all_layer_nongrokked=np.mean(np.array(all_layer_nongrokked),axis=0)
        #avg_squared_difference=np.mean((np.array(all_layer_grokked)-np.array(all_layer_nongrokked))**2)
        avg_squared_difference=(avg_all_layer_grokked[0]-avg_all_layer_nongrokked[0])**2
        avg_all_layer_integral_epochs.append(np.trapz(avg_squared_difference,pruning_percents))
        
        avg_all_layer_grokked_epochs.append(avg_all_layer_grokked)
        avg_all_layer_non_grokked_epochs.append(avg_all_layer_nongrokked)
        grok_IPR_epochs.append(np.mean(all_layer_IPR_grok,axis=0))
        non_grok_IPR_epochs.append(np.mean(all_layer_IPR_non_grok,axis=0))
        #avg_all_layer_squared_difference_epochs.append(avg_all_laye)

    #Epcoh independents

    #Integral
    

    #Training curves
    average_test_acc_grokked=np.mean(avg_test_acc_grokked,axis=0)
    average_train_acc_grokked=np.mean(avg_train_acc_grokked,axis=0)
    average_test_acc_non_grokked=np.mean(avg_test_acc_non_grokked,axis=0)
    average_train_acc_non_grokked=np.mean(avg_train_acc_non_grokked,axis=0)
    average_test_loss_grokked=np.mean(avg_test_loss_grokked,axis=0)
    average_train_loss_grokked=np.mean(avg_train_loss_grokked,axis=0)
    average_test_loss_non_grokked=np.mean(avg_test_loss_non_grokked,axis=0)
    average_train_loss_non_grokked=np.mean(avg_train_loss_non_grokked,axis=0)
    
    
    


    if fig==None:
        fig=make_subplots(rows=4,cols=4,subplot_titles=[r'$\text{Grok accuracy}$',r'$\text{Learn accuracy}$',r'$\text{Grok Loss}$',r'$\text{Learn Loss}$']+[r'$\text{L2 integral between curves}$',r'$\text{Grokking Pruning Curves}$',r'$\text{Learn Pruning}$',r'$\text{IPR2}$'])#['Epoch '  + str(epochs[i])+' Pruning' for i in range(len(epochs))]
    # if fig==None:
    #     fig=make_subplots(rows=2,cols=2,subplot_titles=[r'$\text{(a) Pruning - whole network}$',r'$\text{(a) Pruning - whole network}$',r'$\text{(a) Pruning - whole network}$',r'$\text{(a) Pruning - Compressions}$'])
    
    #Training curves
    fig.add_trace(go.Scatter(x=list(range(len(average_test_acc_grokked))),y=average_test_acc_grokked,mode="lines", line=dict(color="red",dash='solid'),name='Grokking test',showlegend=True),row=1,col=1)
    fig.add_trace(go.Scatter(x=list(range(len(average_train_acc_grokked))),y=average_train_acc_grokked,mode="lines", line=dict(color="red",dash='dash'),name='Grokking train',showlegend=True),row=1,col=1)

    fig.add_trace(go.Scatter(x=list(range(len(average_test_acc_non_grokked))),y=average_test_acc_non_grokked,mode="lines", line=dict(color="blue",dash='solid'),name='Learning test',showlegend=True),row=1,col=2)
    fig.add_trace(go.Scatter(x=list(range(len(average_train_acc_non_grokked))),y=average_train_acc_non_grokked,mode="lines", line=dict(color="blue",dash='dash'),name='Learning train',showlegend=True),row=1,col=2)

    fig.add_trace(go.Scatter(x=list(range(len(average_test_loss_grokked))),y=average_test_loss_grokked,mode="lines", line=dict(color="red",dash='solid'),name='Grokking test',showlegend=False),row=1,col=3)
    fig.add_trace(go.Scatter(x=list(range(len(average_train_loss_grokked))),y=average_train_loss_grokked,mode="lines", line=dict(color="red",dash='dash'),name='Grokking train',showlegend=False),row=1,col=3)
    fig.update_yaxes(title_text=r'$\text{Loss}$',type='log',row=1,col=3)

    fig.add_trace(go.Scatter(x=list(range(len(average_test_loss_non_grokked))),y=average_test_loss_non_grokked,mode="lines", line=dict(color="blue",dash='solid'),name='Learn test',showlegend=False),row=1,col=4)
    fig.add_trace(go.Scatter(x=list(range(len(average_train_loss_non_grokked))),y=average_train_loss_non_grokked,mode="lines", line=dict(color="blue",dash='dash'),name='Learn train',showlegend=False),row=1,col=4)
    fig.update_yaxes(title_text=r'$\text{Loss}$',type='log',row=1,col=4)

    for epoch_ind in range(len(epochs)):
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_grokked_epochs[epoch_ind][0],marker=dict(color='red'),name='Grokking',showlegend=False),row=2+epoch_ind//4,col=1+epoch_ind%4)
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_non_grokked_epochs[epoch_ind][0],marker=dict(color='blue'),name='Learning',showlegend=False),row=2+epoch_ind//4,col=1+epoch_ind%4)
    #Individual epoch plots
    fig.add_trace(go.Scatter(x=epochs,y=avg_all_layer_integral_epochs,marker=dict(color='black'),name='Difference integral',showlegend=False),row=4,col=1)
    for epoch_ind in range(len(epochs)):
        intensity = int(255 * (epoch_ind + 1) / len(epochs))  # Increase redness with epoch index
        color = f'rgb({intensity}, 0, 0)'
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_grokked_epochs[epoch_ind][0],marker=dict(color=color),name=f'Grok {epochs[epoch_ind]}',showlegend=True),row=4,col=2)
        color = f'rgb(0, 0, {intensity})'
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_non_grokked_epochs[epoch_ind][0],marker=dict(color=color),name=f'Non-grok {epochs[epoch_ind]}',showlegend=True),row=4,col=3)
    
    fig.add_trace(go.Scatter(x=epochs,y=grok_IPR_epochs,marker=dict(color='red'),name='IPR2 Grok',showlegend=True),row=4,col=4)
    fig.add_trace(go.Scatter(x=epochs,y=non_grok_IPR_epochs,marker=dict(color='blue'),name='IPR2 Learn',showlegend=True),row=4,col=4)
    fig.update_xaxes(title_text='Epoch',row=4,col=4)
    #fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_non_grokked_epochs[-1][0],marker=dict(color=color),name='Final Learn',showlegend=False),row=2,col=2)
    #fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_grokked_epochs[-1][0],marker=dict(color=color),name='Final Grok',showlegend=False),row=2,col=3)
    


    return fig

        #fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_grokked[1],marker=dict(color='red'),name='Grokking',showlegend=False),row=2,col=1)
        #fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_nongrokked[1],marker=dict(color='blue'),name='Learning',showlegend=False),row=2,col=1)    



def magnitude_prune_epochs_ipr(runfolder,pruning_percents,layers_pruned,fig,epochs):
    
    seed_dic=group_runs_to_dic_ram(runfolder)
    avg_all_layer_grokked_epochs=[]
    avg_all_layer_non_grokked_epochs=[]
    avg_all_layer_integral_epochs=[]
    grok_IPR_epochs=[]
    non_grok_IPR_epochs=[]
    
    avg_train_acc_grokked=[]
    avg_test_acc_grokked=[]
    avg_test_loss_grokked=[]
    avg_train_loss_grokked=[]

    avg_train_acc_non_grokked=[]
    avg_test_acc_non_grokked=[]
    avg_test_loss_non_grokked=[]
    avg_train_loss_non_grokked=[]


    first=True
    for epoch in epochs:
        

        all_layer_grokked=[]
        all_layer_nongrokked=[]
        all_layer_IPR_grok=[]
        all_layer_IPR_non_grok=[]
    
        for seed in tqdm(seed_dic.keys(),desc="Seeds done",position=0,leave=True):#seed_dic.keys():
            with open(seed_dic[seed][0], 'rb') as in_strm:
                grokked_object = torch.load(in_strm,map_location=device)
            with open(seed_dic[seed][1], 'rb') as in_strm:
                non_grokked_object = torch.load(in_strm,map_location=device)
            
            if first:
                avg_train_acc_grokked.append(np.array(grokked_object.train_accuracies))
                avg_test_acc_grokked.append(np.array(grokked_object.test_accuracies))
                avg_train_loss_grokked.append(np.array(grokked_object.train_losses))
                avg_test_loss_grokked.append(np.array(grokked_object.test_losses))

                avg_train_acc_non_grokked.append(np.array(non_grokked_object.train_accuracies))
                avg_test_acc_non_grokked.append(np.array(non_grokked_object.test_accuracies))
                avg_train_loss_non_grokked.append(np.array(non_grokked_object.train_losses))
                avg_test_loss_non_grokked.append(np.array(non_grokked_object.test_losses))
                

            # if seed<1:
            #     grokked_object.plot_traincurves(non_grokked_object).show()

                    
            original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
            original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)
            original_weights_g = save_original_weights(original_model_g, layers_pruned)
            original_weights_ng = save_original_weights(original_model_ng, layers_pruned)
            #IPR
            r=2
            grok_layer_weights=[original_model_dic_g[key+'.weight'] for key in layers_pruned]
            non_grok_layer_weights=[original_model_dic_ng[key+'.weight'] for key in layers_pruned]
            grok_weights=torch.cat([torch.flatten(x) for x in grok_layer_weights])
            grok_IPR=torch.sum(torch.abs(grok_weights)**(2*r))/(torch.sum(grok_weights**2)**r)
            non_grok_weights=torch.cat([torch.flatten(x) for x in non_grok_layer_weights])
            non_grok_IPR=torch.sum(torch.abs(non_grok_weights)**(2*r))/(torch.sum(non_grok_weights**2)**r)

            accs_losses_g_al=iterative_prune_from_original(original_model_g, original_weights_g, layers_pruned, pruning_percents,'all_layers')
            accs_losses_ng_al=iterative_prune_from_original(original_model_ng, original_weights_ng, layers_pruned, pruning_percents,'all_layers')

            all_layer_grokked.append(np.array(accs_losses_g_al))
            all_layer_nongrokked.append(np.array(accs_losses_ng_al))   
            all_layer_IPR_grok.append(grok_IPR.item())
            all_layer_IPR_non_grok.append(non_grok_IPR.item())    
        
        first=False
        avg_all_layer_grokked=np.mean(np.array(all_layer_grokked),axis=0)
        avg_all_layer_nongrokked=np.mean(np.array(all_layer_nongrokked),axis=0)
        #avg_squared_difference=np.mean((np.array(all_layer_grokked)-np.array(all_layer_nongrokked))**2)
        avg_squared_difference=(avg_all_layer_grokked[0]-avg_all_layer_nongrokked[0])**2
        avg_all_layer_integral_epochs.append(np.trapz(avg_squared_difference,pruning_percents))
        
        avg_all_layer_grokked_epochs.append(avg_all_layer_grokked)
        avg_all_layer_non_grokked_epochs.append(avg_all_layer_nongrokked)
        grok_IPR_epochs.append(np.mean(all_layer_IPR_grok,axis=0))
        non_grok_IPR_epochs.append(np.mean(all_layer_IPR_non_grok,axis=0))
        #avg_all_layer_squared_difference_epochs.append(avg_all_laye)

    #Epcoh independents

    #Integral
    

    #Training curves
    average_test_acc_grokked=np.mean(avg_test_acc_grokked,axis=0)
    average_train_acc_grokked=np.mean(avg_train_acc_grokked,axis=0)
    average_test_acc_non_grokked=np.mean(avg_test_acc_non_grokked,axis=0)
    average_train_acc_non_grokked=np.mean(avg_train_acc_non_grokked,axis=0)
    average_test_loss_grokked=np.mean(avg_test_loss_grokked,axis=0)
    average_train_loss_grokked=np.mean(avg_train_loss_grokked,axis=0)
    average_test_loss_non_grokked=np.mean(avg_test_loss_non_grokked,axis=0)
    average_train_loss_non_grokked=np.mean(avg_train_loss_non_grokked,axis=0)
    
    
    


    if fig==None:
        fig=make_subplots(rows=2,cols=4,subplot_titles=[r'$\text{Grok accuracy}$',r'$\text{Learn accuracy}$',r'$\text{Grok Loss}$',r'$\text{Learn Loss}$']+[r'$\text{L2 integral between curves}$',r'$\text{Grokking Pruning Curves}$',r'$\text{Learn Pruning}$',r'$\text{IPR2}$'])#['Epoch '  + str(epochs[i])+' Pruning' for i in range(len(epochs))]
    # if fig==None:
    #     fig=make_subplots(rows=2,cols=2,subplot_titles=[r'$\text{(a) Pruning - whole network}$',r'$\text{(a) Pruning - whole network}$',r'$\text{(a) Pruning - whole network}$',r'$\text{(a) Pruning - Compressions}$'])
    
    #Training curves
    fig.add_trace(go.Scatter(x=list(range(len(average_test_acc_grokked))),y=average_test_acc_grokked,mode="lines", line=dict(color="red",dash='solid'),name='Grokking test',showlegend=True),row=1,col=1)
    fig.add_trace(go.Scatter(x=list(range(len(average_train_acc_grokked))),y=average_train_acc_grokked,mode="lines", line=dict(color="red",dash='dash'),name='Grokking train',showlegend=True),row=1,col=1)

    fig.add_trace(go.Scatter(x=list(range(len(average_test_acc_non_grokked))),y=average_test_acc_non_grokked,mode="lines", line=dict(color="blue",dash='solid'),name='Learning test',showlegend=True),row=1,col=2)
    fig.add_trace(go.Scatter(x=list(range(len(average_train_acc_non_grokked))),y=average_train_acc_non_grokked,mode="lines", line=dict(color="blue",dash='dash'),name='Learning train',showlegend=True),row=1,col=2)

    fig.add_trace(go.Scatter(x=list(range(len(average_test_loss_grokked))),y=average_test_loss_grokked,mode="lines", line=dict(color="red",dash='solid'),name='Grokking test',showlegend=False),row=1,col=3)
    fig.add_trace(go.Scatter(x=list(range(len(average_train_loss_grokked))),y=average_train_loss_grokked,mode="lines", line=dict(color="red",dash='dash'),name='Grokking train',showlegend=False),row=1,col=3)
    fig.update_yaxes(title_text=r'$\text{Loss}$',type='log',row=1,col=3)

    fig.add_trace(go.Scatter(x=list(range(len(average_test_loss_non_grokked))),y=average_test_loss_non_grokked,mode="lines", line=dict(color="blue",dash='solid'),name='Learn test',showlegend=False),row=1,col=4)
    fig.add_trace(go.Scatter(x=list(range(len(average_train_loss_non_grokked))),y=average_train_loss_non_grokked,mode="lines", line=dict(color="blue",dash='dash'),name='Learn train',showlegend=False),row=1,col=4)
    fig.update_yaxes(title_text=r'$\text{Loss}$',type='log',row=1,col=4)

    #Individual epoch plots
    fig.add_trace(go.Scatter(x=epochs,y=avg_all_layer_integral_epochs,marker=dict(color='black'),name='Difference integral',showlegend=False),row=2,col=1)
    for epoch_ind in range(len(epochs)):
        intensity = int(255 * (epoch_ind + 1) / len(epochs))  # Increase redness with epoch index
        color = f'rgb({intensity}, 0, 0)'
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_grokked_epochs[epoch_ind][0],marker=dict(color=color),name=f'Grok {epochs[epoch_ind]}',showlegend=True),row=2,col=2)
        color = f'rgb(0, 0, {intensity})'
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_non_grokked_epochs[epoch_ind][0],marker=dict(color=color),name=f'Non-grok {epochs[epoch_ind]}',showlegend=True),row=2,col=3)
    
    #fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_non_grokked_epochs[-1][0],marker=dict(color=color),name='Final Learn',showlegend=False),row=2,col=2)
    #fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_grokked_epochs[-1][0],marker=dict(color=color),name='Final Grok',showlegend=False),row=2,col=3)
    
    fig.add_trace(go.Scatter(x=epochs,y=grok_IPR_epochs,marker=dict(color='red'),name='IPR2 Grok',showlegend=True),row=2,col=4)
    fig.add_trace(go.Scatter(x=epochs,y=non_grok_IPR_epochs,marker=dict(color='blue'),name='IPR2 Learn',showlegend=True),row=2,col=4)
    fig.update_xaxes(title_text='Epoch',row=2,col=4)

        
    # for epoch_ind in range(len(epochs)):
    #     fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_grokked_epochs[epoch_ind][0],marker=dict(color='red'),name='Grokking',showlegend=False),row=2+epoch_ind//4,col=1+epoch_ind%4)
    #     fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_non_grokked_epochs[epoch_ind][0],marker=dict(color='blue'),name='Learning',showlegend=False),row=2+epoch_ind//4,col=1+epoch_ind%4)
    #     #fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_grokked[1],marker=dict(color='red'),name='Grokking',showlegend=False),row=2,col=1)
    #     #fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_nongrokked[1],marker=dict(color='blue'),name='Learning',showlegend=False),row=2,col=1)    


    return fig

def magnitude_avg_ram(runfolder,pruning_percents,layers_pruned,fig,epoch):
    seed_dic=group_runs_to_dic(runfolder)
    each_layer_grokked=[]
    each_layer_nongrokked=[]
    all_layer_grokked=[]
    all_layer_nongrokked=[]


    all_layers_compression_factor=[]
    each_layer_compression_factors=[]
  

    for seed in tqdm(seed_dic.keys(),desc="Seeds done",position=0,leave=True):#seed_dic.keys():
        grokked_object=seed_dic[seed][0]
        non_grokked_object=seed_dic[seed][1]

                
        original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
        original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)
        original_weights_g = save_original_weights(original_model_g, layers_pruned)
        original_weights_ng = save_original_weights(original_model_ng, layers_pruned)


        original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
        original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)

        accs_losses_g_al=iterative_prune_from_original(original_model_g, original_weights_g, layers_pruned, pruning_percents,'all_layers')
        accs_losses_ng_al=iterative_prune_from_original(original_model_ng, original_weights_ng, layers_pruned, pruning_percents,'all_layers')

        all_layers_compression_factor.append(find_compression_factor(grokked_pruning_acc=accs_losses_g_al[0],non_grokked_pruning_acc=accs_losses_ng_al[0],percents=pruning_percents,start=0.9,stop=0.7))
        
        

        all_layer_grokked.append(np.array(accs_losses_g_al))
        all_layer_nongrokked.append(np.array(accs_losses_ng_al))

        original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
        original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)

        accs_losses_g=[iterative_prune_from_original(original_model_g, original_weights_g, [i], pruning_percents,'local') for i in layers_pruned]
        accs_losses_ng=[iterative_prune_from_original(original_model_ng, original_weights_ng, [i], pruning_percents,'local') for i in layers_pruned]
        each_layer_grokked.append(np.array(accs_losses_g))
        each_layer_nongrokked.append(np.array(accs_losses_ng))
        layer_compressions=[]

        for layer_ind in range(len(accs_losses_g)):
            layer_compression=find_compression_factor(grokked_pruning_acc=accs_losses_g[layer_ind][0],non_grokked_pruning_acc=accs_losses_ng[layer_ind][0],percents=pruning_percents,start=0.9,stop=0.65)
            layer_compressions.append(layer_compression)
        each_layer_compression_factors.append(layer_compressions)
        
    

    avg_each_layer_grokked=np.mean(np.array(each_layer_grokked),axis=0)
    avg_each_layer_nongrokked=np.mean(np.array(each_layer_nongrokked),axis=0)
    avg_all_layer_grokked=np.mean(np.array(all_layer_grokked),axis=0)
    avg_all_layer_nongrokked=np.mean(np.array(all_layer_nongrokked),axis=0)

    # print(all_layers_compression_factor)
    
    
    all_layers_compression_factor_nonnan=np.ma.masked_invalid(np.array(all_layers_compression_factor))
    all_layers_compression_factor_nonnan[all_layers_compression_factor_nonnan>100]=np.nan
    max_all_layers_compression_factor=np.max(all_layers_compression_factor_nonnan,axis=1)

    med_all_layers_compression_factor=np.median(np.array(all_layers_compression_factor_nonnan),axis=1)
    
    max_all_layers_compression_factor=np.where(max_all_layers_compression_factor.mask, np.nan, max_all_layers_compression_factor.data)
    # print(max_all_layers_compression_factor)

    each_layer_compression_factor_nonan=np.ma.masked_invalid(np.array(each_layer_compression_factors))
    each_layer_compression_factor_nonan[each_layer_compression_factor_nonan>100]=np.nan
    # print(each_layer_compression_factor_nonan)
    max_each_layer=np.max(each_layer_compression_factor_nonan,axis=2)
    max_each_layer=np.where(max_each_layer.mask, np.nan, max_each_layer.data)
    
    


    if fig==None:
        fig=make_subplots(rows=2,cols=2+len(accs_losses_g),subplot_titles=[r'$\text{(a) Pruning - whole network}$',r'$\text{(b) Pruning - first convolutional layer}$',r'$\text{(c) Pruning - second convolutional layer}$',r'$\text{(d) Pruning - fully connected layer}$',r'$\text{(e) Pruning - whole network}$',r'$\text{(f) Pruning - first convolutional layer}$',r'$\text{(g) Pruning - second convolutional layer}$',r'$\text{(h) Pruning - fully connected layer}$'])
    # if fig==None:
    #     fig=make_subplots(rows=2,cols=2,subplot_titles=[r'$\text{(a) Pruning - whole network}$',r'$\text{(a) Pruning - whole network}$',r'$\text{(a) Pruning - whole network}$',r'$\text{(a) Pruning - Compressions}$'])
    fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_grokked[0],marker=dict(color='red'),name='Grokking',showlegend=True),row=1,col=1)
    fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_nongrokked[0],marker=dict(color='blue'),name='Learning',showlegend=True),row=1,col=1)
    fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_grokked[1],marker=dict(color='red'),name='Grokking',showlegend=False),row=2,col=1)
    fig.add_trace(go.Scatter(x=pruning_percents,y=avg_all_layer_nongrokked[1],marker=dict(color='blue'),name='Learning',showlegend=False),row=2,col=1)    

    for i in range(len(accs_losses_g)):
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_each_layer_grokked[i][0],marker=dict(color='red'),name='Grokking',showlegend=False),row=1,col=2+i)
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_each_layer_nongrokked[i][0],marker=dict(color='blue'),name='Learning',showlegend=False),row=1,col=2+i)
        showleg=False
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_each_layer_grokked[i][1],marker=dict(color='red'),name='Grok',showlegend=False),row=2,col=2+i)
        fig.add_trace(go.Scatter(x=pruning_percents,y=avg_each_layer_nongrokked[i][1],marker=dict(color='blue'),name='Grok',showlegend=False),row=2,col=2+i)
        fig.update_xaxes(title_text=r"$\text{Percent pruned}$",row=1,col=1)
        fig.update_yaxes(title_text=r"$\text{Accuracy}$", row=1, col=1)
        fig.update_xaxes(title_text=r"$\text{Percent pruned}$", row=2, col=1)
        fig.update_yaxes(title_text=r"$\text{Loss}$",type='log', row=2, col=1)
        fig.update_xaxes(title_text=r"$\text{Percent pruned}$",row=1,col=2+i)
        fig.update_yaxes(title_text=r"$\text{Accuracy}$", row=1, col=2+i)
        fig.update_xaxes(title_text=r"$\text{Percent pruned}$", row=2, col=2+i)
        fig.update_yaxes(title_text=r"$\text{Loss}$",type='log', row=2, col=2+i)

    plotarray=np.where(np.isnan(max_all_layers_compression_factor), 0, max_all_layers_compression_factor)
    fig.add_trace(go.Bar(x=list(range(len(plotarray))),y=plotarray,marker=dict(color='orange'),name='Compression',showlegend=False),row=1,col=2+len(accs_losses_g))

    plotarrayeachlayer=np.where(np.isnan(max_each_layer), 0, max_each_layer)
    for layermax in range(len(plotarrayeachlayer[0])):
        fig.add_trace(go.Bar(x=list(range(len(plotarrayeachlayer))),y=[x[layermax] for x in plotarrayeachlayer],marker=dict(color='orange'),name='Compression',showlegend=False),row=2,col=2+len(accs_losses_g))
    
    #fig2=make_subplots(rows=1,cols=len(list(seed_dic.keys())))
    for seed in seed_dic.keys():
        single_run=seed_dic[seed][0]
        single_run_ng=seed_dic[seed][1]
        single_run.plot_traincurves(single_run_ng).show()

    return fig



def correlation_one_epoch(grokked_object,non_grokked_object,model_grok,model_nogrok,sortby,epoch,neuron_index,images_tensor,feature_funcs,fig,dataset):

    features_tensor=construct_features_tensor(images_tensor=images_tensor,feature_funcs=feature_funcs)
    activations_grok, output_grok, cleanup_grok = get_activations(model_grok,images_tensor)
    sorted_activations_grok,acts_indices_grok=get_acts_dict(single_run=grokked_object,dataset=dataset,epoch=epoch,sortby=sortby)

    #removing 
    sorted_activations_grok.pop('conv_layers.2 (MaxPool2d)')
    if 'conv_layers.5 (MaxPool2d)' in sorted_activations_grok:
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    #sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    sorted_activations_grok.pop('fc_layers.3 (Linear)')
    activations_grok.pop('conv_layers.2 (MaxPool2d)')
    if 'conv_layers.5 (MaxPool2d)' in sorted_activations_grok:
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    #activations_grok.pop('conv_layers.5 (MaxPool2d)')
    activations_grok.pop('fc_layers.3 (Linear)')
    acts_indices_grok.pop('conv_layers.2 (MaxPool2d)')
    if 'conv_layers.5 (MaxPool2d)' in sorted_activations_grok:
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    #acts_indices_grok.pop('conv_layers.5 (MaxPool2d)')
    acts_indices_grok.pop('fc_layers.3 (Linear)')



    activations_nogrok, output_nogrok, cleanup_nogrok = get_activations(model_nogrok,images_tensor)
    sorted_activations_nogrok,acts_indices_nogrok=get_acts_dict(single_run=non_grokked_object,dataset=dataset,epoch=epoch,sortby=sortby)

    #removing 
    sorted_activations_nogrok.pop('conv_layers.2 (MaxPool2d)')
    if 'conv_layers.5 (MaxPool2d)' in sorted_activations_grok:
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    #sorted_activations_nogrok.pop('conv_layers.5 (MaxPool2d)')
    sorted_activations_nogrok.pop('fc_layers.3 (Linear)')
    activations_nogrok.pop('conv_layers.2 (MaxPool2d)')
    if 'conv_layers.5 (MaxPool2d)' in sorted_activations_grok:
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    #activations_nogrok.pop('conv_layers.5 (MaxPool2d)')
    activations_nogrok.pop('fc_layers.3 (Linear)')
    acts_indices_nogrok.pop('conv_layers.2 (MaxPool2d)')
    if 'conv_layers.5 (MaxPool2d)' in sorted_activations_grok:
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
    #acts_indices_nogrok.pop('conv_layers.5 (MaxPool2d)')
    acts_indices_nogrok.pop('fc_layers.3 (Linear)')

    #Now you need to populate the layers
    feature_dim=features_tensor.shape[1]
    if fig==None:
        fig=make_subplots(rows=max(4,2*feature_dim),cols=1+len(sorted_activations_grok.keys()),subplot_titles=['Grok loss']+[f'ENERGY, layer{key}' for key in sorted_activations_grok.keys()]+['No grok loss']+[f'ENERGY, layer {key}' for key in sorted_activations_grok.keys()]+['Grok accuracy']+[f'MAG, Layer {key}' for key in sorted_activations_grok.keys()]+['No grok accuracy']+[f'MAG, Layer {key}' for key in sorted_activations_grok.keys()])#(len(grok_weights)+4)//2

    #titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_nogrok))]

    fig.add_trace(go.Scatter(x=list(range(len(grokked_object.train_losses))),y=grokked_object.train_losses,marker=dict(color='black'),showlegend=True,name='Train'),row=1,col=1)
    fig.add_trace(go.Scatter(x=list(range(len(grokked_object.test_losses))),y=grokked_object.test_losses,marker=dict(color='orange'),showlegend=True,name='Test'),row=1,col=1)
    fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(grokked_object.train_losses), max(grokked_object.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

    fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=non_grokked_object.train_losses,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=2,col=1)
    fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.test_losses,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=2,col=1)
    fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_losses), max(non_grokked_object.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)



    fig.add_trace(go.Scatter(x=list(range(len(grokked_object.test_accuracies))),y=grokked_object.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=3,col=1)
    fig.add_trace(go.Scatter(x=list(range(len(grokked_object.train_accuracies))),y=grokked_object.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=3,col=1)
    fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=3, col=1)

    fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=4,col=1)
    fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=4,col=1)
    fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=4, col=1)

    if epoch==grokked_object.model_epochs()[0]:
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss", type='log',row=1, col=2)

        fig.update_xaxes(title_text="Epoch", row=1, col=3)
        fig.update_yaxes(title_text="Accuracy", row=1, col=3)
        fig.update_xaxes(title_text="Epoch", row=1, col=4)
        fig.update_yaxes(title_text="Accuracy", row=1, col=4)
    count=0
    
    showleg=True
    for key in sorted_activations_grok.keys():
        activations_index=acts_indices_grok[key][neuron_index]
        activations_for_neuron=activations_grok[key][(slice(None),)+tuple(activations_index.tolist())]

        activations_index_nogrok=acts_indices_nogrok[key][neuron_index]
        activations_for_neuron_nogrok=activations_nogrok[key][(slice(None),)+tuple(activations_index_nogrok.tolist())]
        for feature in range(feature_dim):
            fig.add_trace(go.Scatter(x=features_tensor[:,feature],y=activations_for_neuron,mode='markers',marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1+2*feature,col=count+2)
            fig.add_trace(go.Scatter(x=features_tensor[:,feature],y=activations_for_neuron_nogrok,mode='markers',marker=dict(color='blue'),showlegend=showleg,name='No grok'),row=1+2*feature+1,col=count+2)
            #fig.add_trace(go.Scatter(x=sorted_gw,y=ccdf_gw,marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1,col=count+3)
            #fig.add_trace(go.Scatter(x=sorted_ngw,y=ccdf_ngw,marker=dict(color='blue'),showlegend=showleg,name='No Grok'),row=1,col=count+3)
            showleg=False
            #fig.update_yaxes(type='log', row=1, col=count+3)
            #fig.update_yaxes(type='log', row=2, col=count+3)
            #fig.update_xaxes(type='log', row=1, col=count+3)
            #fig.update_xaxes(type='log', row=2, col=count+3)
            # fig.add_trace(go.Histogram(x=[epoch], name=f'Grok_Epoch_{epoch}'),row=1,col=i+2)  # Placeholder histogram
            # fig.add_trace(go.Histogram(x=[epoch + 0.5], name=f'NoGrok_Epoch_{epoch}'),row=2,col=i+2)  # Placeholder histogram
        if epoch==grokked_object.model_epochs()[0]:
            fig.update_xaxes(title_text="Weight", row=1, col=count+1)
            fig.update_xaxes(title_text="Weight", row=2, col=count+1)
        count+=1
    return fig

    # learning_rate=single_run.trainargs.lr
    # weight_decay=single_run.trainargs.weight_decay

class Square(nn.Module):
    ''' 
    Torch-friendly implementation of activation if one wants to use
    quadratic activations a la Gromov (induces faster grokking).
    '''
    def forward(self, x):
        return torch.square(x)

class Network(nn.Module):
    def __init__(self, hidden=[512],P=97,optimizer=torch.optim.Adam,multiplier=1):
        super(Network, self).__init__()
        layers=[]
        input_dim=2*P
        first=True
        for layer_ind in range(len(hidden)):
            if first:
                layers.append(nn.Linear(input_dim, hidden[layer_ind]))
                first=False
                #layers.append(Square())
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(hidden[layer_ind-1],hidden[layer_ind]))
                #layers.append(Square())
                layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden[-1],P))
        self.model = nn.Sequential(*layers)

        # multiply weights by overall factor
        if multiplier != 1:
            with torch.no_grad():
                for param in self.parameters():
                    param.data = multiplier * param.data
            
        # self.model = nn.Sequential(
        #     for hiddenlayer in hidden:
        #         nn.Linear(2*P, int(hiddenlayer)),
        #         Square(), # Toggle between quadratic and ReLU
        #         #nn.ReLU(),
        #         nn.Linear(hidden, P))
        self.optimizer = optimizer(params=self.parameters(),lr=learning_rate, weight_decay=weight_decay)
        
        self.init_weights()
    def forward(self, x):
        x = self.model(x)
        return x
    
    # Weight initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)               
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)


def manual_config_moddadd(run_object):
    config_dict={name:None for name,param in inspect.signature(Network.__init__).parameters.items() if name != 'self'}
    print(config_dict)
    hidden=run_object.trainargs.hiddenlayers
    P=run_object.trainargs.P
    optimizer_0=torch.optim.Adam
    multiplier=run_object.trainargs.weight_multiplier
    networkclass=Network(hidden=hidden,P=P,optimizer=optimizer_0)
    optimizer=torch.optim.Adam(params=networkclass.parameters(),lr=run_object.trainargs.lr,weight_decay=run_object.trainargs.weight_decay)

    config_dict2={}
    for key in config_dict:
        config_dict2[key]=locals()[key]
    print(config_dict2)
    return config_dict2

#Remember if adding this for the modadd that you need to load the modelclass and put learning rate and weight decay above it!
# manual_config_moddadd(single_run)
# filename=data_object_file_name
# single_run.modelconfig=manual_config_moddadd(single_run)
# single_run.modelclass=Network
# try:
#     with open(filename, "wb") as dill_file:
#         torch.save(single_run, dill_file)
# except Exception as e:
#     print(f"An error occurred during serialization: {e}")
#     print(filename)
# # save_missing_bits(data_object_file_name,single_run)
# # save_missing_bits(data_object_file_name_ng,single_run_ng)
# exit()

# exit()

if __name__== "__main__":


    torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/grok_ising/clusterdata/grok_True_time_1714759182/data_seed_0_time_1714762834_train_500_wd_0.08_lr0.0001"
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/grok_ising/modular_arithmetic/reference/cluster_run/clusterdata4/hiddenlayer_[100]_desc_avgIsing/grok_Falsedataseed_0_sgdseed_0_initseed_0_wd_0.08_wm_1_time_1715789462"

    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/clusterdata/hiddenlayer_[256]_desc_test_moadadd/grok_Truedataseed_0_sgdseed_0_initseed_0_wd_0.0003_wm_500.0_time_1717370830"
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/clusterdata/hiddenlayer_[256]_desc_test_moadadd/grok_Truedataseed_0_sgdseed_0_initseed_0_wd_0.0003_wm_1.0_time_1717371339"

    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/AFinalData/LossCurves/grok_True_standard_param.torch"#<--
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/AFinalData/LossCurves/grok_False_standard_param.torch"
    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/fclayer_runs/grok_True_time_1714254742/data_seed_0_time_1714257653_train_500_wd_0.08_lr0.0001"#<--
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/fclayer_runs/grok_False_time_1714254764/data_seed_0_time_1714257737_train_500_wd_0.05_lr0.0001"#<--

    #standard ising seed
    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/clusterdata2/grok_True_time_1715291012_hiddenlayer_[100]/data_seed_2_time_1715295790_train_500_wd_0.08_lr0.0001"
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/clusterdata2/grok_False_time_1715291028_hiddenlayer_[100]/data_seed_3_time_1715295569_train_500_wd_0.05_lr0.0001"
    
    #small ising
    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/fclayer_runs/grok_True_time_1714254742/data_seed_0_time_1714257653_train_500_wd_0.08_lr0.0001"#<--
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/fclayer_runs/grok_False_time_1714254764/data_seed_0_time_1714257737_train_500_wd_0.05_lr0.0001"#<--

    #dynamical data
    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/clusterdata3/grok_True_time_1715446487_hiddenlayer_[100]/data_seed_0_time_1715452552_train_500_wd_0.08_lr0.0001_wm_10"
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/clusterdata3/grok_False_time_1715446556_hiddenlayer_[100]/data_seed_1_time_1715452258_train_500_wd_0.05_lr0.0001_wm_1"
    
    #mult 0.1
    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/clusterdata3/grok_True_time_1715446554_hiddenlayer_[100]/data_seed_0_time_1715448460_train_500_wd_0.08_lr0.0001"
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/clusterdata3/grok_False_time_1715448558_hiddenlayer_[100]/data_seed_1_time_1715450430_train_500_wd_0.05_lr0.0001"

    #lr 10-5
    #data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/clusterdata4/grok_False_time_1715457565_hiddenlayer_[100]/data_seed_0_time_1715458561_train_500_wd_0.05_lr1e-06"
    #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/clusterdata4/grok_False_time_1715457544_hiddenlayer_[100]/data_seed_0_time_1715459206_train_500_wd_0.05_lr1e-05"
    
    #mod add exception
    data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/oppositetest/hiddenlayer_[512]_desc_opp_modadd_wm_10.0/grok_Falsedataseed_0_sgdseed_0_initseed_0_wd_3e-05_wm_10.0_time_1719661186"
    data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/oppositetest/hiddenlayer_[512]_desc_opp_modadd_wm_1.0/grok_Falsedataseed_0_sgdseed_0_initseed_0_wd_3e-05_wm_1.0_time_1719658785"


    dataset_filename="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/Data/IsingML_L16_traintest.pickle"
    
    
    #seed average folder - standard param
    #foldername_seedaverage="/Users/dmitrymanning-coe/Documents/Research/Grokking/clusterdata5/hiddenlayer_[100]_desc_avgIsingstandard"
    #seed average folder - higher weight decay.
    foldername_seedaverage="/Users/dmitrymanning-coe/Documents/Research/Grokking/clusterdata6/hiddenlayer_[100]_desc_avgIsingstandard"
    #Modadd no reg
   # data_object_file_name="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/AFinalData/ModAdd/ModAdd_512_grok_True.torch"
   #data_object_file_name_ng="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/AFinalData/ModAdd/ModAdd_512_grok_False.torch"
    #first,load the run_object
    with open(data_object_file_name, 'rb') as in_strm:
        single_run = torch.load(in_strm,map_location=device)
    with open(data_object_file_name_ng, 'rb') as in_strm:
        single_run_ng = torch.load(in_strm,map_location=device)

    # filename=data_object_file_name_ng
    # runobj=single_run_ng
    # del single_run.modelconfig['modelinstance']
    # try:
    #     with open(filename, "wb") as dill_file:
    #         torch.save(runobj, dill_file)
    # except Exception as e:
    #     print(f"An error occurred during serialization: {e}")
    #     print(filename)
    
    def save_missing_bits(filename,runobj):
        #runobj.modelclass=CNN_nobias
        # if 'modelinstance' in runobj.modelconfig.keys():
        #     del runobj.modelconfig['modelinstance']
        
        config2=manual_config_CNN(runobj)
        runobj.modelconfig=config2
        runobj.modelclass=CNN
        print(config2)
        
        try:
            with open(filename, "wb") as dill_file:
                torch.save(runobj, dill_file)
        except Exception as e:
            print(f"An error occurred during serialization: {e}")
            print(filename)
    
    
    
    # print(single_run_ng.modelconfig['modelinstance'])
    # exit()
    
    #model=single_run.modelclass(**single_run.modelconfig)
    
    # save_missing_bits(data_object_file_name_ng,single_run_ng)
    #save_missing_bits(data_object_file_name,single_run)
    # exit()
    

    #Then load the dataset
    with open(dataset_filename, "rb") as handle:
        dataset = dill.load(handle)[1]


    print(f"trainargs grok\n {vars(single_run.trainargs)}")
    print(f"trainargs no grok\n {vars(single_run_ng.trainargs)}")

    

    test=generate_test_set(dataset,1000)
    criterion=nn.CrossEntropyLoss()
    epoch=2000
    # learning_rate=single_run_ng.trainargs.lr
    # weight_decay=single_run_ng.trainargs.weight_decay
    # print(single_run.modelclass)
    # print(single_run.modelconfig)
    # single_run_ng.modelconfig['optimizer']=torch.optim.Adam
    # filename=data_object_file_name_ng
    # try:
    #     with open(filename, "wb") as dill_file:
    #         torch.save(single_run_ng, dill_file)
    # except Exception as e:
    #     print(f"An error occurred during serialization: {e}")
    #     print(filename)    

    # exit()
    #original_model_g,original_model_g_dic=load_model(single_run,epoch)

    def acc_loss_test_mod(model):
        loss_criterion=nn.MSELoss()
        testloader,test_dataset=generate_test_set_modadd(dataset=dataset,size=1000,run_object=single_run)
        with torch.no_grad():
            test_loss = 0.0
            test_accuracy = 0.0
            for batch in testloader:
                X_test,y_test=batch
                y_val=model(X_test).to(device)
                float_y=y_test.float().clone().detach()
                loss = criterion(y_val, float_y.to(device))
                test_loss += loss.item()
                loss_criterion='MSE'
                if loss_criterion=='MSE':
                    test_accuracy += (y_val.argmax(dim=1) == y_test.argmax(dim=1)).sum().item()
                else:
                    test_accuracy += (y_val.argmax(dim=1) == y_test).sum().item()
            
            test_loss /= len(testloader)
            test_accuracy /= len(test_dataset)
            
        return test_accuracy,test_loss
    
    #print(acc_loss_test(model=original_model_g))
    
    # pca_pruning(grokked_run=single_run,non_grokked_run=single_run_ng,layer_name='conv_layers.0.weight',epoch=epoch,fig=None).show()

    # pca_pruning(grokked_run=single_run,non_grokked_run=single_run_ng,layer_name='conv_layers.3.weight',epoch=epoch,fig=None).show()
    # pca_pruning(grokked_run=single_run,non_grokked_run=single_run_ng,layer_name='fc_layers.0.weight',epoch=epoch,fig=None).show()
    # layers_pruned=['fc_layers.0']
    #pruned_grok_model=prune_share(runobj=single_run,layer_name='conv_layers.3',pruning_percent=0.9)
    #pruned_nogrok_model=prune_share(runobj=single_run_ng,layer_name='conv_layers.3',pruning_percent=0.9)


    # correlation_one_epoch(grokked_object=single_run,non_grokked_object=single_run_ng,model_grok=pruned_grok_model,model_nogrok=pruned_nogrok_model,
    #                   sortby='var',epoch=99900,neuron_index=0,images_tensor=test[0],feature_funcs=features_list,dataset=test,fig=None).show()
    
    # magnitude_prune(grokked_object=single_run,non_grokked_object=single_run_ng,pruning_percents=np.linspace(0,1,200),layers_pruned=['conv_layers.0'],fig=None).show()
    # magnitude_prune(grokked_object=single_run,non_grokked_object=single_run_ng,pruning_percents=np.linspace(0,1,200),layers_pruned=['conv_layers.3'],fig=None).show()
    # magnitude_prune(grokked_object=single_run,non_grokked_object=single_run_ng,pruning_percents=np.linspace(0,1,200),layers_pruned=['fc_layers.0'],fig=None).show()
    #single_run.plot_traincurves(single_run_ng).show()
    #single_run.weights_histogram_epochs2(single_run_ng)
    #print(vars(single_run.trainargs))
    #print(single_run.params_dic)
    #magnitude_prune_prod(grokked_object=single_run,non_grokked_object=single_run_ng,pruning_percents=np.linspace(0,1,100),layers_pruned=['conv_layers.0','conv_layers.3','fc_layers.0'],fig=None,epoch=epoch).show()
    #single_run.plot_traincurves(single_run_ng).show()
    
    #foldername_seedaverage="/Users/dmitrymanning-coe/Documents/Research/Grokking/clusterdatawd0004/hiddenlayer_[100]_desc_avgIsingstandard_grokwd_0-005"
    #magnitude_prune_prod_avg(runfolder=foldername_seedaverage,pruning_percents=np.linspace(0.5,1,20),layers_pruned=['conv_layers.0','conv_layers.3','fc_layers.0'],epoch=30000,fig=None).show()
    #magnitude_prune_epochs_avg(runfolder=foldername_seedaverage,pruning_percents=np.linspace(0.2,1,30),layers_pruned=['conv_layers.0','conv_layers.3','fc_layers.0'],epochs=[18000,20000,23000,24000,25000,27000,30000,99900],fig=None).show()#[500,1000,5000,10000,20000,30000,50000,99900]
    #exit()
    # seed_dic=group_runs_to_dic(foldername_seedaverage)
    # for seed in seed_dic.keys():
    #     print(f'seed {seed}')
    #     magnitude_prune_prod(grokked_object=seed_dic[seed][0],non_grokked_object=seed_dic[seed][1],pruning_percents=np.linspace(0.5,1,25),layers_pruned=['conv_layers.0','conv_layers.3','fc_layers.0'],fig=None,epoch=epoch).show()
    
    #decorator attempt

    
    def magnitude_prune_prod_mod(grokked_object,non_grokked_object,pruning_percents,layers_pruned,fig,epoch):
        original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
        original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)
        original_weights_g = save_original_weights(original_model_g, layers_pruned)
        original_weights_ng = save_original_weights(original_model_ng, layers_pruned)


        
        # accs_losses_g_global=iterative_prune_from_original_mod(original_model_g, original_weights_g, layers_pruned, pruning_percents,'global')
        # accs_losses_ng_global=iterative_prune_from_original_mod(original_model_ng, original_weights_ng, layers_pruned, pruning_percents,'global')

        original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
        original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)

        accs_losses_g_al=iterative_prune_from_original_mod(original_model_g, original_weights_g, layers_pruned, pruning_percents,'all_layers')
        accs_losses_ng_al=iterative_prune_from_original_mod(original_model_ng, original_weights_ng, layers_pruned, pruning_percents,'all_layers')

        original_model_g, original_model_dic_g=load_model(grokked_object,epoch)
        original_model_ng, original_model_dic_ng=load_model(non_grokked_object,epoch)

        accs_losses_g=[iterative_prune_from_original_mod(original_model_g, original_weights_g, [i], pruning_percents,'local') for i in layers_pruned]
        accs_losses_ng=[iterative_prune_from_original_mod(original_model_ng, original_weights_ng, [i], pruning_percents,'local') for i in layers_pruned]

        
        #Let's put the PCA in there for comparison


        if fig==None:
            fig=make_subplots(rows=2,cols=1+len(accs_losses_g),subplot_titles=[r'$\text{(a) Pruning - whole network}$',r'$\text{(b) Pruning - first convolutional layer}$',r'$\text{(c) Pruning - second convolutional layer}$',r'$\text{(d) Pruning - fully connected layer}$',r'$\text{(e) Pruning - whole network}$',r'$\text{(f) Pruning - first convolutional layer}$',r'$\text{(g) Pruning - second convolutional layer}$',r'$\text{(h) Pruning - fully connected layer}$'])

        fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_g_al[0],marker=dict(color='red'),name='Grokking',showlegend=True),row=1,col=1)
        fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_ng_al[0],marker=dict(color='blue'),name='Learning',showlegend=True),row=1,col=1)
        fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_g_al[1],marker=dict(color='red'),name='Grokking',showlegend=False),row=2,col=1)
        fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_ng_al[1],marker=dict(color='blue'),name='Learning',showlegend=False),row=2,col=1)    

        #fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_g_global[0],marker=dict(color='red'),name='Grokking',showlegend=True),row=1,col=2)
        #fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_ng_global[0],marker=dict(color='blue'),name='Learning',showlegend=True),row=1,col=2)
        #fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_g_global[1],marker=dict(color='red'),name='Grokking',showlegend=False),row=2,col=2)
        #fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_ng_global[1],marker=dict(color='blue'),name='Learning',showlegend=False),row=2,col=2)
        for i in range(len(accs_losses_g)):
            fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_g[i][0],marker=dict(color='red'),name='Grokking',showlegend=False),row=1,col=2+i)
            fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_ng[i][0],marker=dict(color='blue'),name='Learning',showlegend=False),row=1,col=2+i)
            showleg=False
            fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_g[i][1],marker=dict(color='red'),name='Grok',showlegend=False),row=2,col=2+i)
            fig.add_trace(go.Scatter(x=pruning_percents,y=accs_losses_ng[i][1],marker=dict(color='blue'),name='Grok',showlegend=False),row=2,col=2+i)
            fig.update_xaxes(title_text=r"$\text{Percent pruned}$",row=1,col=1)
            fig.update_yaxes(title_text=r"$\text{Accuracy}$", row=1, col=1)
            fig.update_xaxes(title_text=r"$\text{Percent pruned}$", row=2, col=1)
            fig.update_yaxes(title_text=r"$\text{Loss}$",type='log', row=2, col=1)
            fig.update_xaxes(title_text=r"$\text{Percent pruned}$",row=1,col=2+i)
            fig.update_yaxes(title_text=r"$\text{Accuracy}$", row=1, col=2+i)
            fig.update_xaxes(title_text=r"$\text{Percent pruned}$", row=2, col=2+i)
            fig.update_yaxes(title_text=r"$\text{Loss}$",type='log', row=2, col=2+i)

        # fig.add_trace(go.Scatter(x=zero_g,y=acc_g,marker=dict(color='red'),name='Grok',showlegend=False),row=2,col=1)
        # fig.add_trace(go.Scatter(x=zero_ng,y=acc_ng,marker=dict(color='blue'),name='Grok',showlegend=False),row=2,col=1)
        # fig.add_trace(go.Scatter(x=zero_g,y=loss_g,marker=dict(color='red'),name='Grok',showlegend=False),row=2,col=2)
        # fig.add_trace(go.Scatter(x=zero_ng,y=loss_ng,marker=dict(color='blue'),name='Grok',showlegend=False),row=2,col=2)

        # fig.update_xaxes(title_text="Percent pruned", row=2, col=1)
        # fig.update_yaxes(title_text="Accuracy", row=2, col=1)
        # fig.update_xaxes(title_text="Percent pruned", row=2, col=2)
        # fig.update_yaxes(title_text="Loss",type='log', row=2, col=2)



        return fig
    
    from functools import wraps
    
    
    grok_foldername_seedaverage="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/oppositetest/hiddenlayer_[512]_desc_opp_modadd_wm_10.0"
    nogrok_foldername_seedaverage="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/oppositetest/hiddenlayer_[512]_desc_opp_modadd_wm_1.0"
    all_run_folder="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAdditionCluster/modaddwd_3e-4"
    #"/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/oppositetest"

    #grok_runs=open_files_in_leaf_directories(grok_foldername_seedaverage)
    # grok_run=grok_runs[1]
    # del grok_runs
    # nogrok_runs=open_files_in_leaf_directories(nogrok_foldername_seedaverage)
    # nogrok_run=nogrok_runs[1]
    # del nogrok_runs
    # grok_run.traincurves_and_iprs(nogrok_run).show()

    def open_files_in_leaf_directories(root_dir):
        all_files=[]
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Check if the current directory is a leaf directory
            if not dirnames:
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    try:
                        with open(file_path, 'rb') as in_strm:
                                single_run = torch.load(in_strm,map_location=device)
                                                # Do something with the content if needed
                                all_files.append(single_run)
                    except Exception as e:
                        print(f"Failed to open {file_path}: {e}")
        return all_files

    # test_files=open_files_in_leaf_directories(all_run_folder)
    # for file in test_files:
    #     file.traincurves_and_iprs(file).show()
    # exit()

    #Let's write a cheeky little script to seperate out the files rather than doing it manually
    import shutil
    def seperate_files(folder,target1,target2):
        
        for dirpath, dirnames, filenames in os.walk(folder):
            # Check if the current directory is a leaf directory
            if not dirnames:
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)                    
                    try:
                        with open(file_path, 'rb') as in_strm:
                                single_run = torch.load(in_strm,map_location=device)
                                if single_run.trainargs.weight_decay==3e-04:
                                    target=target1
                                elif single_run.trainargs.weight_decay==3e-06:
                                    target=target2
                                else:
                                    continue
                                target=f'/hiddenlayer_[512]_desc_modadd_wm_{single_run.trainargs.weight_multiplier}'+file_path
                                # Create all necessary subdirectories
                                os.makedirs(os.path.dirname(target), exist_ok=True)
                                shutil.move(file_path,target)
                    except Exception as e:
                        print(f"Failed to open {file_path}: {e}")
        return None

    # wd4folder="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAdditionCluster/modadd_wd_e-4"
    # wd6folder="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAdditionCluster/modadd_wd_e-6"
    # seperate_files(folder=all_run_folder,target1=wd4folder,target2=wd6folder)
    # exit()

    # all_files_grok=open_files_in_leaf_directories(all_run_folder)
    # for file in all_files_grok:
    #     if file.trainargs.weight_multiplier==5:
    #         file.traincurves_and_iprs(file).show()
    # exit()


    # def move_files_to_parent(parent_folder):
    #     for root, dirs, files in os.walk(parent_folder, topdown=False):
    #         for name in files:
    #             file_path = os.path.join(root, name)
    #             new_path = os.path.join(parent_folder, name)
    #             if os.path.exists(new_path):
    #                 base, extension = os.path.splitext(new_path)
    #                 counter = 1
    #                 while os.path.exists(new_path):
    #                     new_path = f"{base}_{counter}{extension}"
    #                     counter += 1
    #             shutil.move(file_path, new_path)
    #         for name in dirs:
    #             dir_path = os.path.join(root, name)
    #             if not os.listdir(dir_path):  # Check if the directory is empty
    #                 os.rmdir(dir_path)

    # parent_folder = all_run_folder  # Replace this with your folder path
    # move_files_to_parent(parent_folder)

   


    def prune_area():
        def inside_function(func):
            @wraps(func)
            def wrapped_function(*args,**kwargs):
                if type(kwargs['run_object'])==str:
                    print(f'started opening files')
                   
                    
                    all_files_grok=open_files_in_leaf_directories(kwargs['run_object'])
                    print('example trainargs:')
                    print(all_files_grok[0].trainargs)
                    print(f" len grok files {len(all_files_grok)}")
                    def get_areas(files):
                        wm_dic={}
                        wm_areas_dic={}
                        all_layer=[]
                        by_layer=[]
                        print(f'started averaging')
                        for file in tqdm(files,position=0,leave=True):
                            if file.trainargs.weight_multiplier not in wm_dic.keys():
                                wm_dic[file.trainargs.weight_multiplier]=([],[])
                            kwargs['run_object']=file
                            result=func(*args,**kwargs)
                            all_layer.append(result[1])
                            by_layer.append(result[2])
                            wm_dic[file.trainargs.weight_multiplier][0].append(result[1])
                            wm_dic[file.trainargs.weight_multiplier][1].append(result[2])
                            # print(result[1])
                            # print(np.array(result[1]).shape)
                            # print(np.array(result[1])[0,:])
                            # print(np.array(result[1])[1,:])
                            # print(np.array(result[1])[:,:])
                            # print(np.array(result[1])[:,:1][:,0])
                            
                            # print(f'result 1 shape: {np.array(result[1]).shape}')
                            # print(f'result 2 shape: {np.array(result[2]).shape}')
                            # print(f'result 2 shape array: {np.array(result[2])[:,:,0].shape}')
                            #print(f'all_layer first attempt accs {np.array(result[1])[:,0]}')
                            #print(f'all_layer first attempt {np.array(result[1])[:,0]}')
                            
                            # print(f'by_layer first attempt {np.array(result[2])[:,:,0]}')
                            
                        for key in wm_dic.keys():
                            all_layer_areas=np.trapz(np.array(wm_dic[key][0]),x=kwargs['pruning_percents'])
                            by_layer_areas=np.trapz(np.array(wm_dic[key][1]),x=kwargs['pruning_percents'])
                            #all_layer_first=np.array(np.array(wm_dic[key][0])[:,:,:1])
                            all_layer_first=np.array(wm_dic[key][0])[:,:,:1]
                            all_layer_first=all_layer_first.reshape(len(all_layer_first),2)
                            #print(all_layer_first.reshape(len(all_layer_first),2))

                            #print(f'all layer first attempt accs {all_layer_first}')
                            
                            by_layer_first=np.array(np.array(wm_dic[key][1])[:,0,:])
                            
                            
                            
                            
                            
                            
                            wm_areas_dic[key]=(all_layer_areas,by_layer_areas,all_layer_first,by_layer_first)
                        
                        
                        #print(f'all layer shape {all_layer.shape}')
                        #print(all_layer)
                        #print('by layer')
                        #print(f'by_layer shape {by_layer.shape}')
                        
                        return wm_areas_dic
                    
                    grok_areas=get_areas(all_files_grok)
                    
                    
                    print(f'function output (percent pruned, grok_all_layer)')
                    return grok_areas
                else:
                    return func(*args,**kwargs)
                #If you want the args you can just call args!
                # print(f' args: {args}')
                # print(f' args: {kwargs}')
                # return func(*args,**kwargs)
            return wrapped_function
        return inside_function

    def plot_dec_areas(plot):
        def inside_function(func):
            @wraps(func)
            def wrapped_function(*args,**kwargs):
                if plot:
                    fig1=make_subplots(rows=2,cols=2,specs=[[{"secondary_y": True} for _ in range(2)] for _ in range(2)],subplot_titles=[r'$\text{(a) Pruning whole network - area under accuracy curve (average) }$',r'$\text{(a) Pruning whole network - area under accuracy curve (individual) }$',r'$\text{(a) Pruning whole network - area under loss curve (average)}$',r'$\text{(a) Pruning whole network - area under loss curve (individual) }$'])
                    areas=func(*args,**kwargs)
                    all_layer_area_accs_arrays=[]
                    all_layer_area_losses_arrays=[]
                    by_layer_area_accs_arrays=[]
                    by_layer_area_losses_arrays=[]
                    all_layer_first_acc_arrays=[]
                    all_layer_first_loss_arrays=[]
                    by_layer_first_acc_arrays=[]
                    by_layer_first_loss_arrays=[]
                    multipliers=[]
                    avg_multpliers=[]
                    for key in areas.keys():
                        all_layer_area_accs_arrays.append(areas[key][0][:,0])
                        all_layer_area_losses_arrays.append(areas[key][0][:,1])
                        by_layer_area_accs_arrays.append(areas[key][1][:,0,:])
                        by_layer_area_losses_arrays.append(areas[key][1][:,1,:])
                        multipliers.append(np.array([key]*len(areas[key][0][:,0])))
                        avg_multpliers.append(key)
                        all_layer_first_acc_arrays.append(areas[key][2][:,0])
                        all_layer_first_loss_arrays.append(areas[key][2][:,1])
                        #by_layer_first_arrays.append(areas[key][3])
                    def flatten_array(arr):
                        flattened = []
                        for item in arr:
                            if isinstance(item, np.ndarray):
                                flattened.extend(flatten_array(item))
                            else:
                                flattened.append(item)
                        return flattened
                    
                    plot_all_layers_accs=np.concatenate([flatten_array(arr) for arr in all_layer_area_accs_arrays])
                    plot_all_layers_accs_average=np.array([np.mean(arr) for arr in all_layer_area_accs_arrays])
                    plot_all_layers_losses=np.concatenate([flatten_array(arr) for arr in all_layer_area_losses_arrays])
                    plot_all_layers_losses_average=np.array([np.mean(arr) for arr in all_layer_area_losses_arrays])
                    plot_multiplier=np.concatenate([flatten_array(arr) for arr in multipliers])
                    plot_all_layer_acc_first=np.concatenate([flatten_array(arr) for arr in all_layer_first_acc_arrays])
                    plot_all_layer_acc_first_average=np.array([np.mean(arr) for arr in all_layer_first_acc_arrays])
                    plot_all_layer_loss_first=np.concatenate([flatten_array(arr) for arr in all_layer_first_loss_arrays])
                    plot_all_layer_loss_first_average=np.array([np.mean(arr) for arr in all_layer_first_loss_arrays])

                    #plot_by_layer_first=np.concatenate([flatten_array(arr) for arr in all_layer_first])
                    
                    #fig1.add_trace(go.Scatter(x=np.array(multipliers).ravel(),y=np.array(all_layer_area_accs_arrays).ravel(),mode='markers',marker=dict(color='red'),name='Accuracy areas',showlegend=True),row=1,col=1)
                    
                    fig1.add_trace(go.Scatter(x=avg_multpliers,y=plot_all_layers_accs_average,mode='markers',marker=dict(color='red',symbol='cross'),name='Accuracy areas - average',showlegend=True),row=1,col=1)
                    fig1.add_trace(go.Scatter(x=avg_multpliers,y=plot_all_layer_acc_first_average,mode='markers',marker=dict(color='black',symbol='diamond'),name='Start accuracy - average',showlegend=True),row=1,col=1,secondary_y=True)
                    #fig1.add_trace(go.Scatter(x=avg_multpliers,y=plot_all_layers_accs_average,mode='markers',marker=dict(color='black',symbol='diamond'),name='End accuracy - average',showlegend=True),row=1,col=1,secondary_y=True)
                    
                    fig1.add_trace(go.Scatter(x=plot_multiplier,y=plot_all_layers_accs,mode='markers',marker=dict(color='red'),name='Accuracy areas',showlegend=True),row=1,col=2)
                    #fig1.add_trace(go.Scatter(x=avg_multpliers,y=plot_all_layer_acc_first,mode='markers',marker=dict(color='black',symbol='diamond'),name='Start accuracy',showlegend=True),row=1,col=2,secondary_y=True)
                    fig1.add_trace(go.Scatter(x=plot_multiplier,y=plot_all_layer_acc_first,mode='markers',marker=dict(color='black',symbol='diamond'),name='Start accuracy',showlegend=True),row=2,col=2,secondary_y=True)

                    fig1.add_trace(go.Scatter(x=avg_multpliers,y=plot_all_layers_losses_average,mode='markers',marker=dict(color='blue',symbol='cross'),name='Accuracy areas - average',showlegend=True),row=2,col=1)
                    fig1.add_trace(go.Scatter(x=avg_multpliers,y=plot_all_layer_acc_first_average,mode='markers',marker=dict(color='gold',symbol='diamond'),name='Start accuracy - average',showlegend=True),row=2,col=1,secondary_y=True)

                    fig1.add_trace(go.Scatter(x=plot_multiplier,y=plot_all_layers_losses,mode='markers',marker=dict(color='blue'),name='Losses areas',showlegend=True),row=2,col=2)
                    fig1.add_trace(go.Scatter(x=plot_multiplier,y=plot_all_layer_acc_first,mode='markers',marker=dict(color='gold',symbol='diamond'),name='Start accuracy',showlegend=True),row=2,col=2,secondary_y=True)
                    
                    fig1.update_yaxes(secondary_y=True,range=[0, 1.1])  # Replace min_value and max_value with your desired valuesrow=1,
                    #fig1.update_yaxes(secondary_y=True,range=[0, 1.1],row=1,col=2)
                    #fig1.update_yaxes(secondary_y=True,range=[min(plot_all_layer_loss_first_average), max(plot_all_layer_loss_first_average)],row=2,col=1)
                    #fig1.update_yaxes(secondary_y=True,range=[min(plot_all_layer_loss_first), max(plot_all_layer_loss_first)],row=2,col=2)

                    #fig1.add_trace(go.Scatter(x=np.array([[1,1],[2,2]]).ravel(),y=np.array([[2,2],[3,3]]).ravel(),mode='markers',marker=dict(color='blue'),name='Test',showlegend=True),row=2,col=1)
                    return fig1
                    
                    # print(f'areas[0] shape {areas[0].shape}')
                    # print(f'areas[1] shape {areas[1].shape}')


                    # print(f'all layer area accs shape {all_layer_area_accs.shape}')
                    # print(all_layer_area_accs)

                    # print(f'all layer area losses shape {all_layer_area_losses.shape}')
                    # print(all_layer_area_losses)

                    # print(f'by layer area accs shape {by_layer_area_accs.shape}')
                    # print(by_layer_area_accs)

                    # print(f'by layer area losses shape {by_layer_area_losses.shape}')
                    # print(by_layer_area_losses)

                    




                    
                    #print(f"grok averages 1 shape {grok_avgs[1][:,:,0]}")
                    #exit()


                    nogrok_accs_all=non_grok_avgs[0][0]
                    nogrok_losses_all=non_grok_avgs[0][1]
                    nogrok_accs_bl=non_grok_avgs[1][:,0,:]
                    nogrok_losses_bl=non_grok_avgs[1][:,1,:]
                    

                    
                    fig=make_subplots(rows=2,cols=1+len(grok_losses_bl),subplot_titles=[r'$\text{(a) Pruning - whole network}$',r'$\text{(b) Pruning - first convolutional layer}$',r'$\text{(c) Pruning - second convolutional layer}$',r'$\text{(d) Pruning - fully connected layer}$',r'$\text{(e) Pruning - whole network}$',r'$\text{(f) Pruning - first convolutional layer}$',r'$\text{(g) Pruning - second convolutional layer}$',r'$\text{(h) Pruning - fully connected layer}$'])

                    fig.add_trace(go.Scatter(x=percents,y=grok_accs_all,marker=dict(color='red'),name='Grokking',showlegend=True),row=1,col=1)
                    fig.add_trace(go.Scatter(x=percents,y=nogrok_accs_all,marker=dict(color='blue'),name='Learning',showlegend=True),row=1,col=1)
                    fig.add_trace(go.Scatter(x=percents,y=grok_losses_all,marker=dict(color='red'),name='Grokking',showlegend=False),row=2,col=1)
                    fig.add_trace(go.Scatter(x=percents,y=nogrok_losses_all,marker=dict(color='blue'),name='Learning',showlegend=False),row=2,col=1)    

            
                    for i in range(len(grok_losses_bl)):
                        fig.add_trace(go.Scatter(x=percents,y=grok_accs_bl[i],marker=dict(color='red'),name='Grokking',showlegend=False),row=1,col=2+i)
                        fig.add_trace(go.Scatter(x=percents,y=nogrok_accs_bl[i],marker=dict(color='blue'),name='Learning',showlegend=False),row=1,col=2+i)
                        showleg=False
                        fig.add_trace(go.Scatter(x=percents,y=grok_losses_bl[i],marker=dict(color='red'),name='Grok',showlegend=False),row=2,col=2+i)
                        fig.add_trace(go.Scatter(x=percents,y=nogrok_losses_bl[i],marker=dict(color='blue'),name='Grok',showlegend=False),row=2,col=2+i)
                        fig.update_xaxes(title_text=r"$\text{Percent pruned}$",row=1,col=1)
                        fig.update_yaxes(title_text=r"$\text{Accuracy}$", row=1, col=1)
                        fig.update_xaxes(title_text=r"$\text{Percent pruned}$", row=2, col=1)
                        fig.update_yaxes(title_text=r"$\text{Loss}$",type='log', row=2, col=1)
                        fig.update_xaxes(title_text=r"$\text{Percent pruned}$",row=1,col=2+i)
                        fig.update_yaxes(title_text=r"$\text{Accuracy}$", row=1, col=2+i)
                        fig.update_xaxes(title_text=r"$\text{Percent pruned}$", row=2, col=2+i)
                        fig.update_yaxes(title_text=r"$\text{Loss}$",type='log', row=2, col=2+i)

                    fig.update_layout(title_text=f"Wm grokked {grok_foldername_seedaverage.split('wm_')[-1]},WM Learn {nogrok_foldername_seedaverage.split('wm_')[-1]} ")                        
                    #fig.show()
                    return fig
                
                
                
                #If you want the args you can just call args!
                # print(f' args: {args}')
                # print(f' args: {kwargs}')
                # return func(*args,**kwargs)
            return wrapped_function
        return inside_function

    def avg_decorator(non_grokked_folder):
        def inside_function(func):
            @wraps(func)
            def wrapped_function(*args,**kwargs):
                if type(kwargs['run_object'])==str:
                    all_files_grok=open_files_in_leaf_directories(kwargs['run_object'])
                    all_files_nogrok=open_files_in_leaf_directories(non_grokked_folder)
                    print(f" len grok files {len(all_files_grok)}")
                    print(f" len no grok files {len(all_files_nogrok)}")
                    
                    def get_averages(files):
                        all_layer=[]
                        by_layer=[]
                        for file in files:
                            kwargs['run_object']=file
                            result=func(*args,**kwargs)
                            all_layer.append(result[1])
                            by_layer.append(result[2])
                        all_layer=np.mean(np.array(all_layer),axis=0)
                        by_layer=np.mean(np.array(by_layer),axis=0)
                        return all_layer,by_layer
                    
                    grok_avgs=get_averages(all_files_grok)
                    nogrok_avgs=get_averages(all_files_nogrok)
                    

                    
                    print(f'function output (percent pruned, grok_all_layer)')
                    return kwargs['pruning_percents'],grok_avgs,nogrok_avgs
                else:
                    return func(*args,**kwargs)
                #If you want the args you can just call args!
                # print(f' args: {args}')
                # print(f' args: {kwargs}')
                # return func(*args,**kwargs)
            return wrapped_function
        return inside_function
        

    def plot_dec(plot):
        def inside_function(func):
            @wraps(func)
            def wrapped_function(*args,**kwargs):
                if plot:
                    fig=make_subplots(rows=2,cols=3)
                    percents,grok_avgs,non_grok_avgs=func(*args,**kwargs)
                    
                    
                    print(f'grok_avgs {grok_avgs[0].shape}')
                    print(f'grok_avgs {grok_avgs[1].shape}')

                    grok_accs_all=grok_avgs[0][0]
                    grok_losses_all=grok_avgs[0][1]
                    grok_accs_bl=grok_avgs[1][:,0,:]
                    grok_losses_bl=grok_avgs[1][:,1,:]
                    
                    #print(f"grok averages 1 shape {grok_avgs[1][:,:,0]}")
                    #exit()


                    nogrok_accs_all=non_grok_avgs[0][0]
                    nogrok_losses_all=non_grok_avgs[0][1]
                    nogrok_accs_bl=non_grok_avgs[1][:,0,:]
                    nogrok_losses_bl=non_grok_avgs[1][:,1,:]
                    

                    
                    fig=make_subplots(rows=2,cols=1+len(grok_losses_bl),subplot_titles=[r'$\text{(a) Pruning - whole network}$',r'$\text{(b) Pruning - first convolutional layer}$',r'$\text{(c) Pruning - second convolutional layer}$',r'$\text{(d) Pruning - fully connected layer}$',r'$\text{(e) Pruning - whole network}$',r'$\text{(f) Pruning - first convolutional layer}$',r'$\text{(g) Pruning - second convolutional layer}$',r'$\text{(h) Pruning - fully connected layer}$'])

                    fig.add_trace(go.Scatter(x=percents,y=grok_accs_all,marker=dict(color='red'),name='Grokking',showlegend=True),row=1,col=1)
                    fig.add_trace(go.Scatter(x=percents,y=nogrok_accs_all,marker=dict(color='blue'),name='Learning',showlegend=True),row=1,col=1)
                    fig.add_trace(go.Scatter(x=percents,y=grok_losses_all,marker=dict(color='red'),name='Grokking',showlegend=False),row=2,col=1)
                    fig.add_trace(go.Scatter(x=percents,y=nogrok_losses_all,marker=dict(color='blue'),name='Learning',showlegend=False),row=2,col=1)    

            
                    for i in range(len(grok_losses_bl)):
                        fig.add_trace(go.Scatter(x=percents,y=grok_accs_bl[i],marker=dict(color='red'),name='Grokking',showlegend=False),row=1,col=2+i)
                        fig.add_trace(go.Scatter(x=percents,y=nogrok_accs_bl[i],marker=dict(color='blue'),name='Learning',showlegend=False),row=1,col=2+i)
                        showleg=False
                        fig.add_trace(go.Scatter(x=percents,y=grok_losses_bl[i],marker=dict(color='red'),name='Grok',showlegend=False),row=2,col=2+i)
                        fig.add_trace(go.Scatter(x=percents,y=nogrok_losses_bl[i],marker=dict(color='blue'),name='Grok',showlegend=False),row=2,col=2+i)
                        fig.update_xaxes(title_text=r"$\text{Percent pruned}$",row=1,col=1)
                        fig.update_yaxes(title_text=r"$\text{Accuracy}$", row=1, col=1)
                        fig.update_xaxes(title_text=r"$\text{Percent pruned}$", row=2, col=1)
                        fig.update_yaxes(title_text=r"$\text{Loss}$",type='log', row=2, col=1)
                        fig.update_xaxes(title_text=r"$\text{Percent pruned}$",row=1,col=2+i)
                        fig.update_yaxes(title_text=r"$\text{Accuracy}$", row=1, col=2+i)
                        fig.update_xaxes(title_text=r"$\text{Percent pruned}$", row=2, col=2+i)
                        fig.update_yaxes(title_text=r"$\text{Loss}$",type='log', row=2, col=2+i)

                    fig.update_layout(title_text=f"Wm grokked {grok_foldername_seedaverage.split('wm_')[-1]},WM Learn {nogrok_foldername_seedaverage.split('wm_')[-1]} ")                        
                    fig.show()
                    return fig
                
                
                
                #If you want the args you can just call args!
                # print(f' args: {args}')
                # print(f' args: {kwargs}')
                # return func(*args,**kwargs)
            return wrapped_function
        return inside_function

                
    @plot_dec(True)
    @avg_decorator(non_grokked_folder=nogrok_foldername_seedaverage)
    def magnitude_prune_prod_mod_avg2(run_object,pruning_percents,layers_pruned,epoch):
        original_model, original_model_dic=load_model(run_object,epoch)
        original_weights = save_original_weights(original_model, layers_pruned)
        

        original_model, original_model_dic=load_model(run_object,epoch)

        #accuracies and losses for all layers
        accs_losses_alllayers=iterative_prune_from_original_mod(original_model, original_weights, layers_pruned, pruning_percents,'all_layers')
        #reset model
        original_model, original_model_dic=load_model(run_object,epoch)
        #accuracies and losses for individual layers
        accs_losses_bylayer=[iterative_prune_from_original_mod(original_model, original_weights, [i], pruning_percents,'local') for i in layers_pruned]
        

        return (pruning_percents,accs_losses_alllayers,accs_losses_bylayer)
    
    @plot_dec_areas(True)
    @prune_area()
    def magnitude_prune_prod_mod_avg_areas(run_object,pruning_percents,layers_pruned,epoch):
        original_model, original_model_dic=load_model(run_object,epoch)
        original_weights = save_original_weights(original_model, layers_pruned)
        

        original_model, original_model_dic=load_model(run_object,epoch)

        #accuracies and losses for all layers
        accs_losses_alllayers=iterative_prune_from_original_mod(original_model, original_weights, layers_pruned, pruning_percents,'all_layers')
        #reset model
        original_model, original_model_dic=load_model(run_object,epoch)
        #accuracies and losses for individual layers
        accs_losses_bylayer=[iterative_prune_from_original_mod(original_model, original_weights, [i], pruning_percents,'local') for i in layers_pruned]
        

        return (pruning_percents,accs_losses_alllayers,accs_losses_bylayer)

    
    print('func result')
    #percents,grok_avgs,non_grok_avgs=magnitude_prune_prod_mod_avg2(run_object=grok_foldername_seedaverage,pruning_percents=np.linspace(0,1,50),layers_pruned=['model.0','model.2'],epoch=epoch)
    #percents,grok_avgs,non_grok_avgs=magnitude_prune_prod_mod_avg2(run_object=grok_foldername_seedaverage,pruning_percents=np.linspace(0,1,5),layers_pruned=['model.0','model.2'],epoch=epoch)
    
    area_plot=magnitude_prune_prod_mod_avg_areas(run_object=all_run_folder,pruning_percents=np.linspace(0,1,20),layers_pruned=['model.0','model.2'],epoch=epoch)
    area_plot.show()
    exit()
    single_run.traincurves_and_iprs(single_run_ng).show()
    
    magnitude_prune_prod_mod(grokked_object=single_run,non_grokked_object=single_run_ng,pruning_percents=np.linspace(0,1,100),layers_pruned=['model.0','model.2'],fig=None,epoch=epoch).show()
    exit()


# print(acc_loss_test(pruned_grok_model))
# print(acc_loss_test(pruned_nogrok_model))

# def save_missing_bits(filename,runobj):
#     runobj.modelclass=CNN
#     config2=manual_config_CNN(runobj)
#     runobj.modelconfig=config2
#     try:
#         with open(filename, "wb") as dill_file:
#             torch.save(runobj, dill_file)
#     except Exception as e:
#         print(f"An error occurred during serialization: {e}")
#         print(filename)

# save_missing_bits(data_object_file_name,single_run)
# save_missing_bits(data_object_file_name_ng,single_run_ng)
# exit()

#criterion=nn.CrossEntropyLoss()
#features_ten=construct_features_tensor(images_tensor=test[0],feature_funcs=[functools.partial(compute_energy_torch_batch,J=1),magnetization])
#non_grokked_object,sortby,neuron_index,images_tensor,feature_funcs,dataset
#single_run.correlation_epochs(non_grokked_object=single_run_ng,sortby='var',neuron_index=0,images_tensor=test[0],feature_funcs=[functools.partial(compute_energy_torch_batch,J=1),magnetization],dataset=test)














# def single_neuron_vec(image_activations_dist):
#     neuron_vector=[]
#     average_activation=[]
#     variance=[]
#     energy=[]
#     # Quantities calculated over all images
#     for layer in range(len(image_activations_dist)):
#         average_activation.append(sum(image_activations_dist,axis=0)/image_activations_dist.shape[0])
#         variance.append(np.sum(np.square(image_activations_dist),axis=0)/len(image_activations_dist.shape[0])-np.square(average_activation))
#     return

# def image_values(images,func):
#     image_quantity=[]
#     for i in range(images.shape[0]):#Assumes axis 0 is the image axis
#         value=func(images[i,:,:])
#         image_quantity.append(value)
#     return np.array(image_quantity)

# list_of_image_functions=[compute_energy,get_ccs]
# #Let's form an array of image values:

# # energies=[compute_energy(i) for i in X_test]
# # largest_cc=[get_ccs(i) for i in X_test]


# print(f"grok_folder: {model_folder}")
# #epind=int(19900/100)
# epind=-1
# test2,epoch=get_model_neuron_activations(epochindex=epind,pathtodata=model_folder,image_number='var',image_set=example_images)#Note that this gives me average ordered
# testdist,epoch=get_model_neuron_activations(epochindex=epind,pathtodata=model_folder,image_number='dist',image_set=example_images)


# def visualize_images(image_list,image_indices,image_activations,nlargest):
#     rowcount=10
#     depthcount=3
#     image_positions=[int(x) for x in np.linspace(depthcount,len(image_list)-1,rowcount)]
#     image_indices=[[image_indices[x-i] for i in range(depthcount)] for x in image_positions]
#     labels=[y_test[x] for x in image_positions]
#     #print(image_indices)
#     image_acts=[image_activations[x] for x in image_indices]
#     #print(image_acts)
#     #Now just visualize the images

#     fig, ax = plt.subplots(depthcount,rowcount,figsize=(20,5))
    
#     for j in range(depthcount):
#         for i in range(rowcount):
#             a=round(image_acts[i][j],2)
#             ax[j,i].spy(X_test[image_indices[i][j]] + 1 )
#             energy=compute_energy(X_test[image_indices[i][j]])
#             energy=round(energy.item(),3)
#             ax[j,i].set_title(f'Act: {str(a)}')#, , l{y_test[image_indices[i]]},#f'Act: {str(a)}'
#             ax[j,i].set_xlabel(f'T-Tc: {str(np.round((my_y_temp[image_indices[i][j]]-2.69).numpy(),2))}, En: {energy}')#mag: {sum(torch.flatten(X_test[image_indices[i]]))}
#             ax[j,i].set_yticklabels([])
#             ax[j,i].set_xticklabels([])
#             if nlargest<0:
#                 number=-nlargest
#                 text='th most activated neuron'
#             else:
#                 number=nlargest+1
#                 text='th least activated neuron'
#     fig.suptitle(f'{number}{text}')
#     fig.tight_layout()
#     fig.savefig(str(root)+f'/{number}{text}_epoch_{str(int(epind*20))}_quiltgraphs')
#     fig.subplots_adjust(top=0.88)


#     #plt.spy(configs[50]+1)
#     #plt.show()

# def visualize_image_values(image_values,image_activations,func_labels,nlargest):
#     fig, ax=plt.subplots(1,len(image_values),figsize=(25,10))
    
#     if len(image_values)==1:
#         ax.scatter(image_values,image_activations)
#         ax.set_title('Something')
#         ax.set_xlabel('Values')
#         ax.set_ylabel('Activations')
#     else:
#         for i in range(len(image_values)):
#             ax[i].scatter(image_values[i],image_activations)
#             ax[i].set_title(func_labels[i])
#             ax[i].set_xlabel(func_labels[i])
#             ax[i].set_ylabel('Activations')
#     if nlargest<0:
#         number=-nlargest
#         text='th most activated neuron'
#     else:
#         number=nlargest+1
#         text='th least activated neuron'
#     fig.suptitle(f'{number}{text}'+' Image-Activation graphs')
#     fig.tight_layout()
#     fig.savefig(str(root)+f'/{number}{text}_epoch_{str(int(epind*20))}_corrgraphs')
#     #plt.show()


#     return None

# testn,testl=get_neuron(neuron_activation_list=test2,layer=2,nlargest=-1)
# def image_graphs(funcs,image_set,image_activations,image_indices,func_labels,nlargest):
#     all_image_vals=[]
#     new_acts=[image_activations[image_indices[i]]for i in range(len(image_indices))]
#     for func in funcs:
#         image_vals=[func(image_set[image_indices[i]]) for i in range(len(image_indices))]
#         all_image_vals.append(image_vals)
#     viz_all=visualize_image_values(all_image_vals,new_acts,func_labels,nlargest=nlargest)
#     return image_vals

# #testv1=visualize_images(image_list=X_test,image_indices=testg,image_activations=testi,nlargest=-1)
# # testv2=visualize_images(image_list=X_test,image_indices=testg,image_activations=testi)


# np.array(X_test[2].detach().cpu().numpy())

# def get_quilt(image_activations,images,neuron_index,layer,global_funcs,func_labels):
#     testn,testl=get_neuron(neuron_activation_list=test2,layer=layer,nlargest=neuron_index)
#     print(testn)
#     image_activations,image_indices=get_image_indices(image_activations=image_activations,images=images,neuron_index=testn,layer=layer)
#     viz=visualize_images(image_list=images,image_indices=image_indices,image_activations=image_activations,nlargest=neuron_index)
#     viz_all=image_graphs(funcs=global_funcs,image_set=images,image_activations=image_activations,image_indices=image_indices,func_labels=func_labels,nlargest=neuron_index)



# get_quilt(image_activations=testdist,images=X_test,neuron_index=-1,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
# #get_quilt(image_activations=testdist,images=X_test,neuron_index=-3,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
# #get_quilt(image_activations=testdist,images=X_test,neuron_index=-4,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
# #get_quilt(image_activations=testdist,images=X_test,neuron_index=-5,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])


# get_quilt(image_activations=testdist,images=X_test,neuron_index=-2,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
# get_quilt(image_activations=testdist,images=X_test,neuron_index=-3,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
# get_quilt(image_activations=testdist,images=X_test,neuron_index=-4,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
# get_quilt(image_activations=testdist,images=X_test,neuron_index=-5,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
# get_quilt(image_activations=testdist,images=X_test,neuron_index=-10,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
# #get_quilt(image_activations=testdist,images=X_test,neuron_index=-20,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])
# #get_quilt(image_activations=testdist,images=X_test,neuron_index=0,layer=2,global_funcs=[compute_energy,get_ccs,magnetization],func_labels=['Energy','Largest Connected component','Magnetization'])

# #Save user defined variables
# import dill



# # Function to filter user-defined variables
# def filter_user_defined_variables(globals_dict):
#     user_defined_vars = {}
#     for name, value in globals_dict.items():
#         # Check if the variable is not a built-in or imported module/function
#         if not name.startswith('_') and not hasattr(value, '__module__'):
#             user_defined_vars[name] = value
#     return user_defined_vars

# # Filter the user-defined variables
# user_vars = filter_user_defined_variables(globals())

# # Save only user-defined variablese
# filename = str(root)+'/variables.p'
# with open(filename, 'wb') as file:
#     dill.dump(user_vars, file)
