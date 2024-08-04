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
#had to copy and paste data_objects because couldn't figure out foldering...
# from cluster import data_objects
# from cluster.data_objects import seed_average_onerun

from cluster.cluster_run_average import TrainArgs
from cluster.cluster_run_average import CNN
from cluster.cluster_run_average import CNN_nobias
from contextlib import contextmanager
import functools
from cluster.cluster_run_average import ModularArithmeticDataset
from cluster.cluster_run_average import Square, MLP
import glob
from Analysis.activations_Ising import load_model
from Analysis.activations_Ising import generate_test_set_modadd



def _generate_next_training_batch(trainloader_iter,trainloader):
    try:
        inputs,labels = next(trainloader_iter)
        
    except StopIteration:
        trainloader_iter = iter(trainloader)
        inputs,labels = next(trainloader_iter)
        
        
    return inputs, labels

def compute_energy(trainloader,model):
    # this is nL_n,k, sum of the losses at w^* found so far
    energies = []
    with torch.no_grad():
        for batch in trainloader:
            inputs, labels = batch
            outputs=model(inputs).to(device)
            labels=labels.float().to(device)
            outputs = model(inputs)
            criterion2=nn.CrossEntropyLoss(reduction="sum")
            energies.append(criterion2(outputs,labels).item())
    return torch.sum(torch.tensor(energies)).item()


def compute_local_free_energy(runobject,epoch,criterion=nn.CrossEntropyLoss(),num_iter=100, num_chains=1,gamma=None, epsilon=1e-5, verbose=False):
    all_inputs=[]
    all_labels=[]
    model=load_model(runobject,epoch)[0]
    data_loader=runobject.test_loader
    for batch_data, batch_labels in iter(data_loader):
            all_inputs.append(batch_data)
            all_labels.append(batch_labels)
    all_inputs = torch.cat(all_inputs).to(device)
    all_labels = torch.cat(all_labels).to(device)
    
    
    gamma_dict = {}
    total_train=len(data_loader.dataset)
    if gamma is None:
        with torch.no_grad():
            model_copy = copy.deepcopy(model)
            for name, param in model_copy.named_parameters():
                gamma_val = 100.0 / np.linalg.norm(param)
                gamma_dict[name] = gamma_val
    with torch.no_grad():
        loss_fn_noreduce = nn.CrossEntropyLoss(reduction="none")
        m = 0
        loss_sum = np.zeros(len(all_inputs))
        loss_sum_sq = np.zeros(len(all_inputs))
    chain_Lms = []
    for chain in range(num_chains):
        model_copy = copy.deepcopy(model)
        og_params = copy.deepcopy(dict(model_copy.named_parameters()))
        Lms = []
        list_loader=[]
        for batch in data_loader:
            inputs, labels = batch
            labels=labels.float().to(device)
            inputs=inputs.to(device)
            list_loader.append((inputs,labels))
        for counter in range(num_iter):
            with torch.no_grad():
                m += 1
                outputs = model_copy(all_inputs.to(device))
                losses = loss_fn_noreduce(outputs, all_labels.float().to(device)).detach().cpu().numpy()
                loss_sum += losses
                loss_sum_sq += losses * losses
            iter_trainloader=iter(data_loader)
            with torch.enable_grad():#Will come back to this but why is this not parallelized?
                # call a minibatch loss backward
                # so that we have gradient of average minibatch loss with respect to w'
                # inputs, labels = list_loader[counter]
                inputs, labels = _generate_next_training_batch(iter_trainloader,data_loader)
                labels=labels.float().to(device)
                inputs=inputs.to(device)
                outputs = model_copy(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
            
            for name, w in model_copy.named_parameters():
                w_og = og_params[name]
                dw = -w.grad.data / np.log(total_train) * total_train
                if gamma is None:
                    prior_weight = gamma_dict[name]
                else:
                    prior_weight = gamma
                dw.add_(w.data - w_og.data, alpha=-prior_weight)
                w.data.add_(dw, alpha=epsilon / 2)
                gaussian_noise = torch.empty_like(w)
                gaussian_noise.normal_()
                w.data.add_(gaussian_noise, alpha=np.sqrt(epsilon))
                w.grad.zero_()
            Lms.append(loss.item())
        chain_Lms.append(Lms)
        if verbose:
            print(f"Chain {chain + 1}: L_m = {np.mean(Lms)}")

    chain_Lms = np.array(chain_Lms)
    local_free_energy = total_train * np.mean(chain_Lms)
    chain_std = np.std(total_train * np.mean(chain_Lms, axis=1))

    variance = (loss_sum_sq - loss_sum * loss_sum / m) / (m - 1)
    func_var = float(np.sum(variance))
    energy=compute_energy(data_loader,model)
    hatlambda = (local_free_energy - energy) / np.log(total_train)
    
    if verbose:
        print(
            f"LFE: {local_free_energy} (std: {chain_std}, n_chain={num_chains})\n Energy: {energy},hatlambda: {hatlambda},func_var: {func_var}"

        )
    return local_free_energy,energy, hatlambda,chain_std, func_var


#OK that's annoying - the "energy" is just the sum of the losses across the training dataset


def assemble_bits(trainloader):
    all_inputs=[]
    all_labels=[]
    
    for batch_data, batch_labels in iter(trainloader):
            all_inputs.append(batch_data)
            all_labels.append(batch_labels)
    all_inputs = torch.cat(all_inputs).to(device)
    all_labels = torch.cat(all_labels).to(device)
    return all_inputs,all_labels
    
if __name__=="__main__":
    print('herro')
    test_filepath="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAdditionCluster/test/modaddwd_3e-4/hiddenlayer_[512]_desc_modadd_wm_10.0/grok_Falsedataseed_2_sgdseed_2_initseed_2_wd_0.0003_wm_10.0_time_1721762970"
    test_nongrok="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAdditionCluster/test/modaddwd_3e-4/hiddenlayer_[512]_desc_modadd_wm_0.5/grok_Falsedataseed_2_sgdseed_2_initseed_2_wd_0.0003_wm_0.5_time_1721761777"
    test_epoch=2000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    with open(test_filepath, 'rb') as in_strm:
        test_grok = torch.load(in_strm,map_location=device)
        if hasattr(test_grok,'l2norms'):
            test_grok.norms=test_grok.l2norms
    
    with open(test_nongrok, 'rb') as in_strm:
        test_nongrok = torch.load(in_strm,map_location=device)
        if hasattr(test_nongrok,'l2norms'):
            test_nongrok.norms=test_nongrok.l2norms



    test_model_grok=load_model(test_grok,test_epoch)[0]
    test_model_nongrok=load_model(test_nongrok,test_epoch)[0]
    test_dataloader,mod_dataset=generate_test_set_modadd(dataset=None,size=None,run_object=test_grok)
    all_inputs,all_labels=assemble_bits(test_dataloader)
    
    

    #local_free_energy_grok=compute_local_free_energy(all_inputs,all_labels,test_model_grok,nn.CrossEntropyLoss(),test_grok.test_loader)
    #local_free_energy_nogrok=compute_local_free_energy(all_inputs,all_labels,test_model_nongrok,nn.CrossEntropyLoss(),test_nongrok.test_loader)
    def save(initial_name,object):
        import datetime
        root_folder="/Users/dmitrymanning-coe/Documents/Research/Grokking/ModAddition/large_files/saved_data"
        filename=f'{initial_name}_{datetime.date.today().day}-{datetime.date.today().month}-{datetime.datetime.now().year}_min_{datetime.datetime.now().minute}.pickle'
        filename=os.path.join(root_folder,filename)
        with open(filename, 'wb') as handle:
            pickle.dump(object, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return filename


    

    def make_llcs(save_name):
        if save_name==None:
            llcs_grok=[]
            llcs_nongrok=[]
            for epoch in tqdm(test_grok.model_epochs()):
                local_free_energy_grok=compute_local_free_energy(test_grok,epoch)
                local_free_energy_nogrok=compute_local_free_energy(test_nongrok,epoch)
                llcs_grok.append(local_free_energy_grok[2])
                llcs_nongrok.append(local_free_energy_nogrok[2])
            pairs=(llcs_grok,llcs_nongrok)
            saved_filename=save('llc_g_ng_pair',pairs)
            print(f'saved llc pairs as {saved_filename}')
            return pairs
        else:
            with open(pairs_path, 'rb') as handle:
                llc_pair = pickle.load(handle)
                #llcs_grok,llcs_nongrok=llc_pair
        return llc_pair


    def plot_llcs(llc_pair):
        llcs_grok,llcs_nongrok=llc_pair
        fig=make_subplots(rows=2,cols=3,subplot_titles=['Grok Acc','Grok Losses','LLC','Nongrok Acc','Nongrok Losses','LLC'])
        fig.add_trace(go.Scatter(x=list(range(len(test_grok.test_accuracies))),y=test_grok.test_accuracies,name='Grok Test',mode="lines", line=dict(color="red")),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(test_grok.train_accuracies))),y=test_grok.train_accuracies,name='Grok Train',mode="lines", line=dict(color="red",dash='dash')),row=1,col=1)
        fig.update_yaxes(title_text="Accuracy",row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(test_grok.test_losses))),y=test_grok.test_losses,name='Grok Test',mode="lines", line=dict(color="red")),row=1,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(test_grok.train_losses))),y=test_grok.train_losses,name='Grok Train',mode="lines", line=dict(color="red",dash='dash')),row=1,col=2)
        fig.update_yaxes(title_text="Loss", type='log',row=1, col=2)

        fig.add_trace(go.Scatter(x=20*np.array(list(range(len(llcs_grok)))),y=llcs_grok,name='Grok LLC',mode="lines", line=dict(color="red")),row=1,col=3)
        fig.add_trace(go.Scatter(x=20*np.array(list(range(len(llcs_nongrok)))),y=llcs_nongrok,name='Learn LLC',mode="lines", line=dict(color="blue")),row=1,col=3)
        fig.update_yaxes(title_text="LLC",row=1, col=3)
        #non grok
        fig.add_trace(go.Scatter(x=list(range(len(test_nongrok.test_accuracies))),y=test_nongrok.test_accuracies,name='Learn Test',mode="lines", line=dict(color="blue")),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(test_nongrok.train_accuracies))),y=test_nongrok.train_accuracies,name='Learn Train',mode="lines", line=dict(color="blue",dash='dash')),row=2,col=1)
        
        fig.update_yaxes(title_text="Accuracy",row=2, col=1)
        
        fig.add_trace(go.Scatter(x=list(range(len(test_nongrok.test_losses))),y=test_nongrok.test_losses,name='Learn Test',mode="lines", line=dict(color="blue")),row=2,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(test_nongrok.train_losses))),y=test_nongrok.train_losses,name='Learn Train',mode="lines", line=dict(color="blue",dash='dash')),row=2,col=2)
        fig.update_yaxes(title_text="Loss", type='log',row=2, col=2)

        fig.add_trace(go.Scatter(x=20*np.array(list(range(len(llcs_nongrok)))),y=llcs_nongrok,name='Learn LLC',mode="lines", line=dict(color="blue")),row=2,col=3)
        fig.add_trace(go.Scatter(x=20*np.array(list(range(len(llcs_grok)))),y=llcs_grok,name='Grok LLC',mode="lines", line=dict(color="red")),row=2,col=3)
        fig.update_yaxes(title_text="LLC",row=2, col=3)
        fig.update_layout(title_text=f'WD {test_grok.trainargs.weight_decay}, grok wm {test_grok.trainargs.weight_multiplier} non grok wm {test_nongrok.trainargs.weight_multiplier}')
        fig.show()

        test_grok.traincurves_and_iprs(test_nongrok).show()
    
    pairs_path=make_llcs(None)
    plot_llcs(pairs_path)


    


    #will be back to claim my coeffs


