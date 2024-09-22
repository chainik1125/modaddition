
import os
import sys
import dill


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

# machine learning imports

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import TensorDataset, DataLoader
import copy




class seed_average_onerun():
    def __init__(self,data_seed,sgd_seed,init_seed,params_dic):
        self.data_seed=data_seed
        self.sgd_seed=sgd_seed
        self.init_seed=init_seed
        self.params_dic=params_dic#Weight decay, learning rate etc...
        # params_dic={'weight_decay':weight_decay,'weight_multiplier':weight_multiplier,'learning_rate':learning_rate,'hidden_layers':hiddenlayers,'conv_channels':conv_channels,'train_size':train_size,'test_size':test_size,'dropout_p':dropout_prob}
        self.models={} #You'll save your models here
        self.train_losses=None
        self.test_losses=None
        self.train_accuracies=None
        self.test_accuracies=None
        self.train_loader=None
        self.test_loader=None
        self.start_time=None
        self.weights=None
        self.weightshistfig=None
        self.losscurvesfig=None
        self.svdata=None
        self.svdfig=None
        self.pcadata=None#I think I'll probably have a dictionary with all of the objects I need to calculate these
        self.pcafig=None
        self.neuroncorrs=None #Will be a dictionary with epoch, neuron indices as the objects.
        self.trainargs=None
        self.modelclass=None
        self.modelconfig=None 
        self.modelinstance=None
        self.iprs=None
        self.norms=None
        self.euclidcosine=None
        self.euclidcosinesteps=None
        self.linear_decomposition=None
        self.decile_means=None
        self.weight_samples=None
        self.bias_samples=None
    #Now I want to write scripts for the analysis function.
    
    def make_loss_curves(self):
        fig = make_subplots(rows=1, cols=2)
        train_losses=self.train_losses
        test_losses=self.test_losses
        train_accuracies=self.train_accuracies
        test_accuracies=self.test_accuracies

        fig.add_trace(
            go.Scatter(
                x=list(range(len(train_losses))),
                y=train_losses,
                name='Train losses'
            ),
            row=1,
            col=1)

        fig.add_trace(
            go.Scatter(
                x=list(range(len(test_losses))),
                y=test_losses,
                name='Test losses'
            ),
            row=1,
            col=1)

        fig.add_trace(
            go.Scatter(
                x=list(range(len(train_accuracies))),
                y=train_accuracies,
                name='Train Accuracy'
            ),
            row=1,
            col=2)

        fig.add_trace(
            go.Scatter(
                x=list(range(len(test_accuracies))),
                y=test_accuracies,
                name='Test Accuracy'

            ),
            row=1,
            col=2)

        # Update layout
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2,type='log')
        fig.update_yaxes(type='log', row=1, col=1)
        fig.update_xaxes(title_text='Epochs', row=1, col=2)

        fig.update_layout(
            title=f'CNN seeds {self.data_seed,self.sgd_seed,self.init_seed}',#conv_layers{str(single_data_obj.params_dic['conv_channels'])}',# MLP {str(single_data_obj.params_dic['hidden_layers'])}, lr={str(single_data_obj.params_dic['learning_rate'])}, wd={str(single_data_obj.params_dic['weight_decay'])}, wm={str(single_data_obj.params_dic['weight_multiplier'])}, train size={str(single_data_obj.params_dic['train_size'])}, test size={str(single_data_obj.params_dic['test_size'])}',
            xaxis_title='Epochs',
            # yaxis_title='Test accuracy',
            showlegend=True,
        )

        return fig#.write_image(root+"/Losses_plotly.png")
    def model_epochs(self):
        return list(self.models.keys())
    def make_weights_histogram(self,non_grokked_object,epoch):
        #last_epoch=max(self.models.keys())
        weights_grok=[]
        grok_state_dic=self.models[epoch]['model']
        
        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        
        weights_nogrok=[]
        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        weights_nogrok=[grok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]
        titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_nogrok))]


        fig=make_subplots(rows=2,cols=len(weights_grok)+1,subplot_titles=titles)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies,marker=dict(color='black'),showlegend=True,name='Grok Train'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.train_accuracies))),y=self.train_accuracies,marker=dict(color='orange'),showlegend=True,name='Grok Test'),row=1,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,marker=dict(color='black'),showlegend=True,name='Grok Train'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.train_accuracies,marker=dict(color='orange'),showlegend=True,name='Grok Test'),row=2,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)


        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=2, col=1)

        for i in range(len(weights_grok)):
            flattened_gw=torch.flatten(weights_grok[i]).detach().numpy()
            flattened_ngw=torch.flatten(weights_nogrok[i]).detach().numpy()
            showleg=False
            if i==0:
                showleg=True
            # fig.add_trace(go.Histogram(x=flattened_gw,marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1,col=i+2)
            # fig.add_trace(go.Histogram(x=flattened_ngw,marker=dict(color='blue'),showlegend=showleg,name='No Grok'),row=2,col=i+2)

            fig.add_trace(go.Histogram(x=[epoch], name=f'Grok_Epoch_{epoch}'),row=1,col=i+2)  # Placeholder histogram
            fig.add_trace(go.Histogram(x=[epoch + 0.5], name=f'NoGrok_Epoch_{epoch}'),row=2,col=i+2)  # Placeholder histogram
            fig.update_xaxes(title_text="Weight", row=1, col=i+1)
            fig.update_yaxes(title_text="Freq", row=1, col=i+1)
            fig.update_xaxes(title_text="Weight", row=2, col=i+1)
            fig.update_yaxes(title_text="Freq", row=2, col=i+1)
        
        return fig
    
    def make_weights_histogram2(self,non_grokked_object,epoch,fig):
        #last_epoch=max(self.models.keys())
        
        grok_state_dic=self.models[epoch]['model']
        
        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        
        
        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        weights_nogrok=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]
        #titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_nogrok))]

        fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))),y=self.train_losses,marker=dict(color='black'),showlegend=True,name='Train'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_losses))),y=self.test_losses,marker=dict(color='orange'),showlegend=True,name='Test'),row=1,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_losses), max(self.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=non_grokked_object.train_losses,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.test_losses,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=2,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_losses), max(non_grokked_object.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)



        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=1,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(self.train_accuracies))),y=self.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=1,col=2)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=2)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=2,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=2,col=2)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=2)

        if epoch==self.model_epochs()[0]:
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_yaxes(title_text="Loss", type='log',row=2, col=1)

            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_yaxes(title_text="Accuracy", row=1, col=2)
            fig.update_xaxes(title_text="Epoch", row=2, col=2)
            fig.update_yaxes(title_text="Accuracy", row=2, col=2)

        for i in range(len(weights_grok)):
            flattened_gw=np.abs(torch.flatten(weights_grok[i]).detach().numpy())
            sorted_gw=np.sort(flattened_gw)
            cdf_gw=np.arange(1, len(flattened_gw) + 1) / len(flattened_gw)
            ccdf_gw=1-cdf_gw

            flattened_ngw=np.abs(torch.flatten(weights_nogrok[i]).detach().numpy())
            sorted_ngw=np.sort(flattened_ngw)
            cdf_ngw=np.arange(1, len(flattened_ngw) + 1) / len(flattened_ngw)
            ccdf_ngw=1-cdf_ngw
            showleg=False
            if i==0:
                showleg=True
            fig.add_trace(go.Histogram(x=np.abs(flattened_gw),marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1,col=i+3)
            fig.add_trace(go.Histogram(x=np.abs(flattened_ngw),marker=dict(color='blue'),showlegend=showleg,name='No Grok'),row=2,col=i+3)
            #fig.add_trace(go.Scatter(x=sorted_gw,y=ccdf_gw,marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1,col=i+3)
            #fig.add_trace(go.Scatter(x=sorted_ngw,y=ccdf_ngw,marker=dict(color='blue'),showlegend=showleg,name='No Grok'),row=1,col=i+3)
            #fig.update_yaxes(type='log', row=1, col=i+3)
            #fig.update_yaxes(type='log', row=2, col=i+3)
            #fig.update_xaxes(type='log', row=1, col=i+3)
            #fig.update_xaxes(type='log', row=2, col=i+3)
            # fig.add_trace(go.Histogram(x=[epoch], name=f'Grok_Epoch_{epoch}'),row=1,col=i+2)  # Placeholder histogram
            # fig.add_trace(go.Histogram(x=[epoch + 0.5], name=f'NoGrok_Epoch_{epoch}'),row=2,col=i+2)  # Placeholder histogram
            if epoch==self.model_epochs()[0]:
                fig.update_xaxes(title_text="Weight", row=1, col=i+1)
                fig.update_xaxes(title_text="Weight", row=2, col=i+1)

    def make_weights_histogram_all(self,non_grokked_object,epoch,fig):
        #last_epoch=max(self.models.keys())
        
        grok_state_dic=self.models[epoch]['model']

        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        
        
        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        weights_nogrok=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]
        #titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_nogrok))]

        # fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))),y=self.train_losses,marker=dict(color='black'),showlegend=True,name='Train'),row=1,col=1)
        # fig.add_trace(go.Scatter(x=list(range(len(self.test_losses))),y=self.test_losses,marker=dict(color='orange'),showlegend=True,name='Test'),row=1,col=1)
        # fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_losses), max(self.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        # fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=non_grokked_object.train_losses,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=2,col=1)
        # fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.test_losses,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=2,col=1)
        # fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_losses), max(non_grokked_object.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)



        # fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=1,col=2)
        # fig.add_trace(go.Scatter(x=list(range(len(self.train_accuracies))),y=self.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=1,col=2)
        # fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=2)

        # fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=2,col=2)
        # fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=2,col=2)
        # fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=2)

        # if epoch==self.model_epochs()[0]:
        #     fig.update_xaxes(title_text="Epoch", row=1, col=1)
        #     fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
        #     fig.update_xaxes(title_text="Epoch", row=2, col=1)
        #     fig.update_yaxes(title_text="Loss", type='log',row=2, col=1)

        #     fig.update_xaxes(title_text="Epoch", row=1, col=2)
        #     fig.update_yaxes(title_text="Accuracy", row=1, col=2)
        #     fig.update_xaxes(title_text="Epoch", row=2, col=2)
        #     fig.update_yaxes(title_text="Accuracy", row=2, col=2)
        first=True
        if fig==None:
            fig = make_subplots(
            rows=1, cols=2,
            specs=[[{'type': 'xy','secondary_y':True}, {'type': 'xy','secondary_y':True}]],  # Define the types of plots in the main subplots
            horizontal_spacing=0.15,  # Space between the columns
            subplot_titles=('First Subplot with Inset', 'Second Subplot with Inset'),
            insets=[dict(cell=(1,1), l=0.01, b= 0.01),
            dict(cell=(1,2), l=0.01, b=0.01)])
        

        showleg=False
        # if i==0:
        #     showleg=True
        layerweightname='fc_layers.0.weight'
        flattened_gw=torch.flatten(grok_state_dic[layerweightname])
        flattened_ngw=torch.flatten(nogrok_state_dic[layerweightname])
        xmax=torch.max(torch.cat((flattened_gw,flattened_ngw)))
        xmin=torch.min(torch.cat((flattened_gw,flattened_ngw)))

        for i in range(len(weights_grok)):
            if first:
                all_flattened_gw=torch.flatten(weights_grok[i])
                all_flattened_ngw=torch.flatten(weights_nogrok[i])
                first=False
            else:
                all_flattened_gw=torch.cat((all_flattened_gw,torch.flatten(weights_grok[i])),dim=0)
                all_flattened_ngw=torch.cat((all_flattened_ngw,torch.flatten(weights_nogrok[i])),dim=0)
        
            sorted_gw=np.sort(flattened_gw)

            
            sorted_ngw=np.sort(flattened_ngw)
        
        cdf_gw=np.arange(1, len(flattened_gw) + 1) / len(flattened_gw)
        ccdf_gw=1-cdf_gw

        cdf_ngw=np.arange(1, len(flattened_ngw) + 1) / len(flattened_ngw)
        ccdf_ngw=1-cdf_ngw

        all_xmax=torch.max(torch.cat((all_flattened_gw,all_flattened_ngw)))
        all_xmin=torch.min(torch.cat((all_flattened_gw,all_flattened_ngw)))

        fig.add_trace(go.Histogram(x=flattened_gw,marker=dict(color='red'),name='Grokking',showlegend=True,nbinsx=128),row=1,col=1)
        fig.add_trace(go.Histogram(x=all_flattened_gw,marker=dict(color='red'),name='Grok',xbins=dict(start=all_xmin,end=all_xmax,size=(xmax-xmin)//128),showlegend=False,xaxis='x3',yaxis='y3'),row=1,col=1,secondary_y=True)#size=(xmax-xmin)//128)
        #fig.update_xaxes(range=[xmin, xmax], row=1, col=1, anchor='y2')
        fig.update_yaxes(range=[0, 3000], row=1, col=1, secondary_y=True, anchor='x3')

        fig.add_trace(go.Histogram(x=flattened_ngw,marker=dict(color='blue'),name='Learning',showlegend=True,nbinsx=128),row=1,col=2)
        fig.add_trace(go.Histogram(x=all_flattened_ngw,marker=dict(color='blue'),name='Learning',xbins=dict(start=xmin,end=xmax,size=(xmax-xmin)//128),showlegend=False,xaxis='x4',yaxis='y4'),row=1,col=2,secondary_y=True)#size=(xmax-xmin)//128)
        #fig.update_xaxes(range=[all_xmin, all_xmax], row=1, col=2)
        fig.update_yaxes(range=[0, 3000], row=1, col=2, secondary_y=True,anchor='x4')
        # fig.update_layout(
        #     xaxis2=dict(matches='x1'),  # Ensures that the second x-axis matches the first
        #     yaxis2=dict(matches='y1')  # Ensures that the second y-axis matches the first
        # )



        

        # fig.update_yaxes(title_text='Frequency of weights',row=1,col=1)
        # fig.update_xaxes(title_text='Weight value')
        # fig.update_yaxes(title_text='Frequency of weights',row=1,col=2)
        # fig.update_layout(
        #     xaxis2=dict(matches='x1'),  # Ensures that the second x-axis matches the first
        #     yaxis2=dict(matches='y1')  # Ensures that the second y-axis matches the first
        # )





        # bins=128
        # counts_gw = torch.histc(flattened_gw, bins=bins, min=xmin, max=xmax)
        # counts_ngw = torch.histc(flattened_ngw, bins=bins, min=xmin, max=xmax)

        # # Create bin edges
        # bin_edges = torch.linspace(xmin, xmax, steps=bins + 1)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # fig.add_trace(go.Scatter(x=bin_centers,y=counts_gw,mode='lines',name='Grokking',marker=dict(color='red'),showlegend=False),row=2,col=1)
        # fig.add_trace(go.Scatter(x=bin_centers,y=counts_ngw,mode='lines',name='Learning',marker=dict(color='blue'),showlegend=False),row=2,col=1)
        # #Now want to do this for one layer
        # layerweightname='fc_layers.0.weight'
        # flattened_gw=torch.flatten(grok_state_dic[layerweightname])
        # flattened_ngw=torch.flatten(nogrok_state_dic[layerweightname])
        # counts_gw = torch.histc(flattened_gw, bins=bins, min=xmin, max=xmax)
        # counts_ngw = torch.histc(flattened_ngw, bins=bins, min=xmin, max=xmax)

        # # Create bin edges
        # bin_edges = torch.linspace(xmin, xmax, steps=bins + 1)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # fig.add_trace(go.Scatter(x=bin_centers,y=counts_gw,mode='lines',name='Grokking',marker=dict(color='red'),showlegend=False),row=2,col=2)
        # fig.add_trace(go.Scatter(x=bin_centers,y=counts_ngw,mode='lines',name='Learning',marker=dict(color='blue'),showlegend=False),row=2,col=2)


            #fig.add_trace(go.Scatter(x=sorted_gw,y=ccdf_gw,marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1,col=i+3)
            #fig.add_trace(go.Scatter(x=sorted_ngw,y=ccdf_ngw,marker=dict(color='blue'),showlegend=showleg,name='No Grok'),row=1,col=i+3)
            #fig.update_yaxes(type='log', row=1, col=i+3)
            #fig.update_yaxes(type='log', row=2, col=i+3)
            #fig.update_xaxes(type='log', row=1, col=i+3)
            #fig.update_xaxes(type='log', row=2, col=i+3)
            # fig.add_trace(go.Histogram(x=[epoch], name=f'Grok_Epoch_{epoch}'),row=1,col=i+2)  # Placeholder histogram
            # fig.add_trace(go.Histogram(x=[epoch + 0.5], name=f'NoGrok_Epoch_{epoch}'),row=2,col=i+2)  # Placeholder histogram
        # if epoch==self.model_epochs()[0]:
        #     fig.update_xaxes(title_text="Weight", row=1, col=i+1)
        #     fig.update_xaxes(title_text="Weight", row=2, col=i+1)
        return fig

    def make_weights_histogram_all2(self,non_grokked_object,epoch,fig):
        #last_epoch=max(self.models.keys())
        
        grok_state_dic=self.models[epoch]['model']

        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        
        
        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        weights_nogrok=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]

        if fig==None:
            fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[r'$\text{(a) Grokking weights histogram - fully connected layer}$',r'$\text{(b) Learning weights histogram - fully connected layer}$',r'$\text{(c) Grokking weights histogram - whole network}$',r'$\text{(d) Grokking weights histogram - whole network}$'])      


        layerweightname='fc_layers.0.weight'
        flattened_gw=torch.flatten(grok_state_dic[layerweightname])
        flattened_ngw=torch.flatten(nogrok_state_dic[layerweightname])
        xmax=torch.max(torch.cat((flattened_gw,flattened_ngw)))
        xmin=torch.min(torch.cat((flattened_gw,flattened_ngw)))

        for i in range(len(weights_grok)):
            if first:
                all_flattened_gw=torch.flatten(weights_grok[i])
                all_flattened_ngw=torch.flatten(weights_nogrok[i])
                first=False
            else:
                all_flattened_gw=torch.cat((all_flattened_gw,torch.flatten(weights_grok[i])),dim=0)
                all_flattened_ngw=torch.cat((all_flattened_ngw,torch.flatten(weights_nogrok[i])),dim=0)
        
            sorted_gw=np.sort(flattened_gw)

            
            sorted_ngw=np.sort(flattened_ngw)
        
        cdf_gw=np.arange(1, len(flattened_gw) + 1) / len(flattened_gw)
        ccdf_gw=1-cdf_gw

        cdf_ngw=np.arange(1, len(flattened_ngw) + 1) / len(flattened_ngw)
        ccdf_ngw=1-cdf_ngw

        all_xmax=torch.max(torch.cat((all_flattened_gw,all_flattened_ngw)))
        all_xmin=torch.min(torch.cat((all_flattened_gw,all_flattened_ngw)))

        fig.add_trace(go.Histogram(x=flattened_gw,marker=dict(color='red'),name='Grokking',showlegend=True,nbinsx=128),row=1,col=1)
        fig.add_trace(go.Histogram(x=flattened_ngw,marker=dict(color='blue'),name='Learning',showlegend=True,nbinsx=128),row=1,col=2)
        
        fig.add_trace(go.Histogram(x=all_flattened_gw,marker=dict(color='red'),name='Grokking',xbins=dict(start=all_xmin,end=all_xmax,size=(xmax-xmin)//128),showlegend=False),row=2,col=1)#size=(xmax-xmin)//128)
        fig.add_trace(go.Histogram(x=all_flattened_ngw,marker=dict(color='blue'),name='Learning',xbins=dict(start=xmin,end=xmax,size=(xmax-xmin)//128),showlegend=False),row=2,col=2)#size=(xmax-xmin)//128)
   

        fig.update_layout(
            xaxis2=dict(matches='x1'),  # Ensures that the second x-axis matches the first
            yaxis2=dict(matches='y1')  # Ensures that the second y-axis matches the first
        )

        fig.update_layout(
            xaxis4=dict(matches='x3'),  # Ensures that the second x-axis matches the first
            yaxis4=dict(matches='y3')  # Ensures that the second y-axis matches the first
        )

        fig.update_xaxes(title_text=r'$\text{Weight values}$')
        fig.update_yaxes(title_text=r'$\text{Frequency}$',row=1,col=1)
        fig.update_yaxes(title_text=r'$\text{Frequency}$',row=1,col=2)
        fig.update_yaxes(title_text=r'$\text{Frequency}$',row=2,col=1)
        fig.update_yaxes(title_text=r'$\text{Frequency}$',row=2,col=2)
        fig.update_layout(    
        # xaxis=dict(tickformat="$\\text{%f}$"),
        # yaxis=dict(tickformat="$\\text{%f}$")
        )

        return fig
    
    def make_weights_histogram_modadd(self,non_grokked_object,epoch,fig):
        #last_epoch=max(self.models.keys())
        
        grok_state_dic=self.models[epoch]['model']

        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        
        print(grok_state_dic.keys())
        print(f'shapes {[x.shape for x in weights_grok]}')
        
        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        weights_nogrok=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]

        if fig==None:
            fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[r'$\text{(a) Grokking weights histogram - input to MLP}$',r'$\text{(b) Learning weights histogram - input to MLP}$',r'$\text{(c) Grokking weights histogram - MLP to output}$',r'$\text{(d) Learning weights histogram - MLP to output}$',r'$\text{(e) Grokking weights histogram - whole network}$',r'$\text{(f) Learning weights histogram - whole network}$'])      

        for i in range(len(weights_grok)):
            flattened_gw=torch.flatten(weights_grok[i])
            flattened_ngw=torch.flatten(weights_nogrok[i])
            xmax=torch.max(torch.cat((flattened_gw,flattened_ngw)))
            xmin=torch.min(torch.cat((flattened_gw,flattened_ngw)))
            fig.add_trace(go.Histogram(x=flattened_gw,marker=dict(color='red'),name='Grokking',showlegend=True,nbinsx=128),row=1+i,col=1)
            fig.add_trace(go.Histogram(x=flattened_ngw,marker=dict(color='blue'),name='Learning',showlegend=True,nbinsx=128),row=1+i,col=2)

        first=True
        for i in range(len(weights_grok)):
            if first:
                all_flattened_gw=torch.flatten(weights_grok[i])
                all_flattened_ngw=torch.flatten(weights_nogrok[i])
                first=False
            else:
                all_flattened_gw=torch.cat((all_flattened_gw,torch.flatten(weights_grok[i])),dim=0)
                all_flattened_ngw=torch.cat((all_flattened_ngw,torch.flatten(weights_nogrok[i])),dim=0)

        all_xmax=torch.max(torch.cat((all_flattened_gw,all_flattened_ngw)))
        all_xmin=torch.min(torch.cat((all_flattened_gw,all_flattened_ngw)))


        
        fig.add_trace(go.Histogram(x=all_flattened_gw,marker=dict(color='red'),name='Grokking',xbins=dict(start=all_xmin,end=all_xmax,size=(xmax-xmin)//128),showlegend=False),row=1+len(weights_grok),col=1)#size=(xmax-xmin)//128)
        fig.add_trace(go.Histogram(x=all_flattened_ngw,marker=dict(color='blue'),name='Learning',xbins=dict(start=xmin,end=xmax,size=(xmax-xmin)//128),showlegend=False),row=1+len(weights_nogrok),col=2)#size=(xmax-xmin)//128)
   

        fig.update_layout(
            xaxis2=dict(matches='x1'),  # Ensures that the second x-axis matches the first
            yaxis2=dict(matches='y1')  # Ensures that the second y-axis matches the first
        )

        fig.update_layout(
            xaxis4=dict(matches='x3'),  # Ensures that the second x-axis matches the first
            yaxis4=dict(matches='y3')  # Ensures that the second y-axis matches the first
        )

        fig.update_layout(
            xaxis5=dict(matches='x6'),  # Ensures that the second x-axis matches the first
            yaxis5=dict(matches='y6')  # Ensures that the second y-axis matches the first
        )

        fig.update_xaxes(title_text=r'$\text{Weight values}$')
        fig.update_yaxes(title_text=r'$\text{Frequency}$',row=1,col=1)
        fig.update_yaxes(title_text=r'$\text{Frequency}$',row=1,col=2)
        fig.update_yaxes(title_text=r'$\text{Frequency}$',row=2,col=1)
        fig.update_yaxes(title_text=r'$\text{Frequency}$',row=2,col=2)
        # fig.update_layout(    
        # # xaxis=dict(tickformat="$\\text{%f}$"),
        # # yaxis=dict(tickformat="$\\text{%f}$")
        # )

        return fig



    def weights_histogram_epochs(self,non_grokked_object):
        epochs=self.model_epochs()
        if (self.model_epochs()!=non_grokked_object.model_epochs()):
            print('Grok and Non-Grokked epochs are not the same!')
            print(f'Grokked epochs: {self.model_epochs()}')
            print(f'NG epochs: {non_grokked_object.model_epochs()}')
        
        print(epochs[0])
        fig=self.make_weights_histogram(non_grokked_object,epoch=epochs[0])
        def frame_data(epoch):
            print(f'epoch {epoch}')
            grok_state_dic=self.models[epoch]['model']
            weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
            
            
            nogrok_state_dic=non_grokked_object.models[epoch]['model']
            weights_nogrok=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]

            frame_list=[
            go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies),
            go.Scatter(x=list(range(len(self.train_accuracies))),y=self.train_accuracies),
            go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1]),
            go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies),
            go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.train_accuracies),
            go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1])]
            

            for i in range(len(weights_grok)):
            #     flattened_gw=torch.flatten(weights_grok[i]).detach().numpy()
            #     flattened_ngw=torch.flatten(weights_nogrok[i]).detach().numpy()
                frame_list.append(go.Histogram(x=[epoch], name=f'Grok_Epoch_{epoch}'))  # Placeholder histogram
                frame_list.append(go.Histogram(x=[epoch + 0.5], name=f'NoGrok_Epoch_{epoch}'))  # Placeholder histogram

            return go.Frame(data=frame_list,name=f'Epoch_{epoch}')
        
        frames=[]
        for epoch in epochs:
            frames.append(frame_data(epoch))
        fig.frames=frames

        #Now you need to add the slider

        sliders = [dict(
            active=0,
            currentvalue=dict(font=dict(size=12), prefix='Epoch: ', visible=True),
            pad=dict(t=50),
            steps=[dict(method='update',
                        args=[{'visible': [False] * len(fig.data)},
                            {'title': f'Epoch {epoch}'}],  # This second dictionary in the args list is for layout updates.
                        label=f'Epoch {epoch}') for epoch in epochs]
        )]

        # Initially setting all data to invisible, then making the first set visible
        for i, _ in enumerate(fig.data):
            fig.data[i].visible = False
        if fig.data:  # Check if there is any data
            fig.data[0].visible = True  # Making the first trace visible

        fig.update_layout(sliders=sliders,title_text='slide plz')

    #     fig.update_layout(
    #     # height=600,
    #     # width=800,
    #     title_text="Animated Graph with Time Step Control",
    #     sliders=sliders
    # )

        # To animate, Plotly requires that the first frame is explicitly set

        fig.show()
    

    def weights_histogram_epochs2(self,non_grokked_object):
        print('this one?')
        #epochs=self.model_epochs()[50:250][0::50]+[self.model_epochs()[-1]]#self.model_epochs()[::5]#[self.model_epochs()[0]]+[self.model_epochs()[5]]+self.model_epochs()[50:250][0::10]+[self.model_epochs()[-1]]
        epochs=self.model_epochs()
        ng_epochs=non_grokked_object.model_epochs()
        if epochs!=ng_epochs:
            print('Grokked and non-grokked epochs not the same!')
        grok_state_dic=self.models[epochs[0]]['model']
        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        titles=['Grok loss']+['Grok accuracy']+[f'Grok layer {i}' for i in range(len(weights_grok))]+['No grok loss']+['No grok Accuracy']+[f'No grok Layer {i}' for i in range(len(weights_grok))]
        fig=make_subplots(rows=2,cols=len(weights_grok)+2,subplot_titles=titles)
        for epoch in epochs:
            self.make_weights_histogram2(non_grokked_object,epoch,fig)
        
        total_plots=2*6+2*(len(weights_grok))
        print(len(fig.data))
        for i in range(len(fig.data)):
            fig.data[i].visible=False
        for i in range(total_plots):
            fig.data[i].visible=True
        #Now update the slider
        steps = []
        
        for i in range(len(epochs)):
            step = dict(
                method = 'restyle',
                args = ['visible',[False] * len(fig.data)],
            )
            for j in range(total_plots):
                step['args'][1][total_plots*i+j] = True
            steps.append(step)
        
        sliders = [dict(
            steps = steps,
        )]
        fig.layout.sliders = sliders
        fig.show()

    def svd_one_epoch(self,non_grokked_object,epoch,fig):
        grok_state_dic=self.models[epoch]['model']
        grok_weights=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]

        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        nogrok_weights=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]

        if fig==None:
            fig=make_subplots(rows=2,cols=4,subplot_titles=['Grok loss']+['No grok loss']+['Grok accuracy']+['No grok accuracy']+[f'Layer {i}' for i in range(len(grok_weights)-1)])#(len(grok_weights)+4)//2

        fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))),y=self.train_losses, name='Train',mode='lines',line=dict(width=2,color='black'),showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_losses))),y=self.test_losses, name='Test',mode='lines',line=dict(width=2,color='gold'),showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_losses), max(self.train_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=non_grokked_object.train_losses, name='No Grok Train',mode='lines',line=dict(width=2,color='black'),showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.test_losses, name='No Grok Test',mode='lines',line=dict(width=2,color='gold'),showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_losses), max(non_grokked_object.train_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=2)

        fig.add_trace(go.Scatter(x=list(range(len(self.train_accuracies))),y=self.train_accuracies, name='Grok Train',mode='lines',line=dict(width=2,color='black'),showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.test_accuracies, name='Grok Test',mode='lines',line=dict(width=2,color='gold'),showlegend=False), row=1, col=3)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.test_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=3)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.train_accuracies, name='No Grok Train',mode='lines',line=dict(width=2,color='black'),showlegend=False), row=1, col=4)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.test_accuracies, name='No Grok Test',mode='lines',line=dict(width=2,color='gold'),showlegend=False), row=1, col=4)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.test_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=4)

        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Loss",type='log', row=1, col=2)

        fig.update_xaxes(title_text="Epoch", row=1, col=3)
        fig.update_yaxes(title_text="Accuracy", row=1, col=3)
        fig.update_xaxes(title_text="Epoch", row=1, col=4)
        fig.update_yaxes(title_text="Accuracy", row=1, col=4)


        for i in range(len(grok_weights)-1):
            sg=np.flip(np.sort(np.ndarray.flatten(np.linalg.svd(grok_weights[i].detach().numpy(),compute_uv=False))))
            s_ng=np.flip(np.sort(np.ndarray.flatten(np.linalg.svd(nogrok_weights[i].detach().numpy(),compute_uv=False))))
            fig.add_trace(go.Scatter(x=list(range(len(sg))),y=sg/sg[0], name='Grok',mode='lines',line=dict(width=2,color='red'),showlegend=False), row=2, col=i+1)
            fig.add_trace(go.Scatter(x=list(range(len(s_ng))),y=s_ng/s_ng[0], name='No grok',mode='lines',line=dict(width=2,color='blue',dash='dash'),showlegend=False), row=2, col=i+1)
            #fig.update_yaxes(type="log", row=1, col=i+1)
            fig.update_xaxes(title_text="Order", row=2, col=i+1)
            fig.update_yaxes(title_text="SV", row=2, col=i+1)
        fig.data[-1].showlegend=True
        fig.data[-2].showlegend=True
        return fig
    
    def svd_one_epoch_prod(self,non_grokked_object,epoch,fig):
        grok_state_dic=self.models[epoch]['model']
        grok_weights=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]

        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        nogrok_weights=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]

        if fig==None:
            fig=make_subplots(rows=1,cols=len(grok_weights),subplot_titles=[r'$\text{(a) Singular values - first convolutional layer}$',r'$\text{(b) Singular values - second convolutional layer}$',r'$\text{(c) Singular values - fully connected layer}$'])
        fig.update_xaxes(title_text=r'$\text{Ranked singular values}$')
        fig.update_yaxes(title_text=r'$\text{Singular values}$')
        # fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))),y=self.train_losses, name='Train',mode='lines',line=dict(width=2,color='black'),showlegend=True), row=1, col=1)
        # fig.add_trace(go.Scatter(x=list(range(len(self.test_losses))),y=self.test_losses, name='Test',mode='lines',line=dict(width=2,color='gold'),showlegend=True), row=1, col=1)
        # fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_losses), max(self.train_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        # fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=non_grokked_object.train_losses, name='No Grok Train',mode='lines',line=dict(width=2,color='black'),showlegend=False), row=1, col=2)
        # fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.test_losses, name='No Grok Test',mode='lines',line=dict(width=2,color='gold'),showlegend=False), row=1, col=2)
        # fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_losses), max(non_grokked_object.train_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=2)

        # fig.add_trace(go.Scatter(x=list(range(len(self.train_accuracies))),y=self.train_accuracies, name='Grok Train',mode='lines',line=dict(width=2,color='black'),showlegend=False), row=1, col=3)
        # fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.test_accuracies, name='Grok Test',mode='lines',line=dict(width=2,color='gold'),showlegend=False), row=1, col=3)
        # fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.test_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=3)

        # fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.train_accuracies, name='No Grok Train',mode='lines',line=dict(width=2,color='black'),showlegend=False), row=1, col=4)
        # fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.test_accuracies, name='No Grok Test',mode='lines',line=dict(width=2,color='gold'),showlegend=False), row=1, col=4)
        # fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.test_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=4)

        # fig.update_xaxes(title_text="Epoch", row=1, col=1)
        # fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
        # fig.update_xaxes(title_text="Epoch", row=1, col=2)
        # fig.update_yaxes(title_text="Loss",type='log', row=1, col=2)

        # fig.update_xaxes(title_text="Epoch", row=1, col=3)
        # fig.update_yaxes(title_text="Accuracy", row=1, col=3)
        # fig.update_xaxes(title_text="Epoch", row=1, col=4)
        # fig.update_yaxes(title_text="Accuracy", row=1, col=4)


        for i in range(len(grok_weights)):
            sg=np.flip(np.sort(np.ndarray.flatten(np.linalg.svd(grok_weights[i].detach().numpy(),compute_uv=False))))
            s_ng=np.flip(np.sort(np.ndarray.flatten(np.linalg.svd(nogrok_weights[i].detach().numpy(),compute_uv=False))))
            fig.add_trace(go.Scatter(x=list(range(len(sg))),y=sg/sg[0], name='Grok',mode='lines',line=dict(width=2,color='red'),showlegend=False), row=1, col=i+1)
            fig.add_trace(go.Scatter(x=list(range(len(s_ng))),y=s_ng/s_ng[0], name='No grok',mode='lines',line=dict(width=2,color='blue',dash='dash'),showlegend=False), row=1, col=i+1)
            #fig.update_yaxes(type="log", row=1, col=i+1)
            fig.update_xaxes(title_text="Order", row=2, col=i+1)
            fig.update_yaxes(title_text="SV", row=2, col=i+1)
        fig.data[-1].showlegend=True
        fig.data[-2].showlegend=True
        return fig
    
    def svd_epochs(self,non_grokked_object):
        epochs=[self.model_epochs()[0]]+[self.model_epochs()[5]]+self.model_epochs()[10:350][0::50]+[self.model_epochs()[-1]]
        ng_epochs=non_grokked_object.model_epochs()
        if epochs!=ng_epochs:
            print('Grokked and non-grokked epochs not the same!')
        grok_state_dic=self.models[epochs[0]]['model']
        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        titles=['Grok loss']+['No grok loss']+['Grok Accuracy']+['No grok accuracy']+[f'Layer {i}' for i in range(len(weights_grok))]
        fig=make_subplots(rows=2,cols=4,subplot_titles=titles)#(len(weights_grok)+4)//2
        for epoch in epochs:
            self.svd_one_epoch(non_grokked_object,epoch,fig)
        
        total_plots=2*6+2*(len(weights_grok)-1)
        print(len(fig.data))
        for i in range(len(fig.data)):
            fig.data[i].visible=False
        for i in range(total_plots):
            fig.data[i].visible=True
        #Now update the slider
        steps = []
        
        for i in range(len(epochs)):
            step = dict(
                method = 'restyle',
                args = ['visible',[False] * len(fig.data)],
            )
            for j in range(total_plots):
                step['args'][1][total_plots*i+j] = True
            steps.append(step)
        
        sliders = [dict(
            steps = steps,
        )]
        fig.layout.sliders = sliders
        return fig
    
    def pca_one_epoch(self,non_grokked_object,epoch,fig):
        grok_state_dic=self.models[epoch]['model']
        grok_weights=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]

        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        nogrok_weights=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]

        u_grok,s_grok,v_grok=torch.svd(grok_weights[-2])
        u_nogrok,s_nogrok,v_nogrok=torch.svd(nogrok_weights[-2])

        norms = grok_weights[-2].norm(p=2, dim=1, keepdim=True)
        grok_normed_weights=grok_weights[-2]/norms
        ng_norms=nogrok_weights[-2].norm(p=2,dim=1,keepdim=True)
        nogrok_normed_weights=nogrok_weights[-2]/ng_norms

        grok_proj_matrix=torch.matmul(grok_weights[-2], torch.tensor(v_grok[:, :2]))
        grok_proj_matrix_normed=torch.matmul(grok_normed_weights, torch.tensor(v_grok[:, :2]))

        nogrok_proj_matrix=torch.matmul(nogrok_weights[-2], torch.tensor(v_nogrok[:, :2]))
        nogrok_proj_matrix_normed=torch.matmul(nogrok_normed_weights, torch.tensor(v_nogrok[:, :2]))

        # np.flip(np.sort(np.ndarray.flatten(t0[:,:,:,0])))



        if fig==None:
            fig = make_subplots(rows=2, cols=3, subplot_titles=['Grok accuracy','Grok Loss','Grok MLP','Grok normed MLP','No grok accuracy','No grok MLP', 'No Grok normed MLP'])#shared_xaxes=True, shared_yaxes=True

        #Add traces for the loss curves
        fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))),y=self.train_losses, name='Train',mode='lines',line=dict(width=2,color='black'),showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_losses))),y=self.test_losses, name='Test',mode='lines',line=dict(width=2,color='gold'),showlegend=True), row=1, col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_losses), max(self.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=non_grokked_object.train_losses, name='No Grok Train',mode='lines',line=dict(width=2,color='black'),showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.test_losses, name='No Grok Test',mode='lines',line=dict(width=2,color='gold'),showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_losses), max(non_grokked_object.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)
        #Add traces for accuracy curves
        
        fig.add_trace(go.Scatter(x=list(range(len(self.train_accuracies))),y=self.train_accuracies, name='Grok Train',mode='lines',line=dict(width=2,color='black'),showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.test_accuracies, name='Grok Test',mode='lines',line=dict(width=2,color='gold'),showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=2)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.train_accuracies, name='No grok train',mode='lines',line=dict(width=2,color='black'),showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.test_accuracies, name='No grok test',mode='lines',line=dict(width=2,color='gold'),showlegend=False), row=2, col=2)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=2)
        
        
        # Add traces
        fig.add_trace(go.Scatter(x=grok_proj_matrix[:,0].detach().numpy(), y=grok_proj_matrix[:,1].detach().numpy(), name='Grok', mode='markers',marker=dict(size=10,color='red'),showlegend=True), row=1, col=2+1)
        fig.add_trace(go.Scatter(x=nogrok_proj_matrix[:,0].detach().numpy(), y=nogrok_proj_matrix[:,1].detach().numpy(), name='No Grok', mode='markers',marker=dict(size=10,color='blue'),showlegend=True), row=2, col=2+1)
        fig.add_trace(go.Scatter(x=grok_proj_matrix_normed[:,0].detach().numpy(), y=grok_proj_matrix_normed[:,1].detach().numpy(), name='Grok CNN', mode='markers',marker=dict(size=10,color='red'),showlegend=False), row=1, col=3+1)
        fig.add_trace(go.Scatter(x=nogrok_proj_matrix_normed[:,0].detach().numpy(), y=nogrok_proj_matrix_normed[:,1].detach().numpy(), name='No Grok CNN', mode='markers',marker=dict(size=10,color='blue'),showlegend=False), row=2, col=3+1)
        
        #update axis
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Accuracy", row=1, col=2)

        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_yaxes(title_text="Loss",type='log', row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy", row=2, col=2)

        
        # Update x and y axis labels correctly for each subplot
        fig.update_xaxes(title_text="PC 1", row=1, col=1+2)
        fig.update_xaxes(title_text="PC 1", row=1, col=2+2)
        fig.update_xaxes(title_text="PC 1", row=2, col=1+2)
        fig.update_xaxes(title_text="PC 1", row=2, col=2+2)
        fig.update_yaxes(title_text="PC 2", row=1, col=1+2)
        fig.update_yaxes(title_text="PC 2", row=2, col=1+2)
        #fig.update_yaxes(title_text="Principal Component 2", row=1, col=2)
        return fig
    
    def pca_epochs(self,non_grokked_object):
        epochs=[self.model_epochs()[0]]+[self.model_epochs()[5]]+self.model_epochs()[10:350][0::50]+[self.model_epochs()[-1]]
        ng_epochs=non_grokked_object.model_epochs()
        if epochs!=ng_epochs:
            print('Grokked and non-grokked epochs not the same!')
        # grok_state_dic=self.models[epochs[0]]['model']
        # weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        titles=['Grok loss','Grok accuracy','Grok MLP','Grok Normed PCA','No grok loss','No grok accuracy','No grok PCA', 'No grok Normed PCA']
        fig=make_subplots(rows=2,cols=3+1,subplot_titles=titles)
        for epoch in epochs:
            self.pca_one_epoch(non_grokked_object,epoch,fig)
        
        total_plots=2*6+(2*2)
        print(len(fig.data))
        for i in range(len(fig.data)):
            fig.data[i].visible=False
        for i in range(total_plots):
            fig.data[i].visible=True
        #Now update the slider
        steps = []
        
        for i in range(len(epochs)):
            step = dict(
                method = 'restyle',
                args = ['visible',[False] * len(fig.data)],
            )
            for j in range(total_plots):
                step['args'][1][total_plots*i+j] = True
            steps.append(step)
        
        sliders = [dict(
            steps = steps,
        )]
        fig.layout.sliders = sliders
        return fig


    #best way is just to pass dataset as an argument I think
    def make_activations_histogram2(self,non_grokked_object,epoch,fig,sortby,dataset):
        #first need to get the activations
        dtype=torch.float32
        grok_state_dic=self.models[epoch]['model']
        grok_model=self.modelclass(**self.modelconfig)
        result=grok_model.load_state_dict(grok_state_dic,strict=False)
        if len(result.missing_keys)>0 or len(result.unexpected_keys)>0:
            print('Missing keys!')
            print("Missing keys:", result.missing_keys)
            print("Unexpected keys:", result.unexpected_keys)
        # if 'ising_dataset' in locals():
        #     pass
        # else:
        #     dataset_filename="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/Data/IsingML_L16_traintest.pickle"
        #     with open(dataset_filename, "rb") as handle:
        #         dataset = dill.load(handle)[1]
        test_set=generate_test_set(dataset,1000)
        activations_grok, output, cleanup = get_activations(grok_model,test_set[0])
        cleanup()
        if sortby=='var':
            sorted_activations_grok={key:activations_grok[key].var(dim=0) for key in activations_grok.keys()}
        if sortby=='mean':
            sorted_activations_grok={key:activations_grok[key].mean(dim=0) for key in activations_grok.keys()}
        if sortby=='absmean':
            sorted_activations_grok={key:activations_grok[key].abs().mean(dim=0) for key in activations_grok.keys()}
        if sortby=='all':
            sorted_activations_grok=activations_grok

        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        nogrok_model=non_grokked_object.modelclass(**non_grokked_object.modelconfig)
        result=nogrok_model.load_state_dict(nogrok_state_dic,strict=False)
        if len(result.missing_keys)>0 or len(result.unexpected_keys)>0:
            print('Missing keys!')
            print("Missing keys:", result.missing_keys)
            print("Unexpected keys:", result.unexpected_keys)
        activations_nogrok, output_ng, cleanup = get_activations(nogrok_model,test_set[0])
        if sortby=='var':
            sorted_activations_nogrok={key:activations_nogrok[key].var(dim=0) for key in activations_nogrok.keys()}
        if sortby=='mean':
            sorted_activations_nogrok={key:activations_nogrok[key].mean(dim=0) for key in activations_nogrok.keys()}
        if sortby=='absmean':
            sorted_activations_nogrok={key:activations_nogrok[key].abs().mean(dim=0) for key in activations_nogrok.keys()}
        if sortby=='all':
            sorted_activations_nogrok=activations_nogrok
        
        #removing 
        sorted_activations_grok.pop('conv_layers.2 (MaxPool2d)')
        if 'conv_layers.5 (MaxPool2d)' in sorted_activations_grok:
            sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
        sorted_activations_grok.pop('fc_layers.3 (Linear)')
        sorted_activations_nogrok.pop('conv_layers.2 (MaxPool2d)')
        sorted_activations_nogrok.pop('conv_layers.5 (MaxPool2d)')
        sorted_activations_nogrok.pop('fc_layers.3 (Linear)')

        if fig==None:
            fig=make_subplots(rows=2,cols=2+len(list(sorted_activations_grok.keys())),subplot_titles=['Grok loss']+['Grok accuracy']+[f'{key}' for key in sorted_activations_grok.keys()]+['No grok loss']+['No grok accuracy']+[f'Layer {key}' for key in sorted_activations_grok.keys()])#(len(grok_weights)+4)//2

        #titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_nogrok))]

        fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))),y=self.train_losses,marker=dict(color='black'),showlegend=True,name='Train'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_losses))),y=self.test_losses,marker=dict(color='orange'),showlegend=True,name='Test'),row=1,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_losses), max(self.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=non_grokked_object.train_losses,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.test_losses,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=2,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_losses), max(non_grokked_object.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)



        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=1,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(self.train_accuracies))),y=self.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=1,col=2)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=2)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=2,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=2,col=2)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=2)

        if epoch==self.model_epochs()[0]:
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_yaxes(title_text="Loss", type='log',row=2, col=1)

            fig.update_xaxes(title_text="Epoch", row=1, col=2)
            fig.update_yaxes(title_text="Accuracy", row=1, col=2)
            fig.update_xaxes(title_text="Epoch", row=2, col=2)
            fig.update_yaxes(title_text="Accuracy", row=2, col=2)
        count=0
        for key in sorted_activations_grok.keys():
            flattened_gw=np.abs(torch.flatten(sorted_activations_grok[key]).detach().numpy())
            sorted_gw=np.sort(flattened_gw)
            cdf_gw=np.arange(1, len(flattened_gw) + 1) / len(flattened_gw)
            ccdf_gw=1-cdf_gw

            flattened_ngw=np.abs(torch.flatten(sorted_activations_nogrok[key]).detach().numpy())
            sorted_ngw=np.sort(flattened_ngw)
            cdf_ngw=np.arange(1, len(flattened_ngw) + 1) / len(flattened_ngw)
            ccdf_ngw=1-cdf_ngw
            showleg=True
            fig.add_trace(go.Histogram(x=flattened_gw,marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1,col=count+3)
            fig.add_trace(go.Histogram(x=flattened_ngw,marker=dict(color='blue'),showlegend=showleg,name='No Grok'),row=2,col=count+3)
            #fig.add_trace(go.Scatter(x=sorted_gw,y=ccdf_gw,marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1,col=count+3)
            #fig.add_trace(go.Scatter(x=sorted_ngw,y=ccdf_ngw,marker=dict(color='blue'),showlegend=showleg,name='No Grok'),row=1,col=count+3)
            showleg=False
            #fig.update_yaxes(type='log', row=1, col=count+3)
            #fig.update_yaxes(type='log', row=2, col=count+3)
            #fig.update_xaxes(type='log', row=1, col=count+3)
            #fig.update_xaxes(type='log', row=2, col=count+3)
            # fig.add_trace(go.Histogram(x=[epoch], name=f'Grok_Epoch_{epoch}'),row=1,col=i+2)  # Placeholder histogram
            # fig.add_trace(go.Histogram(x=[epoch + 0.5], name=f'NoGrok_Epoch_{epoch}'),row=2,col=i+2)  # Placeholder histogram
            if epoch==self.model_epochs()[0]:
                fig.update_xaxes(title_text="Weight", row=1, col=count+1)
                fig.update_xaxes(title_text="Weight", row=2, col=count+1)
            count+=1
        return fig

    def activations_epochs(self,non_grokked_object,sortby,dataset):
        epochs=[self.model_epochs()[0]]+[self.model_epochs()[5]]+self.model_epochs()[10:350][0::50]+[self.model_epochs()[-1]]
        ng_epochs=non_grokked_object.model_epochs()
        if self.model_epochs()!=ng_epochs:
            print('Grokked and non-grokked epochs not the same!')

        #Just doing this to generate the layer titles
        # grok_state_dic=self.models[epoch]['model']
        # grok_model=self.modelclass(**self.modelconfig)
        # result=grok_model.load_state_dict(grok_state_dic,strict=False)
                

        # fig=self.make_activations_histogram2(non_grokked_object=non_grokked_object,epoch=epochs[0],fig=None,sortby=sortby,dataset=dataset)
        first=True
        for epoch in epochs:#Note that firrst epoch in list gets put in as the first object - why not just put it in a list?
            if first:
                fig=self.make_activations_histogram2(non_grokked_object=non_grokked_object,epoch=epoch,fig=None,sortby=sortby,dataset=dataset)
                first=False
            else:
                self.make_activations_histogram2(non_grokked_object=non_grokked_object,epoch=epoch,fig=fig,sortby=sortby,dataset=dataset)
        total_plots=2*6+3*2
        print(len(fig.data))
        for i in range(len(fig.data)):
            fig.data[i].visible=False
        for i in range(total_plots):
            fig.data[i].visible=True
        #Now update the slider
        steps = []
        
        for i in range(len(epochs)):
            step = dict(
                method = 'restyle',
                args = ['visible',[False] * len(fig.data)],
            )
            for j in range(total_plots):
                step['args'][1][total_plots*i+j] = True
            steps.append(step)
        
        sliders = [dict(
            steps = steps,
        )]
        fig.layout.sliders = sliders
        fig.update_layout(title=f'Activations histogram, sorted by: {sortby}, no pool')
        return fig


    def correlation_one_epoch(self,non_grokked_object,sortby,epoch,neuron_index,images_tensor,feature_funcs,fig,dataset):
        model_grok=self.modelclass(**self.modelconfig)
        grok_state_dic=self.models[epoch]['model']
        result=model_grok.load_state_dict(grok_state_dic,strict=False)
        if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
            print(result.missing_keys,result.unexpected_keys)

        features_tensor=construct_features_tensor(images_tensor=images_tensor,feature_funcs=feature_funcs)
        activations_grok, output_grok, cleanup_grok = get_activations(model_grok,images_tensor)
        sorted_activations_grok,acts_indices_grok=get_acts_dict(single_run=self,dataset=dataset,epoch=epoch,sortby=sortby)

        #removing 
        sorted_activations_grok.pop('conv_layers.2 (MaxPool2d)')
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
        sorted_activations_grok.pop('fc_layers.3 (Linear)')
        activations_grok.pop('conv_layers.2 (MaxPool2d)')
        activations_grok.pop('conv_layers.5 (MaxPool2d)')
        activations_grok.pop('fc_layers.3 (Linear)')
        acts_indices_grok.pop('conv_layers.2 (MaxPool2d)')
        acts_indices_grok.pop('conv_layers.5 (MaxPool2d)')
        acts_indices_grok.pop('fc_layers.3 (Linear)')


        model_nogrok=non_grokked_object.modelclass(**non_grokked_object.modelconfig)
        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        result=model_nogrok.load_state_dict(nogrok_state_dic,strict=False)
        if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
            print(result.missing_keys,result.unexpected_keys)

        activations_nogrok, output_nogrok, cleanup_nogrok = get_activations(model_nogrok,images_tensor)
        sorted_activations_nogrok,acts_indices_nogrok=get_acts_dict(single_run=non_grokked_object,dataset=dataset,epoch=epoch,sortby=sortby)

        #removing 
        sorted_activations_nogrok.pop('conv_layers.2 (MaxPool2d)')
        sorted_activations_nogrok.pop('conv_layers.5 (MaxPool2d)')
        sorted_activations_nogrok.pop('fc_layers.3 (Linear)')
        activations_nogrok.pop('conv_layers.2 (MaxPool2d)')
        activations_nogrok.pop('conv_layers.5 (MaxPool2d)')
        activations_nogrok.pop('fc_layers.3 (Linear)')
        acts_indices_nogrok.pop('conv_layers.2 (MaxPool2d)')
        acts_indices_nogrok.pop('conv_layers.5 (MaxPool2d)')
        acts_indices_nogrok.pop('fc_layers.3 (Linear)')

        #Now you need to populate the layers
        feature_dim=features_tensor.shape[1]
        if fig==None:
            fig=make_subplots(rows=max(4,2*feature_dim),cols=1+len(sorted_activations_grok.keys()),subplot_titles=['Grok loss']+[f'ENERGY, layer{key}' for key in sorted_activations_grok.keys()]+['No grok loss']+[f'ENERGY, layer {key}' for key in sorted_activations_grok.keys()]+['Grok accuracy']+[f'MAG, Layer {key}' for key in sorted_activations_grok.keys()]+['No grok accuracy']+[f'MAG, Layer {key}' for key in sorted_activations_grok.keys()])#(len(grok_weights)+4)//2

        #titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_nogrok))]

        fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))),y=self.train_losses,marker=dict(color='black'),showlegend=True,name='Train'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_losses))),y=self.test_losses,marker=dict(color='orange'),showlegend=True,name='Test'),row=1,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_losses), max(self.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=non_grokked_object.train_losses,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.test_losses,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=2,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_losses), max(non_grokked_object.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)



        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=3,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.train_accuracies))),y=self.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=3,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=3, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=4,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=4,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=4, col=1)

        if epoch==self.model_epochs()[0]:
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
            if epoch==self.model_epochs()[0]:
                fig.update_xaxes(title_text="Weight", row=1, col=count+1)
                fig.update_xaxes(title_text="Weight", row=2, col=count+1)
            count+=1
        return fig
    

    def traincurves_and_iprs(self,non_grokked_object,remove_wdloss=False):
        titles=[r'$\text{(a) Grokking accuracy in training}$',r'$\text{(b) Learning accuracy in training}$',r'$\text{(c) Grokking loss in training}$',r'$\text{(d) Learning loss in training}$']+[r'$\text{(e) Weight norm}$',r'$\text{(f) IPR r=2}$',r'$\text{(f) IPR r=4}$',r'$\text{(f) IPR r=1/2}$']
        fig=make_subplots(rows=2,cols=4,subplot_titles=titles)

        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies,mode='lines',line=dict(color='red',dash='dash'),showlegend=True,name=r'$\text{Grokking train}$'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.test_accuracies,mode='lines',line=dict(color='red',dash='solid'),showlegend=True,name=r'$\text{Grokking test}$'),row=1,col=1)
        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)
        fig.update_yaxes(title_text=r'$\text{Accuracy}$',row=1,col=1)
        # fig.add_vrect(
        #     x0=10000,  # Start of the region (on the x-axis)
        #     x1=30000,  # End of the region (on the x-axis)
        #     fillcolor="grey",  # Color of the rectangle
        #     opacity=0.2,  # Opacity of the rectangle
        #     layer="below",  # Draw below the data points
        #     line_width=0,
        #     row=1,
        #     col=1  # No border line
        #     )
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,mode='lines',line=dict(color='blue',dash='dash'),showlegend=True,name=r'$\text{Learning train}$'),row=1,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.test_accuracies,mode='lines',line=dict(color='blue',dash='solid'),showlegend=True,name=r'$\text{Learning test}$'),row=1,col=2)
        fig.update_yaxes(title_text=r'$\text{Accuracy}$',row=1,col=2)
        #losses
        if remove_wdloss:
            removedloss_self=self.train_losses-(self.trainargs.weightdecay)*np.array(self.weight_norms)#not square-rooted
        else:
            removedloss_self=self.train_losses
        fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))),y=removedloss_self,mode='lines',line=dict(color='red',dash='dash'),showlegend=False,name=r'$\text{Grokking train}$'),row=1,col=3)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_losses))),y=self.test_losses,mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grokking test}$'),row=1,col=3)
        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)
        fig.update_yaxes(title_text=r'$\text{Cross entropy loss}$',type='log',row=1,col=3)
        # fig.add_vrect(
        #     x0=10000,  # Start of the region (on the x-axis)
        #     x1=30000,  # End of the region (on the x-axis)
        #     fillcolor="grey",  # Color of the rectangle
        #     opacity=0.2,  # Opacity of the rectangle
        #     layer="below",  # Draw below the data points
        #     line_width=0,
        #     row=2,
        #     col=1  # No border line
        #     )
        if remove_wdloss:
            removedloss_ng=non_grokked_object.train_losses-(non_grokked_object.trainargs.weightdecay)*np.array(non_grokked_object.weight_norms)#not square-rooted
        else:
            removedloss_ng=non_grokked_object.train_losses
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.test_losses,mode='lines',line=dict(color='blue',dash='dash'),showlegend=False,name=r'$\text{Learning train}$'),row=1,col=4)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=removedloss_ng,mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learning test}$'),row=1,col=4)
        fig.update_yaxes(title_text=r'$\text{Cross entropy loss}$',type='log',row=1,col=4)
        

        fig.add_trace(go.Scatter(x=list(range(len(self.norms))),y=self.norms,mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grok weight norm}$'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.norms))),y=non_grokked_object.norms,mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learn weight norm}$'),row=2,col=1)
        fig.update_yaxes(title_text=r'$\text{Weight norm}$',type='log',row=2,col=1)

        fig.add_trace(go.Scatter(x=list(range(len(self.iprs))),y=np.array(self.iprs)[:,0],mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grok IPR r=2}$'),row=2,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.iprs))),y=np.array(non_grokked_object.iprs)[:,0],mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learn IPR r=2}$'),row=2,col=2)
        fig.update_yaxes(title_text=r'$\text{IPR r=2}$',row=2,col=2)

        fig.add_trace(go.Scatter(x=list(range(len(self.iprs))),y=np.array(self.iprs)[:,1],mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grok IPR r=4}$'),row=2,col=3)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.iprs))),y=np.array(non_grokked_object.iprs)[:,1],mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learn IPR r=4}$'),row=2,col=3)
        fig.update_yaxes(title_text=r'$\text{IPR r=2}$',row=2,col=3)

        fig.add_trace(go.Scatter(x=list(range(len(self.iprs))),y=np.array(self.iprs)[:,2],mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grok IPR r=1/2}$'),row=2,col=4)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.iprs))),y=np.array(non_grokked_object.iprs)[:,2],mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learn IPR r=1/2}$'),row=2,col=4)
        fig.update_yaxes(title_text=r'$\text{IPR r=1/2}$',row=2,col=4)



        fig.update_xaxes(title_text=r'$\text{Epoch}$')

        fig.update_layout(
            #title='Example Plot',
            #legend=dict(
            #    x=1,  # Position legend outside the plot area
            #    xanchor='auto',  # Automatically determine the best horizontal position
            #    y=1,  # Position at the top of the plot
            #    yanchor='auto'  # Automatically determine the best vertical position
            #),
            margin=dict(  # Adjust margins to provide more space
                l=20,  # Left margin
                r=150,  # Right margin increased to prevent overlap
                t=50,  # Top margin
                b=20   # Bottom margin
                ))
        grids=False
        for i in range(1,3):
            for j in range(1,3):
                fig.update_xaxes(showgrid=grids, row=i, col=j)  # Disable x-axis grid lines
                fig.update_yaxes(showgrid=grids, row=i, col=j)  # Disable y-axis grid lines
        #acc

        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)

        # fig.update_xaxes(title_text="Epoch", row=1, col=1)
        # fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
        # fig.update_xaxes(title_text="Epoch", row=1, col=1)
        # fig.update_yaxes(title_text="Loss",type='log', row=1, col=2)

        # fig.update_xaxes(title_text="Epoch", row=2, col=1)
        # fig.update_yaxes(title_text="Accuracy", row=2, col=1)

        # fig.update_xaxes(title_text="Epoch", row=2, col=2)
        # fig.update_yaxes(title_text="Accuracy", row=2, col=2)

        fig.update_layout(title_text=f'Training curves: hidden layers={(self.trainargs.hiddenlayers,non_grokked_object.trainargs.hiddenlayers)},wd={self.trainargs.weight_decay,non_grokked_object.trainargs.weight_decay},wm={self.trainargs.weight_multiplier,non_grokked_object.trainargs.weight_multiplier},train size={(self.trainargs.train_size,non_grokked_object.trainargs.train_size)},train frac={(self.trainargs.train_fraction,non_grokked_object.trainargs.train_fraction)}, lr={(self.trainargs.lr,non_grokked_object.trainargs.lr)}')
        return fig
    
    def traincurves_and_decile(self,non_grokked_object,remove_wdloss=False):
        titles=[r'$\text{(a) Grokking accuracy in training}$',r'$\text{(b) Learning accuracy in training}$',r'$\text{(c) Grokking loss in training}$',r'$\text{(d) Learning loss in training}$']+[r'$\text{(e) Weight norm}$',r'$\text{(f) IPR r=2}$',r'$\text{(f) IPR r=4}$',r'$\text{(f) IPR r=1/2}$']
        fig=make_subplots(rows=2,cols=4,subplot_titles=titles)

        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies,mode='lines',line=dict(color='red',dash='dash'),showlegend=True,name=r'$\text{Grokking train}$'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.test_accuracies,mode='lines',line=dict(color='red',dash='solid'),showlegend=True,name=r'$\text{Grokking test}$'),row=1,col=1)
        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)
        fig.update_yaxes(title_text=r'$\text{Accuracy}$',row=1,col=1)
        # fig.add_vrect(
        #     x0=10000,  # Start of the region (on the x-axis)
        #     x1=30000,  # End of the region (on the x-axis)
        #     fillcolor="grey",  # Color of the rectangle
        #     opacity=0.2,  # Opacity of the rectangle
        #     layer="below",  # Draw below the data points
        #     line_width=0,
        #     row=1,
        #     col=1  # No border line
        #     )
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,mode='lines',line=dict(color='blue',dash='dash'),showlegend=True,name=r'$\text{Learning train}$'),row=1,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.test_accuracies,mode='lines',line=dict(color='blue',dash='solid'),showlegend=True,name=r'$\text{Learning test}$'),row=1,col=2)
        fig.update_yaxes(title_text=r'$\text{Accuracy}$',row=1,col=2)
        #losses
        if remove_wdloss:
            removedloss_self=self.train_losses-(self.trainargs.weightdecay)*np.array(self.weight_norms)#not square-rooted
        else:
            removedloss_self=self.train_losses
        fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))),y=removedloss_self,mode='lines',line=dict(color='red',dash='dash'),showlegend=False,name=r'$\text{Grokking train}$'),row=1,col=3)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_losses))),y=self.test_losses,mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grokking test}$'),row=1,col=3)
        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)
        fig.update_yaxes(title_text=r'$\text{Cross entropy loss}$',type='log',row=1,col=3)
        # fig.add_vrect(
        #     x0=10000,  # Start of the region (on the x-axis)
        #     x1=30000,  # End of the region (on the x-axis)
        #     fillcolor="grey",  # Color of the rectangle
        #     opacity=0.2,  # Opacity of the rectangle
        #     layer="below",  # Draw below the data points
        #     line_width=0,
        #     row=2,
        #     col=1  # No border line
        #     )
        if remove_wdloss:
            removedloss_ng=non_grokked_object.train_losses-(non_grokked_object.trainargs.weightdecay)*np.array(non_grokked_object.weight_norms)#not square-rooted
        else:
            removedloss_ng=non_grokked_object.train_losses
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.test_losses,mode='lines',line=dict(color='blue',dash='dash'),showlegend=False,name=r'$\text{Learning train}$'),row=1,col=4)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=removedloss_ng,mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learning test}$'),row=1,col=4)
        fig.update_yaxes(title_text=r'$\text{Cross entropy loss}$',type='log',row=1,col=4)
        

        fig.add_trace(go.Scatter(x=list(range(len(self.norms))),y=self.norms,mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grok weight norm}$'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.norms))),y=non_grokked_object.norms,mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learn weight norm}$'),row=2,col=1)
        fig.update_yaxes(title_text=r'$\text{Weight norm}$',type='log',row=2,col=1)

        fig.add_trace(go.Scatter(x=list(range(len(self.iprs))),y=np.array(self.iprs)[:,0],mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grok IPR r=2}$'),row=2,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.iprs))),y=np.array(non_grokked_object.iprs)[:,0],mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learn IPR r=2}$'),row=2,col=2)
        fig.update_yaxes(title_text=r'$\text{IPR r=2}$',row=2,col=2)

        fig.add_trace(go.Scatter(x=list(range(len(self.decile))),y=np.array(self.iprs)[:,1],mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grok IPR r=4}$'),row=2,col=3)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.iprs))),y=np.array(non_grokked_object.iprs)[:,1],mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learn IPR r=4}$'),row=2,col=3)
        fig.update_yaxes(title_text=r'$\text{IPR r=2}$',row=2,col=3)

        fig.add_trace(go.Scatter(x=list(range(len(self.iprs))),y=np.array(self.iprs)[:,2],mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grok IPR r=1/2}$'),row=2,col=4)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.iprs))),y=np.array(non_grokked_object.iprs)[:,2],mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learn IPR r=1/2}$'),row=2,col=4)
        fig.update_yaxes(title_text=r'$\text{IPR r=1/2}$',row=2,col=4)



        fig.update_xaxes(title_text=r'$\text{Epoch}$')

        fig.update_layout(
            #title='Example Plot',
            #legend=dict(
            #    x=1,  # Position legend outside the plot area
            #    xanchor='auto',  # Automatically determine the best horizontal position
            #    y=1,  # Position at the top of the plot
            #    yanchor='auto'  # Automatically determine the best vertical position
            #),
            margin=dict(  # Adjust margins to provide more space
                l=20,  # Left margin
                r=150,  # Right margin increased to prevent overlap
                t=50,  # Top margin
                b=20   # Bottom margin
                ))
        grids=False
        for i in range(1,3):
            for j in range(1,3):
                fig.update_xaxes(showgrid=grids, row=i, col=j)  # Disable x-axis grid lines
                fig.update_yaxes(showgrid=grids, row=i, col=j)  # Disable y-axis grid lines
        #acc

        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)

        # fig.update_xaxes(title_text="Epoch", row=1, col=1)
        # fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
        # fig.update_xaxes(title_text="Epoch", row=1, col=1)
        # fig.update_yaxes(title_text="Loss",type='log', row=1, col=2)

        # fig.update_xaxes(title_text="Epoch", row=2, col=1)
        # fig.update_yaxes(title_text="Accuracy", row=2, col=1)

        # fig.update_xaxes(title_text="Epoch", row=2, col=2)
        # fig.update_yaxes(title_text="Accuracy", row=2, col=2)

        fig.update_layout(title_text=f'Training curves: hidden layers={(self.trainargs.hiddenlayers,non_grokked_object.trainargs.hiddenlayers)},wd={self.trainargs.weight_decay,non_grokked_object.trainargs.weight_decay},wm={self.trainargs.weight_multiplier,non_grokked_object.trainargs.weight_multiplier},train size={(self.trainargs.train_size,non_grokked_object.trainargs.train_size)},train frac={(self.trainargs.train_fraction,non_grokked_object.trainargs.train_fraction)}, lr={(self.trainargs.lr,non_grokked_object.trainargs.lr)}')
        return fig

    def cosine_sim(self,non_grokked_object,remove_wdloss=False):
        titles=[r'$\text{(a) Grokking accuracy in training}$',r'$\text{(b) Learning accuracy in training}$',r'$\text{(c) Grokking loss in training}$',r'$\text{(d) Learning loss in training}$']+[r'$\text{(e) Full network cosine}$',r'$\text{Layer 1 cosine}$',r'$\text{Layer 2 cosine}$',r'$\text{Full network step changes}$']
        fig=make_subplots(rows=2,cols=4,subplot_titles=titles)
        titles2=[r'$\text{(a) Grokking accuracy in training}$',r'$\text{(b) Learning accuracy in training}$',r'$\text{(c) Grokking loss in training}$',r'$\text{(d) Learning loss in training}$']+[r'$\text{(e) Full network cosine}$',r'$\text{Full network step changes}$']
        fig = make_subplots(
            rows=2, cols=4,
            specs=[
                [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
                [{"colspan": 2}, None, {"colspan": 2}, None]],
                subplot_titles=titles2)

        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies,mode='lines',line=dict(color='red',dash='dash'),showlegend=True,name=r'$\text{Grokking train}$'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.test_accuracies,mode='lines',line=dict(color='red',dash='solid'),showlegend=True,name=r'$\text{Grokking test}$'),row=1,col=1)
        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)
        fig.update_yaxes(title_text=r'$\text{Accuracy}$',row=1,col=1)
        # fig.add_vrect(
        #     x0=10000,  # Start of the region (on the x-axis)
        #     x1=30000,  # End of the region (on the x-axis)
        #     fillcolor="grey",  # Color of the rectangle
        #     opacity=0.2,  # Opacity of the rectangle
        #     layer="below",  # Draw below the data points
        #     line_width=0,
        #     row=1,
        #     col=1  # No border line
        #     )
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,mode='lines',line=dict(color='blue',dash='dash'),showlegend=True,name=r'$\text{Learning train}$'),row=1,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.test_accuracies,mode='lines',line=dict(color='blue',dash='solid'),showlegend=True,name=r'$\text{Learning test}$'),row=1,col=2)
        fig.update_yaxes(title_text=r'$\text{Accuracy}$',row=1,col=2)
        #losses
        if remove_wdloss:
            removedloss_self=self.train_losses-(self.trainargs.weightdecay)*np.array(self.weight_norms)#not square-rooted
        else:
            removedloss_self=self.train_losses
        fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))),y=removedloss_self,mode='lines',line=dict(color='red',dash='dash'),showlegend=False,name=r'$\text{Grokking train}$'),row=1,col=3)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_losses))),y=self.test_losses,mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grokking test}$'),row=1,col=3)
        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)
        fig.update_yaxes(title_text=r'$\text{Cross entropy loss}$',type='log',row=1,col=3)
        # fig.add_vrect(
        #     x0=10000,  # Start of the region (on the x-axis)
        #     x1=30000,  # End of the region (on the x-axis)
        #     fillcolor="grey",  # Color of the rectangle
        #     opacity=0.2,  # Opacity of the rectangle
        #     layer="below",  # Draw below the data points
        #     line_width=0,
        #     row=2,
        #     col=1  # No border line
        #     )

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=non_grokked_object.train_losses,mode='lines',line=dict(color='blue',dash='dash'),showlegend=False,name=r'$\text{Learning train}$'),row=1,col=4)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.test_losses,mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learning test}$'),row=1,col=4)
        fig.update_yaxes(title_text=r'$\text{Cross entropy loss}$',type='log',row=1,col=4)
        

        
        fig.add_trace(go.Scatter(x=list(range(len(self.euclidcosine[:,0]))),y=np.array(self.euclidcosine)[:,0],mode='lines',line=dict(color='red',dash='solid'),showlegend=True,name=r'$\text{Grok euclid whole network}$'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.euclidcosine[:,0]))),y=np.array(non_grokked_object.euclidcosine)[:,0],mode='lines',line=dict(color='blue',dash='solid'),showlegend=True,name=r'$\text{Learn euclid whole network}$'),row=2,col=1)
        fig.update_yaxes(title_text=r'$\text{All layer cosine similarity}$',row=2,col=1)

        #If you want layers
        # for layer_index in range(1,self.euclidcosine.shape[1]):
        #     fig.add_trace(go.Scatter(x=list(range(len(self.euclidcosine))),y=np.array(self.euclidcosine)[:,layer_index],mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grok IPR r=4}$'),row=2,col=1+layer_index)
        #     fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.euclidcosine))),y=np.array(non_grokked_object.euclidcosine)[:,layer_index],mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learn IPR r=4}$'),row=2,col=1+layer_index)
        #     fig.update_yaxes(title_text=fr'Layer {str(layer_index)} cosine sim ',row=2,col=1+layer_index)

        if self.euclidcosine.shape[1]<4:
            fig.add_trace(go.Scatter(x=list(range(len(self.euclidcosinesteps[:,0]))),y=self.euclidcosinesteps[:,0],mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grok weight norm}$'),row=2,col=3)
            fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.euclidcosinesteps[:,0]))),y=non_grokked_object.euclidcosinesteps[:,0],mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learn weight norm}$'),row=2,col=3)
            fig.update_yaxes(title_text=r'$\text{Cosine of steps}$',row=2,col=3)



        fig.update_xaxes(title_text=r'$\text{Epoch}$')

        fig.update_layout(
            #title='Example Plot',
            #legend=dict(
            #    x=1,  # Position legend outside the plot area
            #    xanchor='auto',  # Automatically determine the best horizontal position
            #    y=1,  # Position at the top of the plot
            #    yanchor='auto'  # Automatically determine the best vertical position
            #),
            margin=dict(  # Adjust margins to provide more space
                l=20,  # Left margin
                r=150,  # Right margin increased to prevent overlap
                t=50,  # Top margin
                b=20   # Bottom margin
                ))
        grids=False
        for i in range(1,3):
            for j in range(1,3):
                fig.update_xaxes(showgrid=grids, row=i, col=j)  # Disable x-axis grid lines
                fig.update_yaxes(showgrid=grids, row=i, col=j)  # Disable y-axis grid lines
        #acc

        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)

        # fig.update_xaxes(title_text="Epoch", row=1, col=1)
        # fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
        # fig.update_xaxes(title_text="Epoch", row=1, col=1)
        # fig.update_yaxes(title_text="Loss",type='log', row=1, col=2)

        # fig.update_xaxes(title_text="Epoch", row=2, col=1)
        # fig.update_yaxes(title_text="Accuracy", row=2, col=1)

        # fig.update_xaxes(title_text="Epoch", row=2, col=2)
        # fig.update_yaxes(title_text="Accuracy", row=2, col=2)

        fig.update_layout(title_text=f'Training curves: hidden layers={(self.trainargs.hiddenlayers,non_grokked_object.trainargs.hiddenlayers)},wd={self.trainargs.weight_decay,non_grokked_object.trainargs.weight_decay},wm={self.trainargs.weight_multiplier,non_grokked_object.trainargs.weight_multiplier},train size={(self.trainargs.train_size,non_grokked_object.trainargs.train_size)},train frac={(self.trainargs.train_fraction,non_grokked_object.trainargs.train_fraction)}, lr={(self.trainargs.lr,non_grokked_object.trainargs.lr)}, step=1')
        return fig
    

    def linear_decomposition_plot(self,non_grokked_object):
        titles=[r'$\text{(a) Grokking accuracy in training}$',r'$\text{(b) Learning accuracy in training}$',r'$\text{(c) Grokking loss in training}$',r'$\text{(d) Learning loss in training}$',
        r'$\text{(e) Grok linear and non-linear loss - test}$',r'$\text{Learn linear and non-linear loss - test}$',r'$\text{Grok linear and non-linear norms - test}$',r'$\text{Learn linear and non-linear norms - test}$',
        r'$\text{(e) Grok linear and non-linear loss - train}$',r'$\text{Learn linear and non-linear loss - train}$',r'$\text{Grok linear and non-linear norms - train}$',r'$\text{Learn linear and non-linear norms - train}$']


        fig=make_subplots(rows=3,cols=4,subplot_titles=titles)

        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies,mode='lines',line=dict(color='red',dash='dash'),showlegend=True,name=r'$\text{Grokking train}$'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.test_accuracies,mode='lines',line=dict(color='red',dash='solid'),showlegend=True,name=r'$\text{Grokking test}$'),row=1,col=1)
        
        fig.update_yaxes(title_text=r'$\text{Accuracy}$',row=1,col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,mode='lines',line=dict(color='blue',dash='dash'),showlegend=True,name=r'$\text{Learning train}$'),row=1,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.test_accuracies,mode='lines',line=dict(color='blue',dash='solid'),showlegend=True,name=r'$\text{Learning test}$'),row=1,col=2)
        fig.update_yaxes(title_text=r'$\text{Accuracy}$',row=1,col=2)
        #losses

        fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))),y=self.train_losses,mode='lines',line=dict(color='red',dash='dash'),showlegend=False,name=r'$\text{Grokking train}$'),row=1,col=3)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_losses))),y=self.test_losses,mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grokking test}$'),row=1,col=3)
        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)
        fig.update_yaxes(title_text=r'$\text{Cross entropy loss}$',type='log',row=1,col=3)


        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=non_grokked_object.train_losses,mode='lines',line=dict(color='blue',dash='dash'),showlegend=False,name=r'$\text{Learning train}$'),row=1,col=4)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.test_losses,mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learning test}$'),row=1,col=4)
        fig.update_yaxes(title_text=r'$\text{Cross entropy loss}$',type='log',row=1,col=4)
        
        #Test decomposition plots
        fig.add_trace(go.Scatter(x=list(range(len(self.linear_decomposition['linear_loss_test']))),y=self.linear_decomposition['linear_loss_test'],mode='lines',line=dict(color='red',dash='dash'),showlegend=True,name=r'$\text{Grok linear loss test}$'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.linear_decomposition['non_linear_loss_test']))),y=self.linear_decomposition['non_linear_loss_test'],mode='lines',line=dict(color='red',dash='dot'),showlegend=True,name=r'$\text{Grok non-linear loss test}$'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.linear_decomposition['full_loss_test']))),y=self.linear_decomposition['full_loss_test'],mode='lines',line=dict(color='red',dash='solid'),showlegend=True,name=r'$\text{Grok full loss test}$'),row=2,col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.linear_decomposition['linear_loss_test']))),y=non_grokked_object.linear_decomposition['linear_loss_test'],mode='lines',line=dict(color='blue',dash='dash'),showlegend=True,name=r'$\text{Learn linear loss test}$'),row=2,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.linear_decomposition['non_linear_loss_test']))),y=non_grokked_object.linear_decomposition['non_linear_loss_test'],mode='lines',line=dict(color='blue',dash='dot'),showlegend=True,name=r'$\text{Learn non-linear loss test}$'),row=2,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.linear_decomposition['full_loss_test']))),y=non_grokked_object.linear_decomposition['full_loss_test'],mode='lines',line=dict(color='blue',dash='solid'),showlegend=True,name=r'$\text{Learn full loss test}$'),row=2,col=2)

        fig.add_trace(go.Scatter(x=list(range(len(self.linear_decomposition['linear_norm_test']))),y=self.linear_decomposition['linear_norm_test'],mode='lines',line=dict(color='red',dash='dash'),showlegend=True,name=r'$\text{Grok linear norm test}$'),row=2,col=3)
        fig.add_trace(go.Scatter(x=list(range(len(self.linear_decomposition['non_linear_norm_test']))),y=self.linear_decomposition['non_linear_norm_test'],mode='lines',line=dict(color='red',dash='dot'),showlegend=True,name=r'$\text{Grok non-linear norm test}$'),row=2,col=3)
        fig.add_trace(go.Scatter(x=list(range(len(self.linear_decomposition['diff_norm_test']))),y=self.linear_decomposition['diff_norm_test'],mode='lines',line=dict(color='red',dash='solid'),showlegend=True,name=r'$\text{Grok diff norm test}$'),row=2,col=3)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.linear_decomposition['linear_norm_test']))),y=non_grokked_object.linear_decomposition['linear_norm_test'],mode='lines',line=dict(color='blue',dash='dash'),showlegend=True,name=r'$\text{Learn linear norm test }$'),row=2,col=4)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.linear_decomposition['non_linear_norm_test']))),y=non_grokked_object.linear_decomposition['non_linear_norm_test'],mode='lines',line=dict(color='blue',dash='dot'),showlegend=True,name=r'$\text{Learn non-linear norm test}$'),row=2,col=4)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.linear_decomposition['diff_norm_test']))),y=non_grokked_object.linear_decomposition['diff_norm_test'],mode='lines',line=dict(color='blue',dash='solid'),showlegend=True,name=r'$\text{Learn diff norm test}$'),row=2,col=4)

        #Train decomposition plots
        fig.add_trace(go.Scatter(x=list(range(len(self.linear_decomposition['linear_loss_train']))),y=self.linear_decomposition['linear_loss_train'],mode='lines',line=dict(color='red',dash='dash'),showlegend=False,name=r'$\text{Grok linear loss train}$'),row=3,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.linear_decomposition['non_linear_loss_train']))),y=self.linear_decomposition['non_linear_loss_train'],mode='lines',line=dict(color='red',dash='dot'),showlegend=False,name=r'$\text{Grok non-linear loss train}$'),row=3,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.linear_decomposition['full_loss_train']))),y=self.linear_decomposition['full_loss_train'],mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grok full loss train}$'),row=3,col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.linear_decomposition['linear_loss_train']))),y=non_grokked_object.linear_decomposition['linear_loss_train'],mode='lines',line=dict(color='blue',dash='dash'),showlegend=False,name=r'$\text{Learn linear loss train}$'),row=3,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.linear_decomposition['non_linear_loss_train']))),y=non_grokked_object.linear_decomposition['non_linear_loss_train'],mode='lines',line=dict(color='blue',dash='dot'),showlegend=False,name=r'$\text{Learn non-linear loss train}$'),row=3,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.linear_decomposition['full_loss_train']))),y=non_grokked_object.linear_decomposition['full_loss_train'],mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learn full loss train}$'),row=3,col=2)

        fig.add_trace(go.Scatter(x=list(range(len(self.linear_decomposition['linear_norm_train']))),y=self.linear_decomposition['linear_norm_train'],mode='lines',line=dict(color='red',dash='dash'),showlegend=False,name=r'$\text{Grok linear norm train}$'),row=3,col=3)
        fig.add_trace(go.Scatter(x=list(range(len(self.linear_decomposition['non_linear_norm_train']))),y=self.linear_decomposition['non_linear_norm_train'],mode='lines',line=dict(color='red',dash='dot'),showlegend=False,name=r'$\text{Grok non-linear norm train}$'),row=3,col=3)
        fig.add_trace(go.Scatter(x=list(range(len(self.linear_decomposition['diff_norm_train']))),y=self.linear_decomposition['diff_norm_train'],mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grok diff norm train}$'),row=3,col=3)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.linear_decomposition['linear_norm_train']))),y=non_grokked_object.linear_decomposition['linear_norm_train'],mode='lines',line=dict(color='blue',dash='dash'),showlegend=False,name=r'$\text{Learn linear norm train}$'),row=3,col=4)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.linear_decomposition['non_linear_norm_train']))),y=non_grokked_object.linear_decomposition['non_linear_norm_train'],mode='lines',line=dict(color='blue',dash='dot'),showlegend=False,name=r'$\text{Learn non-linear norm train}$'),row=3,col=4)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.linear_decomposition['diff_norm_train']))),y=non_grokked_object.linear_decomposition['diff_norm_train'],mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learn diff norm train}$'),row=3,col=4)


        fig.update_xaxes(title_text=r'$\text{Epoch}$')

        fig.update_layout(
            #title='Example Plot',
            #legend=dict(
            #    x=1,  # Position legend outside the plot area
            #    xanchor='auto',  # Automatically determine the best horizontal position
            #    y=1,  # Position at the top of the plot
            #    yanchor='auto'  # Automatically determine the best vertical position
            #),
            margin=dict(  # Adjust margins to provide more space
                l=20,  # Left margin
                r=150,  # Right margin increased to prevent overlap
                t=50,  # Top margin
                b=20   # Bottom margin
                ))
        grids=False
        for i in range(1,3):
            for j in range(1,3):
                fig.update_xaxes(showgrid=grids, row=i, col=j)  # Disable x-axis grid lines
                fig.update_yaxes(showgrid=grids, row=i, col=j)  # Disable y-axis grid lines
        #acc

        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)

        # fig.update_xaxes(title_text="Epoch", row=1, col=1)
        # fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
        # fig.update_xaxes(title_text="Epoch", row=1, col=1)
        # fig.update_yaxes(title_text="Loss",type='log', row=1, col=2)

        # fig.update_xaxes(title_text="Epoch", row=2, col=1)
        # fig.update_yaxes(title_text="Accuracy", row=2, col=1)

        # fig.update_xaxes(title_text="Epoch", row=2, col=2)
        # fig.update_yaxes(title_text="Accuracy", row=2, col=2)

        fig.update_layout(title_text=f'Training curves: hidden layers={(self.trainargs.hiddenlayers,non_grokked_object.trainargs.hiddenlayers)},wd={self.trainargs.weight_decay,non_grokked_object.trainargs.weight_decay},wm={self.trainargs.weight_multiplier,non_grokked_object.trainargs.weight_multiplier},train size={(self.trainargs.train_size,non_grokked_object.trainargs.train_size)},train frac={(self.trainargs.train_fraction,non_grokked_object.trainargs.train_fraction)}, lr={(self.trainargs.lr,non_grokked_object.trainargs.lr)}')
        return fig

    def correlation_epochs(self,non_grokked_object,sortby,neuron_index,images_tensor,feature_funcs,dataset):
        epochs=self.model_epochs()[0:4]+[self.model_epochs()[5]]+self.model_epochs()[10:350][0::50]+[self.model_epochs()[-1]]
        ng_epochs=non_grokked_object.model_epochs()
        if self.model_epochs()!=ng_epochs:
            print('Grokked and non-grokked epochs not the same!')

        #Just doing this to generate the layer titles
        # grok_state_dic=self.models[epoch]['model']
        # grok_model=self.modelclass(**self.modelconfig)
        # result=grok_model.load_state_dict(grok_state_dic,strict=False)
                

        # fig=self.make_activations_histogram2(non_grokked_object=non_grokked_object,epoch=epochs[0],fig=None,sortby=sortby,dataset=dataset)
        first=True
        for epoch in epochs:#Note that firrst epoch in list gets put in as the first object - why not just put it in a list?
            if first:
                fig=self.correlation_one_epoch(non_grokked_object=non_grokked_object,epoch=epoch,fig=None,sortby=sortby,images_tensor=images_tensor,feature_funcs=feature_funcs,dataset=dataset,neuron_index=neuron_index)
                first=False
            else:
                self.correlation_one_epoch(non_grokked_object=non_grokked_object,epoch=epoch,sortby=sortby,images_tensor=images_tensor,feature_funcs=feature_funcs,fig=fig,dataset=dataset,neuron_index=neuron_index)
        total_plots=2*6+2*3*2#2 features, 3 layers, 2=1 grokked +1 non-grokked
        print(len(fig.data))
        for i in range(len(fig.data)):
            fig.data[i].visible=False
        for i in range(total_plots):
            fig.data[i].visible=True
        #Now update the slider
        steps = []
        
        for i in range(len(epochs)):
            step = dict(
                method = 'restyle',
                args = ['visible',[False] * len(fig.data)],
            )
            for j in range(total_plots):
                step['args'][1][total_plots*i+j] = True
            steps.append(step)
        
        sliders = [dict(
            steps = steps,
        )]
        fig.layout.sliders = sliders
        fig.update_layout(title=f'Activations histogram, sorted by: {sortby}, no pool')
        return fig

    def plot_traincurves(self,non_grokked_object):
        titles=[r'$\text{(a) Grokking accuracy in training}$',r'$\text{(b) Learning accuracy in training}$',r'$\text{(c) Grokking loss in training}$',r'$\text{(d) Learning loss in training}$']
        fig=make_subplots(rows=2,cols=2,subplot_titles=titles)

        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies,mode='lines',line=dict(color='red',dash='dash'),showlegend=True,name=r'$\text{Grokking train}$'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.test_accuracies,mode='lines',line=dict(color='red',dash='solid'),showlegend=True,name=r'$\text{Grokking test}$'),row=1,col=1)
        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)
        fig.update_yaxes(title_text=r'$\text{Accuracy}$',row=1,col=1)
        # fig.add_vrect(
        #     x0=10000,  # Start of the region (on the x-axis)
        #     x1=30000,  # End of the region (on the x-axis)
        #     fillcolor="grey",  # Color of the rectangle
        #     opacity=0.2,  # Opacity of the rectangle
        #     layer="below",  # Draw below the data points
        #     line_width=0,
        #     row=1,
        #     col=1  # No border line
        #     )
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,mode='lines',line=dict(color='blue',dash='dash'),showlegend=True,name=r'$\text{Learning train}$'),row=1,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.test_accuracies,mode='lines',line=dict(color='blue',dash='solid'),showlegend=True,name=r'$\text{Learning test}$'),row=1,col=2)
        fig.update_yaxes(title_text=r'$\text{Accuracy}$',row=1,col=2)
        #losses
        fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))),y=self.train_losses,mode='lines',line=dict(color='red',dash='dash'),showlegend=False,name=r'$\text{Grokking train}$'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_losses))),y=self.test_losses,mode='lines',line=dict(color='red',dash='solid'),showlegend=False,name=r'$\text{Grokking test}$'),row=2,col=1)
        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)
        fig.update_yaxes(title_text=r'$\text{Cross entropy loss}$',type='log',row=2,col=1)
        # fig.add_vrect(
        #     x0=10000,  # Start of the region (on the x-axis)
        #     x1=30000,  # End of the region (on the x-axis)
        #     fillcolor="grey",  # Color of the rectangle
        #     opacity=0.2,  # Opacity of the rectangle
        #     layer="below",  # Draw below the data points
        #     line_width=0,
        #     row=2,
        #     col=1  # No border line
        #     )
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.train_losses,mode='lines',line=dict(color='blue',dash='dash'),showlegend=False,name=r'$\text{Learning train}$'),row=2,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=non_grokked_object.test_losses,mode='lines',line=dict(color='blue',dash='solid'),showlegend=False,name=r'$\text{Learning test}$'),row=2,col=2)
        fig.update_yaxes(title_text=r'$\text{Cross entropy loss}$',type='log',row=2,col=2)
        
        fig.update_xaxes(title_text=r'$\text{Epoch}$',type='log')

        fig.update_layout(
            #title='Example Plot',
            #legend=dict(
            #    x=1,  # Position legend outside the plot area
            #    xanchor='auto',  # Automatically determine the best horizontal position
            #    y=1,  # Position at the top of the plot
            #    yanchor='auto'  # Automatically determine the best vertical position
            #),
            margin=dict(  # Adjust margins to provide more space
                l=20,  # Left margin
                r=150,  # Right margin increased to prevent overlap
                t=50,  # Top margin
                b=20   # Bottom margin
                ))
        grids=False
        for i in range(1,3):
            for j in range(1,3):
                fig.update_xaxes(showgrid=grids, row=i, col=j)  # Disable x-axis grid lines
                fig.update_yaxes(showgrid=grids, row=i, col=j)  # Disable y-axis grid lines
        #acc

        #fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)

        # fig.update_xaxes(title_text="Epoch", row=1, col=1)
        # fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
        # fig.update_xaxes(title_text="Epoch", row=1, col=1)
        # fig.update_yaxes(title_text="Loss",type='log', row=1, col=2)

        # fig.update_xaxes(title_text="Epoch", row=2, col=1)
        # fig.update_yaxes(title_text="Accuracy", row=2, col=1)

        # fig.update_xaxes(title_text="Epoch", row=2, col=2)
        # fig.update_yaxes(title_text="Accuracy", row=2, col=2)

        #fig.update_layout(title_text=f'Training curves: hidden layers={(self.trainargs.hiddenlayers,non_grokked_object.trainargs.hiddenlayers)},wd={self.trainargs.weight_decay,non_grokked_object.trainargs.weight_decay},wm={self.trainargs.weight_multiplier,non_grokked_object.trainargs.weight_multiplier},train size={(self.trainargs.train_size,non_grokked_object.trainargs.train_size)}, lr={(self.trainargs.lr,non_grokked_object.trainargs.lr)}')
        return fig
    def correlation_one_epoch_prod(self,non_grokked_object,sortby,epoch,neuron_index,images_tensor,feature_funcs,fig,dataset):
        model_grok=self.modelclass(**self.modelconfig)
        grok_state_dic=self.models[epoch]['model']
        result=model_grok.load_state_dict(grok_state_dic,strict=False)
        if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
            print(result.missing_keys,result.unexpected_keys)

        features_tensor=construct_features_tensor(images_tensor=images_tensor,feature_funcs=feature_funcs)
        activations_grok, output_grok, cleanup_grok = get_activations(model_grok,images_tensor)
        sorted_activations_grok,acts_indices_grok=get_acts_dict(single_run=self,dataset=dataset,epoch=epoch,sortby=sortby)

        #removing 
        sorted_activations_grok.pop('conv_layers.3 (Conv2d)')
        sorted_activations_grok.pop('conv_layers.0 (Conv2d)')
        sorted_activations_grok.pop('conv_layers.2 (MaxPool2d)')
        sorted_activations_grok.pop('conv_layers.5 (MaxPool2d)')
        sorted_activations_grok.pop('fc_layers.3 (Linear)')
        activations_grok.pop('conv_layers.3 (Conv2d)')
        activations_grok.pop('conv_layers.0 (Conv2d)')
        activations_grok.pop('conv_layers.2 (MaxPool2d)')
        activations_grok.pop('conv_layers.5 (MaxPool2d)')
        activations_grok.pop('fc_layers.3 (Linear)')
        acts_indices_grok.pop('conv_layers.3 (Conv2d)')
        acts_indices_grok.pop('conv_layers.0 (Conv2d)')
        acts_indices_grok.pop('conv_layers.2 (MaxPool2d)')
        acts_indices_grok.pop('conv_layers.5 (MaxPool2d)')
        acts_indices_grok.pop('fc_layers.3 (Linear)')


        model_nogrok=non_grokked_object.modelclass(**non_grokked_object.modelconfig)
        nogrok_state_dic=non_grokked_object.models[epoch]['model']
        result=model_nogrok.load_state_dict(nogrok_state_dic,strict=False)
        if len(result.missing_keys)>1 or len(result.unexpected_keys)>1:
            print(result.missing_keys,result.unexpected_keys)

        activations_nogrok, output_nogrok, cleanup_nogrok = get_activations(model_nogrok,images_tensor)
        sorted_activations_nogrok,acts_indices_nogrok=get_acts_dict(single_run=non_grokked_object,dataset=dataset,epoch=epoch,sortby=sortby)

        #removing 
        sorted_activations_nogrok.pop('conv_layers.3 (Conv2d)')
        sorted_activations_nogrok.pop('conv_layers.0 (Conv2d)')
        sorted_activations_nogrok.pop('conv_layers.2 (MaxPool2d)')
        sorted_activations_nogrok.pop('conv_layers.5 (MaxPool2d)')
        sorted_activations_nogrok.pop('fc_layers.3 (Linear)')
        activations_nogrok.pop('conv_layers.3 (Conv2d)')
        activations_nogrok.pop('conv_layers.0 (Conv2d)')
        activations_nogrok.pop('conv_layers.2 (MaxPool2d)')
        activations_nogrok.pop('conv_layers.5 (MaxPool2d)')
        activations_nogrok.pop('fc_layers.3 (Linear)')
        acts_indices_nogrok.pop('conv_layers.2 (MaxPool2d)')
        acts_indices_nogrok.pop('conv_layers.5 (MaxPool2d)')
        acts_indices_nogrok.pop('fc_layers.3 (Linear)')
        acts_indices_nogrok.pop('conv_layers.3 (Conv2d)')
        acts_indices_nogrok.pop('conv_layers.0 (Conv2d)')

        #Now you need to populate the layers
        feature_dim=features_tensor.shape[1]
        if fig==None:
            fig=make_subplots(rows=max(4,2*feature_dim),cols=1+len(sorted_activations_grok.keys()),subplot_titles=['Grok loss']+[f'ENERGY, layer{key}' for key in sorted_activations_grok.keys()]+['No grok loss']+[f'ENERGY, layer {key}' for key in sorted_activations_grok.keys()]+['Grok accuracy']+[f'MAG, Layer {key}' for key in sorted_activations_grok.keys()]+['No grok accuracy']+[f'MAG, Layer {key}' for key in sorted_activations_grok.keys()])#(len(grok_weights)+4)//2

        #titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_nogrok))]

        fig.add_trace(go.Scatter(x=list(range(len(self.train_losses))),y=self.train_losses,marker=dict(color='black'),showlegend=True,name='Train'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.test_losses))),y=self.test_losses,marker=dict(color='orange'),showlegend=True,name='Test'),row=1,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_losses), max(self.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_losses))),y=non_grokked_object.train_losses,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_losses))),y=non_grokked_object.test_losses,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=2,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_losses), max(non_grokked_object.test_losses)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)



        fig.add_trace(go.Scatter(x=list(range(len(self.test_accuracies))),y=self.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=3,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(self.train_accuracies))),y=self.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=3,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(self.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=3, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.test_accuracies))),y=non_grokked_object.train_accuracies,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=4,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(non_grokked_object.train_accuracies))),y=non_grokked_object.test_accuracies,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=4,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(non_grokked_object.train_accuracies), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=4, col=1)

        if epoch==self.model_epochs()[0]:
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
            if epoch==self.model_epochs()[0]:
                fig.update_xaxes(title_text="Weight", row=1, col=count+1)
                fig.update_xaxes(title_text="Weight", row=2, col=count+1)
            count+=1
        return fig
#Object for storing all the runs within a given param
class seed_run_container():
    def __init__(self):
        # params_dic={'weight_decay':weight_decay,'weight_multiplier':weight_multiplier,'learning_rate':learning_rate,'hidden_layers':hiddenlayers,'conv_channels':conv_channels,'train_size':train_size,'test_size':test_size,'dropout_p':dropout_prob}
        self.params_dic={}#Rememeber that I want an object that holds different seeds of the same params
        self.runs_dic={}#
    
    def aggregate_runs(self,folder):
        onlyfiles = [f for f in os.listdir(folder) if 'time_' in f]
        #extract the trainargs object and the params_dic
        for runfile in onlyfiles:
            print(runfile)
            if os.path.getsize(str(folder)+'/'+runfile) > 0:  # Checks if the file is not empty
                with open(str(folder)+'/'+runfile, 'rb') as file:
                    try:
                        runobj = dill.load(file)
                        # Proceed with using runobj
                    except EOFError:
                        print("Failed to load the object. The file may be corrupted or incomplete.")
            else:
                print("The file is empty.")

            
            with open(str(folder)+'/'+runfile,'rb') as file:
                runobj=dill.load(file)
            keys_to_ignore=['data_seed','sgd_seed','init_seed']
            
            filtered_dict1 = {k: v for k, v in runobj.params_dic.items() if k not in keys_to_ignore}
            filtered_dict2 = {k: v for k, v in self.params_dic.items() if k not in keys_to_ignore}


            if filtered_dict1==filtered_dict2:
                key0=(runobj.data_seed,runobj.sgd_seed,runobj.init_seed)
                if key0 in self.runs_dic.keys():
                    if runobj==self.runs_dic[key0]:
                        pass
                    else:
                        print('duplicate non-identical runs!')
                else:
                    self.runs_dic[(runobj.data_seed,runobj.sgd_seed,runobj.init_seed)]=runobj
            
    def create_average_run(self,fixed_seeds):
        # self.models={} #You'll save your models here
        # self.train_losses=None
        # self.test_losses=None
        # self.train_accuracies=None
        # self.test_accuracies=None
        averaged_attributes=['train_losses','test_losses','train_accuracies','test_accuracies']
        if fixed_seeds==None:
            avg_run=seed_average_onerun(data_seed=None,sgd_seed=None,init_seed=None,params_dic=self.params_dic)
            
            for seed_key in self.runs_dic.keys():
                run_obj=self.runs_dic[seed_key]
                for attribute in averaged_attributes:
                    pass

        return None
    
    def make_weights_histogram2(self,non_grokked_container,epoch,fig):
        #last_epoch=max(self.models.keys())
        avg_weights_grok=np.array([])
        avg_weights_nogrok=np.array([])
        first=True
        for seed_key in self.runs_dic.keys():
            grokked_run=self.runs_dic[seed_key]
            grok_state_dic=grokked_run.models[epoch]['model']

            non_grokked_object=non_grokked_container.runs_dic[seed_key]
            nogrok_state_dic=non_grokked_object.models[epoch]['model']
            grok_weights=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
            nogrok_weights=[nogrok_state_dic[key] for key in nogrok_state_dic.keys() if 'weight' in key]
            if first:
                avg_weights_grok=grok_weights
                avg_weights_nogrok=nogrok_weights
            else:
                avg_weights_grok=[avg_weights_grok[i]+grok_weights[i] for i in range(len(avg_weights_grok))]
                avg_weights_nogrok=[avg_weights_nogrok[i]+nogrok_weights[i] for i in range(len(avg_weights_nogrok))]
            
            
        #titles=['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_nogrok))]
        avg_weights_grok=[avg_weights_grok[i]/(len(self.runs_dic.keys())) for i in range(len(avg_weights_grok))]
        avg_weights_nogrok=[avg_weights_nogrok[i]/(len(self.runs_dic.keys())) for i in range(len(avg_weights_nogrok))]
        #add loss curves
        avg_test_loss_grok=np.mean(np.array([np.array(self.runs_dic[seed_key].test_losses) for seed_key in self.runs_dic.keys()]),axis=0)
        avg_train_loss_grok=np.mean(np.array([np.array(self.runs_dic[seed_key].train_losses) for seed_key in self.runs_dic.keys()]),axis=0)
        avg_test_loss_nogrok=np.mean(np.array([np.array(non_grokked_container.runs_dic[seed_key].test_losses) for seed_key in self.runs_dic.keys()]),axis=0)
        avg_train_loss_nogrok=np.mean(np.array([np.array(non_grokked_container.runs_dic[seed_key].train_losses) for seed_key in self.runs_dic.keys()]),axis=0)


        avg_test_accuracies_grok=np.mean(np.array([np.array(self.runs_dic[seed_key].test_accuracies) for seed_key in self.runs_dic.keys()]),axis=0)
        avg_train_accuracies_grok=np.mean(np.array([np.array(self.runs_dic[seed_key].train_accuracies) for seed_key in self.runs_dic.keys()]),axis=0)
        avg_test_accuracies_nogrok=np.mean(np.array([np.array(non_grokked_container.runs_dic[seed_key].test_accuracies) for seed_key in self.runs_dic.keys()]),axis=0)
        avg_train_accuracies_nogrok=np.mean(np.array([np.array(non_grokked_container.runs_dic[seed_key].train_accuracies) for seed_key in self.runs_dic.keys()]),axis=0)
        print(len(avg_test_accuracies_grok))

        fig.add_trace(go.Scatter(x=list(range(len(avg_train_loss_grok))),y=avg_train_loss_grok,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=1,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(avg_test_loss_grok))),y=avg_test_loss_grok,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=1,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(avg_train_loss_grok), max(avg_test_loss_grok)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=1)

        fig.add_trace(go.Scatter(x=list(range(len(avg_train_loss_nogrok))),y=avg_train_loss_nogrok,marker=dict(color='black'),showlegend=False,name='Grok Train'),row=2,col=1)
        fig.add_trace(go.Scatter(x=list(range(len(avg_test_loss_nogrok))),y=avg_test_loss_nogrok,marker=dict(color='orange'),showlegend=False,name='Grok Test'),row=2,col=1)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(avg_train_loss_nogrok), max(avg_test_loss_nogrok)],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=1)


        fig.add_trace(go.Scatter(x=list(range(len(avg_train_accuracies_grok))),y=avg_train_accuracies_grok,marker=dict(color='black'),showlegend=True,name='Grok Train'),row=1,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(avg_test_accuracies_grok))),y=avg_test_accuracies_grok,marker=dict(color='orange'),showlegend=True,name='Grok Test'),row=1,col=2)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(avg_test_accuracies_grok), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=1, col=2)

        fig.add_trace(go.Scatter(x=list(range(len(avg_train_accuracies_nogrok))),y=avg_train_accuracies_nogrok,marker=dict(color='black'),showlegend=True,name='No Grok Train'),row=2,col=2)
        fig.add_trace(go.Scatter(x=list(range(len(avg_test_accuracies_nogrok))),y=avg_test_accuracies_nogrok,marker=dict(color='orange'),showlegend=True,name='No Grok Test'),row=2,col=2)
        fig.add_trace(go.Scatter(x=[epoch, epoch], y=[min(avg_test_accuracies_nogrok), 1],mode="lines", line=dict(color="green",dash='dash'), showlegend=False),row=2, col=2)
        
        run0=list(self.runs_dic.values())[0]
        if epoch==run0.model_epochs()[0]:
            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_yaxes(title_text="Loss",type='log', row=1, col=1)
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_yaxes(title_text="Loss", type='log',row=2, col=1)

            fig.update_xaxes(title_text="Epoch", row=1, col=1)
            fig.update_yaxes(title_text="Accuracy", row=1, col=2)
            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_yaxes(title_text="Accuracy", row=2, col=2)


        for i in range(len(avg_weights_grok)):
            flattened_gw=torch.flatten(avg_weights_grok[i]).detach().numpy()
            flattened_ngw=torch.flatten(avg_weights_nogrok[i]).detach().numpy()
            showleg=False
            if i==0:
                showleg=True
            #fig.add_trace(go.Histogram(x=flattened_gw,marker=dict(color='red'),showlegend=showleg,name='Grok'),row=1,col=i+3)
            #fig.add_trace(go.Histogram(x=flattened_ngw,marker=dict(color='blue'),showlegend=showleg,name='No Grok'),row=2,col=i+3)

            fig.add_trace(go.Histogram(x=[epoch], name=f'Grok_Epoch_{epoch}'),row=1,col=i+2)  # Placeholder histogram
            fig.add_trace(go.Histogram(x=[epoch + 0.5], name=f'NoGrok_Epoch_{epoch}'),row=2,col=i+2)  # Placeholder histogram
            if epoch==run0.model_epochs()[0]:
                fig.update_xaxes(title_text="Weight", row=1, col=i+2)
                fig.update_yaxes(title_text="Freq", row=1, col=i+2)
                fig.update_xaxes(title_text="Weight", row=2, col=i+2)
                fig.update_yaxes(title_text="Freq", row=2, col=i+2)

    
    #Now I want to write scripts for the analysis function.


    def weights_histogram_epochs2(self,non_grokked_container):
        run0=list(self.runs_dic.values())[0]
        epochs=[run0.model_epochs()[0]]+[run0.model_epochs()[5]]#+run0.model_epochs()[10:350][0::50]+[run0.model_epochs()[-1]]
        run0_ng=list(non_grokked_container.runs_dic.values())[0]
        ng_epochs=run0_ng.model_epochs()
        if epochs!=ng_epochs:
            print('Grokked and non-grokked epochs not the same!')
        grok_state_dic=run0.models[epochs[0]]['model']
        weights_grok=[grok_state_dic[key] for key in grok_state_dic.keys() if 'weight' in key]
        titles=['Grok Loss']+['Grok Accuracy']+[f'Grok Layer {i}' for i in range(len(weights_grok))]+['No Grok Loss']+['No Grok Accuracy']+[f'No Grok Layer {i}' for i in range(len(weights_grok))]
        fig=make_subplots(rows=2,cols=len(weights_grok)+2,subplot_titles=titles)
        for epoch in epochs:
            self.make_weights_histogram2(non_grokked_container,epoch,fig)
        
        total_plots=2*(6)+2*(len(weights_grok))
        print(len(fig.data))
        for i in range(len(fig.data)):
            fig.data[i].visible=False
        for i in range(total_plots):
            fig.data[i].visible=True
        #Now update the slider
        steps = []
        
        for i in range(len(epochs)):
            step = dict(
                method = 'restyle',
                args = ['visible',[False] * len(fig.data)],
            )
            for j in range(total_plots):
                step['args'][1][total_plots*i+j] = True
            steps.append(step)
        
        sliders = [dict(
            steps = steps,
        )]
        fig.layout.sliders = sliders
        fig.show()


# def get_activations(model, x):
#     activations = {}
#     hooks = []

#     def save_activation(name):
#         """Hook function that saves the output of the layer to the activations dict."""
#         def hook(model, input, output):
#             activations[name] = output.detach()
#         return hook

#     def register_hooks():
#         """Registers hooks on specified layer types across the entire model."""
#         print('names')
#         for name, module in model.named_modules():
#             print(f'name, module {name}, {module}')
#             # Check specifically for the layer types or names you are interested in
#             if isinstance(module, (nn.Conv2d, nn.Linear,nn.MaxPool2d)):
#                 # Adjust name to include full path for clarity, especially useful for layers within ModuleList
#                 full_name = f'{name} ({module.__class__.__name__})'
#                 print(f'Registering hook on {full_name}')
#                 hook = module.register_forward_hook(save_activation(full_name))
#                 hooks.append(hook)
#             #Have to manually add the hook for pooling layer 5
#         # full_name = f'conv_layers.2 ({model.conv_layers[2].__class__.__name__})'
#         # hook=model.conv_layers[2].register_forward_hook(save_activation(full_name))
#         # full_name = f'conv_layers.5 ({model.conv_layers[5].__class__.__name__})'
#         # hook=model.conv_layers[5].register_forward_hook(save_activation(full_name))

#         #     # Explicitly check if this is the layer you are particularly interested in
#         # if name == "conv_layers.5":
#         #     print(f"Special hook registered on {full_name}")

#     def remove_hooks():
#         """Removes all hooks from the model."""
#         for hook in hooks:
#             hook.remove()
#         print('All hooks removed.')

#     register_hooks()
#     # Forward pass to get outputs
#     output = model(x)

#     return activations, output, remove_hooks

def generate_test_set(dataset,size):
    dtype=torch.float32
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

#activations functions



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



def test_image_func(image_tensor,func,func_name,samples,feature_tensor):
    if func!=None and feature_tensor==None:
        result_tensor=func(image_tensor)
    else:
        result_tensor=feature_tensor
    indices=[random.randint(0, len(image_tensor)) for _ in range(samples)]
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




#ModAdd Fourier activations

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
