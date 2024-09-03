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

# machine learning imports

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import copy
import inspect
import itertools
#import kaleido

################Seeds
from cluster.data_objects import seed_average_onerun#I think you need the cluster. when you set the PYTHONPATH to be Code
import functools
from plotly.graph_objects import FigureWidget
import cluster.dynamic_plot

#grokfast attempt
#from grokfast import gradfilter_ma, gradfilter_ema




@dataclass
class TrainArgs:
	epochs: int
	lr: float
	weight_decay: float
	weight_multiplier: float
	dropout_prob: float
	data_seed: int | None = None
	sgd_seed: int | None = None
	init_seed: int | None = None
	device: str = "cpu"
	grok: str = False
	test_size: int = 1000
	train_size: int = 100
	hiddenlayers: list = None
	conv_channels: list = None
	train_fraction: float = 0.5
	P: int = 97
	loss_criterion: str = "MSE"
	batch_size: int = 64
	activation: any=nn.ReLU
	


def create_model(init_seed,model_type):
	random.seed(init_seed)
	for set_seed in [torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
		set_seed(init_seed)

	if model_type=="CNN":
	# initialize model and print details
		model = CNN(input_dim=16, output_size=2, input_channels=1, conv_channels=conv_channels, hidden_widths=hiddenlayers,
						activation=nn.ReLU(), optimizer=torch.optim.SGD,
						learning_rate=learning_rate, weight_decay=weight_decay, multiplier=weight_multiplier, dropout_prob=dropout_prob)
	elif model_type=="ModMLP":
		model=MLP(P=P,hidden=hiddenlayers,learning_rate=learning_rate,bias=True,weight_decay=weight_decay,weight_multiplier=weight_multiplier,optimizer=optimizer_fn)
		
	
	print(model)
	# Define loss function
	criterion = nn.CrossEntropyLoss()
	# Define optimizer for stochastic gradient descent (including learning rate and weight decay)
	# use the one I defined as an attribute in the CNN class
	optimizer = model.optimizer
	return model

def check_weights_update(initial_weights, updated_weights):
	return not torch.equal(initial_weights, updated_weights)

#Functions for plotting as you train:
import plotly.io as pio
import webbrowser
import subprocess
import http.server
import socketserver
import os
import time
import webbrowser
from threading import Thread


def start_server():
    PORT = 8000
    os.chdir(os.path.dirname(os.path.realpath(__file__)))  # Change directory to where the HTML file is located
    Handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print("Server running at localhost:{}".format(PORT))
        httpd.serve_forever()

# Function to open the browser window
def open_browser():
    time.sleep(2)  # Give the server some time to start
    webbrowser.open_new("http://localhost:8000/updated_plot.html")

def plot_traincurves(epochs,test_accuracies,train_accuracies,test_losses,train_losses,config_dict):
	fig=make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
		
	test_acc_trace = go.Scatter(x=epochs, y=test_accuracies, mode='lines', name='Test accuracy', line=dict(color='red',dash='solid'))
	train_acc_trace = go.Scatter(x=epochs, y=train_accuracies, mode='lines', name='Train accuracy', line=dict(color='blue',dash='dash'))
	
	test_loss_trace = go.Scatter(x=epochs, y=test_losses, mode='lines', name='Test loss', line=dict(color='red',dash='solid'))
	train_loss_trace = go.Scatter(x=epochs, y=train_losses, mode='lines', name='Train loss', line=dict(color='blue',dash='dash'))

	# Add traces to the figure
	fig.add_trace(test_acc_trace, row=1, col=1)
	fig.add_trace(train_acc_trace, row=1, col=1)

	fig.add_trace(test_loss_trace, row=1, col=2)
	fig.add_trace(train_loss_trace, row=1, col=2)
	

	fig.update_xaxes(title_text="Epoch")
	fig.update_yaxes(title_text="Accuracy", row=1, col=1)
	fig.update_yaxes(title_text="Loss", row=1, col=2,type='log')
	
	fig.update_layout(title_text=f'mlp {config_dict["hidden"]}, wd {config_dict["weight_decay"]}, lr {learning_rate}, multiplier {config_dict["weight_multiplier"]}, train frac {train_fraction}, optimizer {str(config_dict["optimizer"])}') 
	
	#,  )

	return fig

class TrainingPlot:
	def __init__(self, plot_enabled=False):
		self.plot_enabled = plot_enabled
		

		self.fig=make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "Loss"))
		if self.plot_enabled:
			# Initialize traces for loss and accuracy
			self.test_acc_trace = go.Scatter(x=[], y=[], mode='lines', name='Test accuracy', line=dict(color='red',dash='solid'))
			self.train_acc_trace = go.Scatter(x=[], y=[], mode='lines', name='Train accuracy', line=dict(color='blue',dash='dash'))
			
			self.test_loss_trace = go.Scatter(x=[], y=[], mode='lines', name='Test loss', line=dict(color='red',dash='solid'))
			self.train_loss_trace = go.Scatter(x=[], y=[], mode='lines', name='Train loss', line=dict(color='blue',dash='dash'))

			# Add traces to the figure
			self.fig.add_trace(self.test_acc_trace, row=1, col=1)
			self.fig.add_trace(self.train_acc_trace, row=1, col=1)

			self.fig.add_trace(self.test_loss_trace, row=1, col=2)
			self.fig.add_trace(self.train_loss_trace, row=1, col=2)
			

			self.fig.update_xaxes(title_text="Epoch")
			self.fig.update_yaxes(title_text="Accuracy", row=1, col=1)
			self.fig.update_yaxes(title_text="Loss", row=1, col=1)
			self.fig=FigureWidget(self.fig)
			webbrowser.open('http://localhost:8000')
			pio.write_html(self.fig, 'updated_plot.html')
			# dynamic_plot.start_http_server('updated_plot.html')
			# dynamic_plot.display_html_in_browser()
			dynamic_plot.update_html_content()
			
			# server_thread = Thread(target=start_server)
			# server_thread.daemon = True  # Allow the program to exit even if this thread is still running
			# server_thread.start()
			# browser_thread = Thread(target=open_browser)
			# browser_thread.start()

			
	def update(self, epochs,test_accs,train_accs,test_losses,train_losses,first):
		if self.plot_enabled:
			# Append new data points to the traces
			self.test_acc_trace['x'] = epochs
			self.test_acc_trace['y'] = test_accs
			self.train_acc_trace['x'] = epochs
			self.train_acc_trace['y'] = train_accs
			
			self.test_loss_trace['x'] = epochs
			self.test_loss_trace['y'] = test_losses
			self.train_acc_trace['x'] = epochs
			self.train_loss_trace['y'] = train_losses
			# Update the figure with new trace data
			self.fig.update_traces(selector=dict(name='Test accuracy'), x=self.test_acc_trace['x'], y=self.test_acc_trace['y'])
			self.fig.update_traces(selector=dict(name='Train accuracy'), x=self.train_acc_trace['x'], y=self.train_acc_trace['y'])

			self.fig.update_traces(selector=dict(name='Test loss'), x=self.test_loss_trace['x'], y=self.test_loss_trace['y'])
			self.fig.update_traces(selector=dict(name='Train Loss'), x=self.train_loss_trace['x'], y=self.train_loss_trace['y'])
			
			pio.write_html(self.fig, 'updated_plot.html')
			
			# dynamic_plot.refresh_browser_window()

			dynamic_plot.update_html_content()
			# if first:
			# 	browser=webbrowser.get()
			# 	browser.open('file://' + os.path.realpath('updated_plot.html'))
			# else:
			# 	browser.open('file://' + os.path.realpath('updated_plot.html'), new=0)
			#subprocess.call(['open', 'updated_plot.html'])


	def close(self):
			if self.plot_enabled:
				pass  # Plotly automatically handles closing the plot

def extract_top_percent(tensor, percent):
    # Ensure the tensor is flattened
    tensor = tensor.view(-1)
    
    # Calculate the number of elements to keep (top percent)
    num_elements = int(percent * len(tensor))
    
    # Use torch.topk to get the top elements
    top_values, _ = torch.topk(tensor, num_elements, sorted=False,largest=True)
    
    return top_values


def calculate_ipr(model,r,weight_share=1):
	with torch.no_grad():
		flat_parameters = [param.view(-1) for param in model.parameters()]
		flat_weights = torch.cat(flat_parameters)
		#flat_weights=extract_top_percent(flat_weights,weight_share)
		ipr_denom=torch.sqrt(torch.sum(flat_weights**2))**(2*r)
		ipr_num=torch.sum(np.abs(flat_weights)**(2*r))
		ipr=ipr_num/ipr_denom
		return ipr.item()

def calculate_weight_norm(model,n):
	with torch.no_grad():
		flat_parameters = [param.view(-1) for param in model.parameters()]
		flat_weights = torch.cat(flat_parameters)
		weight_norm=torch.sum(flat_weights**n)
		return weight_norm

def calculate_cosine_similarity(model1,model2,weight_keys=None):
	cosine_tensor_list=[]
	names_list=[]
	with torch.no_grad():
		def prep_models(model):
			list_of_weights=[]
			list_of_names=[]
			for name, param in model.named_parameters():
				if param.requires_grad and len(param.shape)>1:
					list_of_weights.append(param)
					list_of_names.append(name)
			
			flat_weights=torch.cat([(torch.flatten(p)) for p in list_of_weights])

			return list_of_weights,flat_weights,list_of_names
		
		list_of_weights_1,flattened_weights_1,weight_keys_1=prep_models(model1)
		list_of_weights_2,flattened_weights_2,weight_keys_2=prep_models(model2)
		

		flattened_cosine=nn.CosineSimilarity(dim=0,eps=1e-6)(flattened_weights_1,flattened_weights_2)
		cosine_tensor_list.append(flattened_cosine.item())
		names_list.append('all model weights flattened')
		if weight_keys!=None:
			for weight_key in weight_keys:
				index=weight_keys_1.index(weight_key)
				cosine=nn.CosineSimilarity(dim=-1,eps=1e-6)(torch.flatten(list_of_weights_1[index]),torch.flatten(list_of_weights_2[index]))
				cosine_tensor_list.append(cosine.item())
				names_list.append(weight_key)
		cosine_tensor=torch.Tensor(cosine_tensor_list)
		
		return cosine_tensor,names_list


def linear_decomposition(model,criterion,data_loader):
	with torch.no_grad():
		copied_model=copy.deepcopy(model)
		for batch in data_loader:
			loss_full= 0.0
			loss_linear = 0.0
			loss_non_linear = 0.0

			linear_norms=0.0
			non_linear_norms=0.0
			diff_norms=0.0


			X_test,y_test=batch
			y_pred_full=copied_model(X_test).to(device)

			non_linear_y_pred=copied_model(X_test).to(device)+copied_model(-X_test).to(device)
			linear_y_pred=y_pred_full-non_linear_y_pred


			float_y=y_test.float().clone().to(device)
			full_loss = criterion(y_pred_full, float_y)
			loss_full += full_loss.item()

			non_linear_loss=criterion(non_linear_y_pred,float_y)
			loss_non_linear += non_linear_loss.item()

			linear_loss=criterion(linear_y_pred,float_y)
			loss_linear += linear_loss.item()

			linear_norm=torch.sqrt(torch.sum(linear_y_pred**2)).item()
			linear_norms+=linear_norm

			non_linear_norm=torch.sqrt(torch.sum(non_linear_y_pred**2)).item()
			non_linear_norms+=non_linear_norm

			diff_norm=torch.sqrt(torch.sum((linear_y_pred-non_linear_y_pred)**2)).item()
			diff_norms+=diff_norm

	return loss_full,loss_linear,loss_non_linear,linear_norms,non_linear_norms,diff_norms
		
		
			
		
		


def train(epochs,initial_model,save_interval,train_loader,test_loader,sgd_seed,batch_size,one_run_object,loss_criterion,train_type,config_dict,plot_as_train=False):
	plot_interval=50
	start_time = timer()
	first_time_training = True
	epochs = epochs # how many runs through entire training data
	save_models=True
	save_interval=save_interval
	fix_norm=False
	model=initial_model
	start=time.time()
	print(f'l2 norm: {calculate_weight_norm(model,2)}')
	end=time.time()
	print(f'time to calculate l2 norm: {end-start}')
	break_epoch=None
	second_lr=10e-3
	done=False
	compare_models=50
	#training_plot = TrainingPlot(plot_as_train)
	#os.open('dynamic_plot.html')
	
		
	if loss_criterion=="MSE":
		criterion = nn.MSELoss()
	elif loss_criterion=="CrossEntropy":
		criterion = nn.CrossEntropyLoss()
	optimizer = model.optimizer
	random.seed(sgd_seed)
	for set_seed in [torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
		set_seed(sgd_seed)
		
	if save_models:
		save_dict0 = {'model':model.state_dict()}#'train_data':train_data, 'test_data':test_data <-- I don't need this because I'll have this saved in the data. 
		one_run_object.models[0]=save_dict0
		#torch.save(save_dict, root+'/'+run_name+'/'+'/0.pth')

	if  first_time_training == True:
		train_losses = []
		test_losses = []
		train_accuracy = []
		test_accuracy = []
		iprs=[]
		norms=[]
		cosines=[]
		cosine_steps=[]
		linear_decomposition_dict={'full_loss_test':[],'linear_loss_test':[],'non_linear_loss_test':[],'linear_norm_test':[],'non_linear_norm_test':[],'diff_norm_test':[],
		'full_loss_train':[],'linear_loss_train':[],'non_linear_loss_train':[],'linear_norm_train':[],'non_linear_norm_train':[],'diff_norm_train':[]}
	else:
		print("Starting additional training")
		epochs = 2900
	if fix_norm:
		norm=np.sqrt(sum(param.pow(2).sum().item() for param in model.parameters()))
	#for grokfast

	grads = None
	print(f'grads = {grads}')

	back_compare=[]
	back_compare_steps=[]
	for i in tqdm(range(epochs)):
		train_correct = 0
		test_correct = 0

		# Run the training batches
		

			# Apply the current model to make prediction of training data

			# Flatten input tensors to two index object with shape (batch_size, input_dims) using .view()
			# Predict label probabilities (y_train) based on current model for input (X_train)
		
		if compare_models!=None:
			if i<compare_models:
				back_compare.append(copy.deepcopy(model))
			if i>=compare_models:
				back_compare.append(copy.deepcopy(model))
			#back_compare=[copy.deepcopy(model)]
		if train_type=="Ising":
			for batch, (X_train, y_train) in enumerate(train_loader):
				y_pred = model(X_train.view(batch_size, 1, 16, 16).to(device))#note- data dimension set to number of points, 1 only one channel, 16x16 for each data-point. Model transforms 2d array into 3d tensors with 4 channels
				predicted = torch.max(y_pred.data, 1)[1]
				train_correct += (predicted == y_train.to(device)).sum().item()
				train_loss = criterion(y_pred, y_train.to(device))

				optimizer.zero_grad() # clears old gradients stored in previous step
				train_loss.backward() # calculates gradient of loss function using backpropagation (stochastic)
				optimizer.step()

				predicted = torch.max(y_pred.data, 1)[1]
				train_losses.append(train_loss.item())
				train_accuracy.append(train_correct/train_size)
				
		elif train_type=="Mod":
			train_loss = 0.0
			train_acc = 0.0
			full_loss_test,linear_loss_test,non_linear_loss_test,linear_norm_test,non_linear_norm_test,diff_norm_test=linear_decomposition(model,criterion,test_loader)
			linear_decomposition_dict['full_loss_test'].append(full_loss_test)
			linear_decomposition_dict['linear_loss_test'].append(linear_loss_test)
			linear_decomposition_dict['non_linear_loss_test'].append(non_linear_loss_test)
			linear_decomposition_dict['linear_norm_test'].append(linear_norm_test)
			linear_decomposition_dict['non_linear_norm_test'].append(non_linear_norm_test)
			linear_decomposition_dict['diff_norm_test'].append(diff_norm_test)

			full_loss_train,linear_loss_train,non_linear_loss_train,linear_norm_train,non_linear_norm_train,diff_norm_train=linear_decomposition(model,criterion,train_loader)
			linear_decomposition_dict['full_loss_train'].append(full_loss_train)
			linear_decomposition_dict['linear_loss_train'].append(linear_loss_train)
			linear_decomposition_dict['non_linear_loss_train'].append(non_linear_loss_train)
			linear_decomposition_dict['linear_norm_train'].append(linear_norm_train)
			linear_decomposition_dict['non_linear_norm_train'].append(non_linear_norm_train)
			linear_decomposition_dict['diff_norm_train'].append(diff_norm_train)


			for batch in train_loader:
				X_train,y_train=batch
				optimizer.zero_grad()
				
				y_pred=model(X_train).to(device)
				y_train=y_train.float().to(device)
				loss = criterion(y_pred, y_train.to(device))
				loss.backward()
				#grads = gradfilter_ema(model, grads=grads, alpha=0.8, lamb=1.0)
				optimizer.step()
				train_loss += loss.item()
				
				# print('now')
				# print(break_epoch)
				# print(done)
				
				if break_epoch!=None and i>break_epoch and done==False:
					print(f'optimizer learning rate before change {model.optimizer.param_groups[0]["lr"]}')
					model.optimizer.param_groups[0]['lr'] = second_lr
					print(f'optimizer learning rate after change {model.optimizer.param_groups[0]["lr"]}')
					done=True
					
				if loss_criterion=='MSE':
					train_acc += (y_pred.argmax(dim=1) == y_train.argmax(dim=1)).sum().item()
				else:
					train_acc += (y_pred.argmax(dim=1) == y_train.argmax(dim=1)).sum().item()
				if fix_norm:
					with torch.no_grad():
						new_norm = np.sqrt(sum(param.pow(2).sum().item() for param in model.parameters()))
						for param in model.parameters():
							param.data *= norm / new_norm
			
			train_loss /= len(train_loader)
			train_losses.append(train_loss)
			train_acc /= len(train_dataset)
			train_accuracy.append(train_acc)
					
		model_ipr2=calculate_ipr(model,2,1)
		model_ipr4=calculate_ipr(model,4,1)
		model_ipr_05=calculate_ipr(model,0.5,1)
		iprs.append([model_ipr2,model_ipr4,model_ipr_05])
		l2norm=calculate_weight_norm(model,2)
		norms.append(l2norm)
		if i>compare_models:
			cosines_tensor=calculate_cosine_similarity(back_compare[0],model,weight_keys=['model.0.weight','model.2.weight'])[0]
			cosines.append(cosines_tensor)
			back_compare.pop(0)

	

			# Assume `model_template` is an instance of the model's class (same architecture)
			back_steps_model = copy.deepcopy(back_compare[0])  # Deep copy to preserve the architecture

			# Subtract the parameters
			with torch.no_grad():  # Disable gradient computation as we're manually updating parameters
				for p_new, p0, p1 in zip(back_steps_model.parameters(), back_compare[1].parameters(), back_compare[0].parameters()):
					p_new.copy_(p0 - p1)
			
			front_steps_model = copy.deepcopy(back_compare[-1])  # Deep copy to preserve the architecture

			# Subtract the parameters
			with torch.no_grad():  # Disable gradient computation as we're manually updating parameters
				for p_new, p0, p1 in zip(front_steps_model.parameters(), model.parameters(), back_compare[-1].parameters()):
					p_new.copy_(p0 - p1)

			cosine_steps_tensor=calculate_cosine_similarity(back_steps_model,front_steps_model,weight_keys=['model.0.weight','model.2.weight'])[0]			
			cosine_steps.append(cosine_steps_tensor)



		
			

			# Tally the number of correct predictions per epoch
			
			


		# Update overall train loss (most recent batch) & accuracy (of all batches) for the epoch


		# Run the testing batches
		with torch.no_grad():
			if train_type=="Ising":
				for batch, (X_test, y_test) in enumerate(test_loader):
						# Apply the model
						y_val = model(X_test.view(test_size, 1, 16, 16).to(device))

						# Tally the number of correct predictions
						predicted = torch.max(y_val.data, 1)[1]
						test_correct += (predicted == y_test.to(device)).sum().item()
						test_loss = criterion(y_val, y_test.to(device))
						test_losses.append(test_loss.item())
						test_accuracy.append(test_correct/test_size)
			elif train_type=="Mod":
				test_loss = 0.0
				test_acc = 0.0
				for batch in test_loader:
					X_test,y_test=batch
					y_val=model(X_test).to(device)
					float_y=y_test.float().clone().to(device)
					loss = criterion(y_val, float_y.to(device))
					test_loss += loss.item()
					if loss_criterion=='MSE':
						test_acc += (y_val.argmax(dim=1) == float_y.argmax(dim=1)).sum().item()
					else:
						test_acc += (y_val.argmax(dim=1) == float_y.argmax(dim=1)).sum().item()
				
				test_loss /= len(test_loader)
				test_losses.append(test_loss)
				test_acc /= len(test_dataset)
				test_accuracy.append(test_acc)


		#epochs_plot=list(range(len(test_accuracy)))
		#first=True
		# if i%plot_interval==0:
		# 	training_plot.update(epochs_plot, test_accuracy,train_accuracy,test_losses,train_losses,first)
		# 	first=False
					


		# Update test loss & accuracy for the epoch


		
		if save_models and i%save_interval==0:
			# updated_fc_weights = model.fc_layers[0].weight.data
			# if check_weights_update(initial_fc_weights, updated_fc_weights):
			#     print("fc weights have been updated.")
			# else:
			#     print("fc weights have not been updated.")
			
			# print(save_dict0['model']['fc_layers.3.weight']==model.state_dict()['fc_layers.3.weight'])
			save_dict = {
					'model': copy.deepcopy(model.state_dict()),
					'optimizer': copy.deepcopy(optimizer.state_dict()),
					# 'scheduler': scheduler.state_dict(),
					'train_loss': train_loss,
					'test_loss': test_loss,
					'epoch': i,
				}
			one_run_object.models[i]=save_dict
			
			# if i>0:
			#     print(torch.equal(one_run_object.models[i]['model']['fc_layers.3.weight'],one_run_object.models[i-100]['model']['fc_layers.3.weight']))
			# if i>1000:
			#         fig,axs=plt.subplots(1,2)
			#         axs[0].hist(np.ndarray.flatten(one_run_object.models[i]['model']['fc_layers.0.weight'].detach().numpy()))
			#         axs[1].hist(np.ndarray.flatten(one_run_object.models[0]['model']['fc_layers.0.weight'].detach().numpy()))
			#         plt.show()
			#         exit()

		if i % 50 == 0 or i == epochs-1:
			# Print interim result
			if train_type=="Ising":
				print(f"epoch: {i}, train loss: {train_loss.item():.4f}, accuracy: {train_accuracy[-1]:.4f}")
				print(12*" " + f"test loss: {test_loss.item():.4f}, accuracy: {test_accuracy[-1]:.4f}" )
				print(60*"*")
			elif train_type=="Mod":
				print(f"epoch: {i}, train loss: {train_loss:.4f}, accuracy: {train_accuracy[-1]:.4f}")
				print(12*" " + f"test loss: {test_loss:.4f}, accuracy: {test_accuracy[-1]:.4f}" )
				print(60*"*")

	#training_plot.close()#Not necessary in plotly I think, but just in case
	print(f'\nDuration: {timer() - start_time:.0f} seconds') # print the time elapsed

	one_run_object.train_losses=train_losses
	one_run_object.test_losses=test_losses
	one_run_object.train_accuracies=train_accuracy
	one_run_object.test_accuracies=test_accuracy
	one_run_object.iprs=iprs
	one_run_object.norms=norms
	
	one_run_object.euclidcosine=torch.stack(cosines).numpy()
	one_run_object.euclidcosinesteps=torch.stack(cosine_steps).numpy()

	one_run_object.linear_decomposition=linear_decomposition_dict


	# data = ["data[1] is of form [train_loss, test_loss], [train_acc, test_acc]", [[train_losses, test_losses], [train_accuracy, test_accuracy]]]
	# with open(str(root)+'/'+f"Isinglosses_hw{hiddenlayers}_cnn_{conv_channels}_data_seed_{data_seed}_{int(time.time())}.p", "wb") as handle:
	#     pickle.dump(data, handle)
	# print("data saved")

	


# basic Convolutional Neural Network for input datapoints that are 2D tensors (matrices)
# size of input is (num_datapts x input_channels x input_dim x input_dim)  and requires input_dim % 4 = 0
# Conv, Pool, Conv, Pool, Fully Connected, Fully Connected, ...,  Output
# zero padding is included to ensure same dimensions pre and post convolution
def store_init_args(method):
	@functools.wraps(method)
	def wrapper(self, *args, **kwargs):
		# Store the arguments in a dictionary
		self._init_args = (args, kwargs)
		return method(self, *args, **kwargs)
	return wrapper

class CNN(nn.Module):
	@store_init_args
	def __init__(self, input_dim, output_size, input_channels, conv_channels, hidden_widths,
				activation=nn.ReLU(), optimizer=torch.optim.SGD,
				learning_rate=0.001, weight_decay=0, multiplier=1, dropout_prob=0.01):

		super().__init__()

		# transforms input from input_channels x input_dim x input_dim
		# to out_channels x input_dim x input_dim
		conv1 = nn.Conv2d(in_channels=input_channels, out_channels=conv_channels[0],
									kernel_size=3, stride=1, padding=1)
		conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1],
									kernel_size=3, stride=1, padding=1)
		# divide shape of input_dim by two after applying each convolution
		# since this is applied twice, make sure that input_dim is divisible by 4!!!
		pool = nn.MaxPool2d(kernel_size=2, stride=2)

		# Construct convolution and pool layers
		self.conv_layers = nn.ModuleList()
		for conv_layer in [conv1, conv2]:
			self.conv_layers.append(conv_layer)
			self.conv_layers.append(activation)
			self.conv_layers.append(pool)

		# dropout to apply in FC layers
		self.dropout = nn.Dropout(dropout_prob)

		# construct fully connected layers
		self.fc_layers = nn.ModuleList()
		# flattened size after two convolutions and two poolings of data
		input_size = (input_dim//4)**2 * conv_channels[1]
		for size in hidden_widths:
			self.fc_layers.append(nn.Linear(input_size, size))
			input_size = size  # For the next layer
			self.fc_layers.append(activation)
			self.fc_layers.append(self.dropout)
		# add last layer without activation or dropout
		self.fc_layers.append(nn.Linear(input_size, output_size))

		# multiply weights by overall factor
		if multiplier != 1:
			with torch.no_grad():
				for param in self.parameters():
					param.data = multiplier * param.data

		# use GPU if available
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

		# set optimizer
		self.optimizer = optimizer(params=self.parameters(),
									lr=learning_rate, weight_decay=weight_decay)

	def forward(self, X):
		# convolution and pooling
		for layer in self.conv_layers:
			X = layer(X)
		# fully connected layers
		X = X.view(X.size(0), -1)   # Flatten data for FC layers
		for layer in self.fc_layers:
			X = layer(X)
		return X
	
class CNN_nobias(nn.Module):
	def __init__(self, input_dim, output_size, input_channels, conv_channels, hidden_widths,
				activation=nn.ReLU(), optimizer=torch.optim.SGD,
				learning_rate=0.001, weight_decay=0, multiplier=1, dropout_prob=0.01):

		super().__init__()

		# transforms input from input_channels x input_dim x input_dim
		# to out_channels x input_dim x input_dim
		conv1 = nn.Conv2d(in_channels=input_channels, out_channels=conv_channels[0],
									kernel_size=3, stride=1, padding=1,bias=False)
		conv2 = nn.Conv2d(in_channels=conv_channels[0], out_channels=conv_channels[1],
									kernel_size=3, stride=1, padding=1,bias=False)
		# divide shape of input_dim by two after applying each convolution
		# since this is applied twice, make sure that input_dim is divisible by 4!!!
		pool = nn.MaxPool2d(kernel_size=2, stride=2)

		# Construct convolution and pool layers
		self.conv_layers = nn.ModuleList()
		for conv_layer in [conv1, conv2]:
			self.conv_layers.append(conv_layer)
			self.conv_layers.append(activation)
			self.conv_layers.append(pool)

		# dropout to apply in FC layers
		self.dropout = nn.Dropout(dropout_prob)

		# construct fully connected layers
		self.fc_layers = nn.ModuleList()
		# flattened size after two convolutions and two poolings of data
		input_size = (input_dim//4)**2 * conv_channels[1]
		for size in hidden_widths:
			self.fc_layers.append(nn.Linear(input_size, size,bias=False))
			input_size = size  # For the next layer
			self.fc_layers.append(activation)
			self.fc_layers.append(self.dropout)
		# add last layer without activation or dropout
		self.fc_layers.append(nn.Linear(input_size, output_size,bias=False))

		# multiply weights by overall factor
		if multiplier != 1:
			with torch.no_grad():
				for param in self.parameters():
					param.data = multiplier * param.data

		# use GPU if available
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.to(self.device)

		# set optimizer
		self.optimizer = optimizer(params=self.parameters(),
									lr=learning_rate, weight_decay=weight_decay)

	def forward(self, X):
		# convolution and pooling
		for layer in self.conv_layers:
			X = layer(X)
		# fully connected layers
		X = X.view(X.size(0), -1)   # Flatten data for FC layers
		for layer in self.fc_layers:
			X = layer(X)
		return X



class Square(nn.Module):
	''' 
	Torch-friendly implementation of activation if one wants to use
	quadratic activations a la Gromov (induces faster grokking).
	'''
	def forward(self, x):
		return torch.square(x)

class MLP(nn.Module):
	@store_init_args
	def __init__(self, hidden=[512], P=97, optimizer=torch.optim.SGD, weight_multiplier=1, bias=False, learning_rate=0.01, weight_decay=0.0001):
		super(MLP, self).__init__()
		layers = []
		input_dim = 2 * P
		first = True
		for layer_ind in range(len(hidden)):
			if first:
				layers.append(nn.Linear(input_dim, hidden[layer_ind], bias=bias))
				layers.append(nn.ReLU())
				first = False
			else:
				layers.append(nn.Linear(hidden[layer_ind - 1], hidden[layer_ind], bias=bias))
				layers.append(nn.ReLU())
		layers.append(nn.Linear(hidden[-1], P,bias=bias))
		self.model = nn.Sequential(*layers)

		# if weight_multiplier != 1:#I think this is redundant


		self.optimizer = optimizer(params=self.parameters(), lr=learning_rate, weight_decay=weight_decay)
		self.init_weights()

		with torch.no_grad():
			for param in self.parameters():
				param.data = weight_multiplier * param.data

	def forward(self, x):
		x = self.model(x)
		return x

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight)
				if m.bias is not None:
					nn.init.zeros_(m.bias)



def create_ising_dataset(data_seed,train_size,test_size):
	random.seed(data_seed)
	for set_seed in [torch.manual_seed, torch.cuda.manual_seed_all, np.random.seed]:
		set_seed(data_seed)
	
	if cluster:
		datafilename="../Data/IsingML_L16_traintest.pickle"
	else:
		datafilename="/Users/dmitrymanning-coe/Documents/Research/Grokking/Ising_Code/Data/IsingML_L16_traintest.pickle"
	
	with open(datafilename, "rb") as handle:
		data = pickle.load(handle)
		print(f'Data length: {len(data[1])}')
		print(data[0])
		data = data[1]
	# shuffle data list
	random.shuffle(data)
	# split data into input (array) and labels (phase and temp)
	inputs, phase_labels, temp_labels = zip(*data)
	# for now ignore temp labels
	my_X = torch.Tensor(np.array(inputs)).to(dtype) # transform to torch tensor of FLOATS
	my_y = torch.Tensor(np.array(phase_labels)).to(torch.long) # transform to torch tensor of INTEGERS
	my_y_temp = torch.Tensor(np.array(temp_labels)).to(dtype)
	print(my_X.dtype, my_y.dtype)
	print("Created Ising Dataset")
	
	# manually do split between training and testing data (only necessary if ising data)
	# otherwise use torch.utils.data.Subset to get subset of MNIST data
	train_size, test_size, batch_size = train_size, test_size, train_size
	a, b = train_size, test_size
	train_data = TensorDataset(my_X[b:a+b], my_y[b:a+b]) # Choose training data of specified size
	test_data = TensorDataset(my_X[:b], my_y[:b]) # test
	scramble_snapshot=False

	# load data in batches for reduced memory usage in learning
	train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_data, batch_size=test_size, shuffle=False)
	for b, (X_train, y_train) in enumerate(train_loader):
		print("batch:", b)
		print("input tensors shape (data): ", X_train.shape)
		print("output tensors shape (labels): ", y_train.shape)

	for b, (X_test, y_test) in enumerate(test_loader):
		if scramble_snapshot:
			X_test_a=np.array(X_test)
			X_test_perc=np.zeros((test_size,16,16))
			for t in range(test_size):
				preshuff=X_test_a[t,:,:].flatten()
				np.random.shuffle(preshuff)
				X_test_perc[t,:,:]=np.reshape(preshuff,(16,16))
			X_test=torch.Tensor(X_test_perc).to(dtype)
		print("batch:", b)
		print("input tensors shape (data): ", X_test.shape)

		print("output tensors shape (labels): ", y_test.shape)



	# desc='data[1] is [[data_seed,sgd_seed,init_seed,np.random.randint(1,1000,1000)],[X_test,y_test],[X_train,y_train]]'
	# data_save1=[[data_seed,np.random.randint(1,1000,1000)],[X_test,y_test],[X_train,y_train]]
	# data_save=[desc,data_save1]
	return train_loader,test_loader

class ModularArithmeticDataset(Dataset):
	def __init__(self, P=97, seed=0,loss_criterion="MSE"):
		self.P = P
		self.seed = seed
		self.loss_criterion=loss_criterion

		# Instantiate the data
		self.data, self.indices = self.gen_data()

	def gen_data(self):
		data = torch.empty((self.P**2, 2*self.P),dtype=torch.float32)
		if self.loss_criterion=='MSE':
			indices = torch.empty((self.P**2,self.P),dtype=torch.float32)
		else:
			indices=torch.empty((self.P**2,self.P), dtype=torch.long)
		for i in torch.arange(self.P):
			for j in torch.arange(self.P):
				combined_idx = self.P*i + j
				i_onehot = torch.nn.functional.one_hot(i, self.P)
				j_onehot = torch.nn.functional.one_hot(j, self.P)
				data[combined_idx] = torch.cat((i_onehot, j_onehot), axis=0)
				# indices[combined_idx] = (i+j) % self.P
				#if you wanted to implement MSE as MSE between the one-hots but better between the indices methinks.
				newval=(i+j)%(self.P)
				if self.loss_criterion=='MSE':
					indices[combined_idx]=torch.nn.functional.one_hot(newval,self.P)
					#I think to calculate it the other way around you, i.e. for MSE, you need 
					#indices[combined_idx]=newval
				else:
					indices[combined_idx]=torch.nn.functional.one_hot(newval,self.P)
		return data, indices

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return (self.data[idx], self.indices[idx])


if __name__ == '__main__':

	import yaml
	import argparse

	# Load YAML file
	with open('mod_config.yaml', 'r') as file:
		config = yaml.safe_load(file)

	# This allows you to modify any of the yaml configs from the command line.
	parser = argparse.ArgumentParser(description="Update YAML configuration values.")
	for key in config:
		parser.add_argument(f'--{key}', type=str, help=f'Override value for {key}')

	args = parser.parse_args()
	

	# Update config with command-line arguments if provided
	for key, value in vars(args).items():
		if value is not None:
			config[key] = value

	#Now I want to initialize the variables - the slick way is below, but maybe more readable this way
	# Create variables with names from YAML keys
	# for key, value in config.items():
	# 	globals()[key] = value
	
	
	cluster_array=1
	data_seed_start=int(config['data_seed_start'])
	data_seed_end=int(config['data_seed_end'])
	sgd_seed_start=int(config['sgd_seed_start'])
	sgd_seed_end=int(config['sgd_seed_end'])
	init_seed_start=int(config['init_seed_start'])
	init_seed_end=int(config['init_seed_end'])
	weight_decay=float(config['weight_decay'])
	train_size=int(config['train_size'])
	test_size=int(config['test_size'])
	if type(config['hiddenlayers_input'])==list:
		hiddenlayers_input=list(config['hiddenlayers_input']) #Note list([1,2,3]) is the same list
	elif type(config['hiddenlayers_input'])==str:
		hiddenlayers_input=[int(i) for i in config['hiddenlayers_input'].split(',')]
	learning_rate_input=float(config['learning_rate_input'])
	train_type=str(config['train_type'])
	P=int(config['P'])
	train_fraction=float(config['train_fraction'])
	weight_multiplier=float(config['weight_multiplier'])
	cluster_arg=bool(config['cluster_arg'])
	epochs=int(config['epochs'])

	
	
	
	data_seeds=[i for i in range(data_seed_start,data_seed_end)]
	print(f' data_seeds={data_seeds}')
	sgd_seeds=[i for i in range(sgd_seed_start,sgd_seed_end)]
	print(sgd_seeds)	
	init_seeds=[i for i in range(init_seed_start,init_seed_end)]
	print(init_seeds)
		
	print(f" seeds: {data_seeds}, sgd_seeds: {sgd_seeds},init_seeds {init_seeds}")
	print(f" wd: {weight_decay}")
	

	dtype = torch.float32 # very important
	# seed = seed # fixed random seed for reproducibility
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # use GPU if available
	print(f'device: {device}')
	
	example = False # dataset is MNIST example if True
	ising = True # import ising dataset

	# set functions for neural network
	loss_criterion="CrossEntropy" # 'MSE' or 'CrossEntropy'
	loss_fn = nn.CrossEntropyLoss   # 'MSELoss' or 'CrossEntropyLoss'
	optimizer_fn = torch.optim.Adam     # 'Adam' or 'AdamW' or 'SGD'
	activation = nn.ReLU    # 'ReLU' or 'Tanh' or 'Sigmoid' or 'GELU'

	# set parameters for neural network
	weight_decay = weight_decay
	learning_rate = learning_rate_input # lr
	weight_multiplier = weight_multiplier # initialization-scale
	dropout_prob=0
	input_dim = 16**2 # 28 x 28 array of pixel values in MNIST (16x16 in Ising data)
	output_dim = 2 # 10 different digits it could recognize (2 diff phases for Ising)
	hiddenlayers=hiddenlayers_input
	conv_channels=[2,4]
	bs=200


	#Train params
	save_interval=200
	# set torch data type and random seeds
	torch.set_default_dtype(dtype)


	
	desc='modadd'
	if cluster_arg==True:
		root=f'../../large_files/modaddwd_lr_{learning_rate}_wm_{weight_multiplier}/hiddenlayer_{hiddenlayers}_desc_{desc}_wm_{weight_multiplier}_lr_{learning_rate}'#happens to be the same file structure in this case
	else:
		root=f'../../large_files/test_runs/linear/hiddenlayer_{hiddenlayers}_desc_{desc}_wm_{weight_multiplier}'
	
	os.makedirs(root,exist_ok=True)
	print('makedirs called')
	print(str(root))
	all_same=True
	if all_same:
		seeds=[(i,i,i) for i in data_seeds]
	else:
		seeds=list(itertools.product(data_seeds,sgd_seeds,init_seeds))
	for seed_triple in seeds:
		data_seed=seed_triple[0]
		sgd_seed=seed_triple[1]
		init_seed=seed_triple[2]
		#Define a data_run_object. Save this instead of the dictionary.
		args = TrainArgs(
			epochs=epochs,
			lr=learning_rate,
			weight_decay=weight_decay, # 0.001,
			weight_multiplier=weight_multiplier,
			dropout_prob=dropout_prob,
			data_seed=data_seed,
			sgd_seed=sgd_seed,
			init_seed=init_seed,
			device=device,
			grok=False,
			hiddenlayers=hiddenlayers,
			conv_channels=conv_channels,
			test_size=test_size,
			train_size=train_size,
			train_fraction=train_fraction,
			P=P,
			batch_size=bs,
			loss_criterion=loss_criterion
			)
		
		params_dic={'weight_decay':weight_decay,'weight_multiplier':weight_multiplier,'learning_rate':learning_rate,'hidden_layers':hiddenlayers,'conv_channels':conv_channels,'train_size':train_size,'test_size':test_size,'dropout_p':dropout_prob}
		save_object=seed_average_onerun(data_seed=args.data_seed,sgd_seed=args.sgd_seed,init_seed=args.init_seed,params_dic=params_dic)
		save_object.trainargs=args

		#config_dict['modelinstance']=create_model(init_seed=init_seed,model_type="CNN")
		
		
		save_object.start_time=int(time.time())
		if train_type=="Ising":
			train_loader,test_loader=create_ising_dataset(data_seed=data_seed,train_size=train_size,test_size=test_size)
			
		elif train_type=="Mod":
			ma_dataset = ModularArithmeticDataset(args.P, args.data_seed,loss_criterion=loss_criterion)
			
			train_dataset, test_dataset = torch.utils.data.random_split(ma_dataset, [args.train_fraction, 1-args.train_fraction])

			train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
			test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
			
		
		



		save_object.train_loader=train_loader
		save_object.test_loader=test_loader
		if train_type=="Ising":
			model=create_model(init_seed=init_seed,model_type="CNN")
		elif train_type=="Mod":
			model=create_model(init_seed=init_seed,model_type="ModMLP")
		

		print(f'model {train_type}')
		# Check if the instance has the _init_args attribute
		if hasattr(model, '_init_args'):
			print('attr')
			#had to call 'args' 'margs' because of the other args!
			# Extract the stored initialization arguments
			margs, kwargs = model._init_args

			# Reconstruct the configuration dictionary
			config_dict = {**dict(zip(inspect.signature(model.__class__.__init__).parameters, margs)), **kwargs}
			if 'self' in config_dict:
				config_dict.pop('self')
		else:
			print("The instance was created before the decorator was added and does not have initialization arguments stored.")
		
		save_object.modelinstance=copy.deepcopy(model)
		save_object.modelclass=model.__class__
		save_object.modelconfig=config_dict
		print(config_dict)
		
		

		train(epochs=args.epochs,initial_model=model,save_interval=save_interval,train_loader=train_loader,test_loader=test_loader,sgd_seed=args.sgd_seed,batch_size=args.train_size,one_run_object=save_object,train_type=train_type,loss_criterion=args.loss_criterion,plot_as_train=(not cluster),config_dict=config_dict)
		

		
		
		save_name=f'grok_dataseed_{data_seed}_sgdseed_{sgd_seed}_initseed_{init_seed}_wd_{weight_decay}_wm_{weight_multiplier}_time_{int(time.time())}'
		run_folder=str(root)
		# with open(str(root)+"/"+save_name, "wb") as dill_file:
		#     dill.dump(save_object, dill_file)
		try:
			with open(str(root)+"/"+save_name, "wb") as dill_file:
				torch.save(save_object, dill_file)
		except Exception as e:
			print(f"An error occurred during serialization: {e}")
		print(str(root)+"/"+save_name)

		if cluster_arg==False:
			# plot_traincurves(list(range(len(test_accuracy))),test_accuracy,train_accuracy,test_losses,train_losses,config_dict).show()
			#one_run_object.traincurves_and_iprs(one_run_object).show()
			save_object.cosine_sim(save_object).show()
			print(f'liner decomp plot')
			save_object.linear_decomposition_plot(save_object).show()
			
	
	# while True:
	# 	time.sleep(10)
	# 	print("Sleeping")






