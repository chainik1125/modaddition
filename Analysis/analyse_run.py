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
from Analysis.activations_Ising import open_files_in_leaf_directories
from cluster.cluster_run_average import TrainArgs
from cluster.cluster_run_average import CNN
from cluster.cluster_run_average import CNN_nobias
from contextlib import contextmanager
import functools
from cluster.cluster_run_average import ModularArithmeticDataset
from cluster.cluster_run_average import Square, MLP
import glob


foldername="/Users/dmitrymanning-coe/Documents/Research/Grokking/IsingCluster/Ising_areas/Ising_varynorm_wd_0.0/hiddenlayer_[100]_desc_avgIsingstandard_fixedwm_0.1_wm_0.1"
objects=open_files_in_leaf_directories(foldername)
test_seed=1
for object in objects:
    if object.trainargs.data_seed==test_seed:
        print(object.train_losses[:5])

