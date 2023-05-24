import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datetime import datetime

torch.set_num_threads(2)
with open("par.yaml", "r") as f:
    par = yaml.load(f, Loader=yaml.FullLoader)

source_path = "src/MtMs"
for file in os.listdir(source_path):
    os.path.join(source_path, file)

source_path = "src/helpers"
for file in os.listdir(source_path):
    os.path.join(source_path, file)

temp_file_path = "temp"
torch.manual_seed(par["seed"])
random.seed(par["seed"])
start_time = datetime.now()

x_range = [-5, 5]
A_range = [0.1, 5]
b_range = [0, np.pi]

def f(x, theta):
    return theta["A"] * torch.sin(x + theta["b"])

def DGP(K, theta, task):
    x = torch.FloatTensor(np.random.uniform(x_range[0], x_range[1], size=(K, 1)))
    y = f(x, theta) + torch.normal(0, 0.1, size=(K, 1))
    task = [task] * K
    return {"y": y, "x": x, "task": task}

thetas = [{"A": random.uniform(A_range[0], A_range[1]), "b": random.uniform(b_range[0], b_range[1])} for _ in range(par["M_in"] + par["M_out"])]
thetas_in = thetas[:par["M_in"]]
thetas_out = thetas[par["M_in"]:]
train_in = pd.concat([pd.DataFrame(DGP(par["K_train"], thetas_in[i], i)) for i in range(par["M_in"])])
test_in = pd.concat([pd.DataFrame(DGP(par["K_test"], thetas_in[i], i)) for i in range(par["M_in"])])