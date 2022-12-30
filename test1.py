import numpy
from PIL import Image
from conf import cfg, load_cfg_fom_args
from robustbench.data import load_cifar10
import torch
import torch.nn as nn
import keras
import math
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel
from dent import Dent
from dent import copy_model_state
from copy import deepcopy
from keras.datasets import cifar10


bs=128
torch.cuda.empty_cache()
device = torch.device("cpu")



def cal_accuracy(model_copy,x_test,y_test):
    n_batches = int(numpy.ceil(x_test.shape[0] / bs))
    robust_flags = torch.zeros(x_test.shape[0], dtype=torch.bool, device=x_test.device)
    for batch_idx in range(n_batches):
        start_idx = batch_idx * bs
        end_idx = min( (batch_idx + 1) * bs, x_test.shape[0])
        x = x_test[start_idx:end_idx, :].clone().to(device)
        y = y_test[start_idx:end_idx].clone().to(device)
        print(x.shape)
        output1 = model_copy(x).max(dim=1)[1]
        cb=y.eq(output1)
        robust_flags[start_idx:end_idx] = cb.detach().to(robust_flags.device)
    return torch.sum(robust_flags).item() / x_test.shape[0]
    
def random_image():
    imarray = numpy.random.rand(1,3,32,32) * 255
    imarray=numpy.tile(imarray,[128,1,1,1])
    #print(imarray.shape)
    #print(imarray[:5,:3,:5,:5])
    imarray=imarray/255
    imarray=numpy.float32(imarray)
    res=torch.from_numpy(imarray)
    return res

def acc_with_random(model,x_test,y_test,res):
    itr=100
    acc_list = []
    for i in range(0, itr, 10):
        fin=model(res)
        #torch.save(model.state_dict(), 'model_state.pth')
        #model_copy = dent_model
        #model_copy.load_state_dict(torch.load('model_state.pth'))
        model_copy = deepcopy(model.state_dict())
        #print(model_copy)
        acc = cal_accuracy(model_copy, x_test, y_test)
        acc_list.append(acc)
    return acc_list
        
    
    





res=random_image()
x_test, y_test = load_cifar10(128)

#(X_train, Y_train) , (X_test, Y_test) = cifar10.load_data()
#X_test=X_test[:128]
#Y_test=Y_test[:128]
#X_test=X_test.reshape(128,3,32,32)
#X_test=numpy.float32(X_test)
#X_test=torch.from_numpy(X_test)
#Y_test=Y_test.reshape(128)
#Y_test=numpy.float32(Y_test)
#Y_test=torch.from_numpy(Y_test)

#rint(X_test.dtype)
#print(X_test.dtype)
#print(Y_test.shape)

#print(res[0][0][0].dtype)
base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,'cifar10', ThreatModel.Linf)
if cfg.MODEL.ADAPTATION == "dent":
    #assert cfg.MODEL.EPISODIC
    dent_model = Dent(base_model, cfg.OPTIM)


p=acc_with_random(dent_model,x_test, y_test, res)
print(p)
