import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from matplotlib import pyplot as plt
from cnn import CNN, FCN, FCN2, varFCN, varFCN2
from torch.utils.data import RandomSampler

import time
import os
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

def train(save_path, conductance_path, resize, seed, batch_size, learning_rate, epoch_num, features, min_weight, activation, bias, top_n) :
    
    # # path information
    # save_path = r"Memristor"
    # conductance_path = r"C:\Users\rholab\OneDrive - SNU\바탕 화면\Python\paper\Memristor\memristor_data\conductance.csv"

    # # learning hyper parameter
    # resize  = 16
    # seed = 42
    # batch_size = 64
    # learning_rate = 0.0001
    # epoch_num = 2
    # features = [(256,128),(128,10)]
    # min_weight = 5

    # activation = False
    # bias = False

    # # save pt parameter
    # top_n = 3
    
    
    data = pd.read_csv(conductance_path)
    conduct = data['conductance']


    # Get the max and min values from conductance data
    min_conduct = min(conduct)
    max_conduct = max(conduct)

    # Set offset ~5%
    min_conduct = 0.95 * min_conduct
    max_conduct = 0.95 * max_conduct

    # [min_conduct, max_conduct] -> [min_weight, max_weight]
    ratio = max_conduct/min_conduct
    max_weight = min_weight * ratio * 0.9

    print(f"min : {min_weight}, max : {max_weight}")

    is_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if is_cuda else 'cpu')
    composed = transforms.Compose([transforms.Resize((resize,resize)),transforms.ToTensor()])
    train_data = datasets.MNIST(root ='./data/02/',
                                train=True,
                                download=True,
                                transform=composed)
    test_data = datasets.MNIST(root='./data/02/',
                            train=False,
                            download=True,
                            transform=composed)

    train_sampler = RandomSampler(train_data, generator=torch.Generator().manual_seed(seed))
    test_sampler = RandomSampler(test_data, generator=torch.Generator().manual_seed(seed))

    train_loader = torch.utils.data.DataLoader(dataset = train_data, 
                                            batch_size = batch_size, 
                                            sampler = train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset = test_data, 
                                            batch_size = batch_size, 
                                            sampler = test_sampler)

    # model = varFCN(min=min_weight, max=max_weight, size=resize, activation=activation, bias=bias).to(device=device)
    model = varFCN(min=min_weight, max=max_weight, features=features, activation=activation, bias=bias).to(device=device)

    optimizer = optim.Adam(model.parameters(),lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    tm = time.localtime()
    name_tag = f"{tm.tm_mon:02}_{tm.tm_mday:02}_{tm.tm_hour:02}_{tm.tm_min:02}"
    checkpoint_path = os.path.join(save_path,name_tag)
    os.mkdir(checkpoint_path)

    model.train()
    i = 1
    best = np.ones(top_n) * np.inf
    best_idx = np.ones(top_n) * np.inf

    for epoch in range(epoch_num) :
        print(f"Current epoch is {epoch + 1}")
        for data, target in train_loader :
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            for p in model.parameters():
                p.data.clamp_(min=min_weight, max=max_weight)
                
            if i % 1000 == 0 :
                print("Train Step : {}\tLoss : {:3f}".format(i,loss.item()))
                if loss.item() < np.max(best) :
                    with open(os.path.join(checkpoint_path,"model.txt"), 'a') as f :
                        f.write(f"epoch : {epoch+1} | loss : {loss.item()}\n")
                    if epoch >= top_n + 1 :
                        os.remove(os.path.join(checkpoint_path,f"model_{int(best_idx[-1])}.pth"))
                    torch.save({
                                    'model_state_dict' : model.state_dict(),
                                    'optimizer_state_dict' : optimizer.state_dict(),
                                    'criterion' : criterion,
                                }, os.path.join(checkpoint_path, f"model_{epoch + 1}.pth"))
                    
                    best[-1] = loss.item()
                    best_idx[-1] = epoch + 1
                    sorted_index = best.argsort()
                    best = np.take_along_axis(best, sorted_index, axis=-1)
                    best_idx = np.take_along_axis(best_idx, sorted_index, axis=-1)
            i += 1

    torch.save({
        'model_state_dict' : model.state_dict(),
        # 'optimizer_state_dict' : optimizer.state_dict(),
        'criterion' : criterion,
    }, os.path.join(checkpoint_path, "model_last.pth"))
    
    accuracy_list = list()
    path_list = [os.path.join(checkpoint_path,f"model_{int(idx)}.pth") for idx in best_idx]
    path_list.append(os.path.join(checkpoint_path,"model_last.pth"))
    for path in path_list:
        checkpoint = torch.load(path)
        model_state_dict = checkpoint['model_state_dict']   
        model = varFCN(min=min_weight, max=max_weight, features=features, activation=activation, bias=bias).to(device=device)
        model.load_state_dict(model_state_dict)
        
        model.eval()    # 평가시에는 dropout이 OFF 된다.
        correct = 0
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()
            
        accuracy = 100. * correct / len(test_loader.dataset)
        print('Test set Accuracy : {:.2f}%'.format(accuracy))
        accuracy_list.append(accuracy)
    best_idx = best_idx.tolist()
    best_idx.append("last")

    with open(os.path.join(checkpoint_path,"info.txt"), 'w') as f :
        f.write(str(model))
        f.write('\n')
        f.write(f"scaler range  ({min_conduct/min_weight},{max_conduct/max_weight})\n")
        for idx,accuracy in zip(best_idx,accuracy_list) :
            f.write(f"model {idx}'s test score : {accuracy}%\n")
        
    with open(os.path.join(checkpoint_path,"arch.txt"), 'w') as f :
        my_dict = {'resize' : resize, 'seed' : seed, 'batch_size' : batch_size, 'learning_rate' : learning_rate, 
                   'epoch_num' : epoch_num, 'activation' : activation, 'bias' : bias, 'features' : features,
                   'min_weight' : min_weight, 'max_weight' : max_weight
                   }
        json.dump(my_dict,f)
    
    with open(os.path.join(checkpoint_path,"arch.pickle"), mode='wb') as f:
        my_dict = {'resize' : resize, 'seed' : seed, 'batch_size' : batch_size, 'learning_rate' : learning_rate, 
                   'epoch_num' : epoch_num, 'activation' : activation, 'bias' : bias, 'features' : features,
                   'min_weight' : min_weight, 'max_weight' : max_weight
                   }
        pickle.dump(my_dict, f)
        
# path information
save_path = r"sweep4"
conductance_path = r"memristor_data\conductance.csv"

# learning hyper parameter
resize  = 28
seed = 42
batch_size = 64
learning_rate = 0.0001
epoch_num = 50
# features = [(resize**2,128),(128,10)] # resize : 16 -> 256
# min_weight = 5
activation = False
bias = False

# save pt parameter
top_n = 3

###########################################

features_list = [
    [(resize**2,10)],
   [(resize**2,512),(512,10)],
   [(resize**2,256),(256,10)],
   [(resize**2,128),(128,10)],
   [(resize**2,64),(64,10)],
   [(resize**2,32),(32,10)],
    # [(resize**2,128),(128,64),(64,10)],
    # [(resize**2,128),(128,32),(32,10)]
]

min_weight_list = [15,20,25,30,50]

for features in tqdm(features_list) :
    for min_weight in min_weight_list :
        params_dict = {
            "save_path" : save_path, 
            "conductance_path" : conductance_path, 
            "resize" : resize, 
            "seed" : seed, 
            "batch_size" : batch_size, 
            "learning_rate" : learning_rate, 
            "epoch_num" : epoch_num, 
            "features" : features, 
            "min_weight" : min_weight, 
            "activation" : activation, 
            "bias" : bias, 
            "top_n" : top_n
        }
        train(**params_dict)

