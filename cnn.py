import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module) :
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1,32,3,1, padding='same')
        self.conv2 = nn.Conv2d(32,64,3,1,padding='same')
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(3136,1000) #3136 = 7 * 7 * 64
        self.fc2 = nn.Linear(1000,10)
    
    def forward(self, x) :
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x,2)
        x = self.dropout(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        
        return output
    
class FCN(nn.Module) :
    def __init__(self) :
        super(FCN,self).__init__()
        self.fc1 = nn.Linear(784,196)
        self.fc2 = nn.Linear(196,98)
        self.fc3 = nn.Linear(98,10)
    
    def forward(self, x) :
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x))
        
        return output        
    
class FCN2(nn.Module) :
    def __init__(self) :
        super(FCN2,self).__init__()
        self.fc1 = nn.Linear(256,128)
        self.fc2 = nn.Linear(128,10)
    
    def forward(self, x) :
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        output = F.log_softmax(self.fc2(x))
        
        return output    
    
class RangeConstraint :
    def __init__(self, min_val, max_val) :
        self.min_val = min_val
        self.max_val = max_val
    def __call__(self, param) :
        param.data = torch.clamp(param.data,self.min_val, self.max_val)
        return param
    
class varFCN(nn.Module) :
    # ex. features : [(256,128),(128,10)]
    def __init__(self, min, max, features : list ,activation=True, bias=True) :
        super(varFCN,self).__init__()
        self.bias = bias
        
        # n = size**2
        # self.fc1 = nn.Linear(n,int(n/2), bias=self.bias)
        # self.fc2 = nn.Linear(int(n/2),10, bias=self.bias)
        # self.fc1.weight.data.uniform_(min,max)
        # self.fc2.weight.data.uniform_(min,max)
        
        fc_list = list()
        for feature in features :
            fc = nn.Linear(feature[0], feature[1], bias=self.bias)
            fc.weight.data.uniform_(min,max)
            fc_list.append(fc)
            if activation :
                fc_list.append(nn.ReLU())
        self.sequential = nn.Sequential(*fc_list)
        
    
    def forward(self, x) :
        x = torch.flatten(x,1)
        # if self.activation :
        #     x = F.relu(self.fc1(x))
        # else :
        #     x = self.fc1(x)
        x = self.sequential(x)
        output = F.log_softmax(x)
        
        return output    
    

class varFCN2(nn.Module) :
    def __init__(self,size=28,activation=True, bias=True) :
        super(varFCN2,self).__init__()
        n = size**2
        self.bias = bias
        self.fc1 = nn.Linear(n,10, bias=self.bias)
        self.activation = activation
    
    def forward(self, x) :
        x = torch.flatten(x,1)
        output = F.softmax(self.fc1(x))
        
        return output    