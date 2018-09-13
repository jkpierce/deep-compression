import numpy as np
import skimage as ski
from skimage import io, transform
import matplotlib.pyplot as plt
import os
import copy

import torch
from torchvision import transforms
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torch.autograd import Variable
import PIL
import torch.optim as optim



class Dataset:
    def __init__(self, path):
        self.path=path

    def __getitem__(self, index):
        img = PIL.Image.open(self.path+'/im-%04d.jpg'%index)
        vect = np.load(self.path+'/compressed-points-%04d.npy'%index)
        transform = transforms.Compose([transforms.ToTensor()])
        img = transform(img)
        vect = torch.FloatTensor(np.concatenate(vect))
        return img, vect 

    def __len__(self):
        return len([f for f in os.listdir(self.path) if f.endswith('.jpg')])

    
train_data = Dataset('./training')


class AlexNet(nn.Module):

    def __init__(self,D_out=10000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, D_out),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x
    
model = AlexNet(D_out=5*100)    
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


k=30 #size of batch
N = 100 #number epochs

train_loader = DataLoader(train_data, batch_size=k, shuffle=True) #data loader for training
losses = [] #track the losses 

for epoch in range(N): 
    for i,(inputs,targets) in enumerate(train_loader): 
        
        #prepare batch
        inputs,targets = Variable(inputs), Variable(targets,requires_grad=False)
        
        #zero gradients 
        optimizer.zero_grad()
        
        #calculate model prediction
        outputs = model(inputs)
        
        #calculate loss
        loss = criterion(outputs,targets)
        
        #backpropagate loss 
        loss.backward()
        optimizer.step()

        # print statistics
        print(loss.data[0])
        losses.append(loss.data[0])
        
    #at each epoch save the model and the losses 
    np.save('mylosses',losses)
    model.save_state_dict('mytraining.pt')
