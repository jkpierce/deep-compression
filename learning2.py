import numpy as np
import os
import copy
import torch
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader
import PIL
import torch.optim as optim

class Dataset:
    """load the image / encoded object position representation pairs for training and testing data"""
    def __init__(self, path, mode = 'train'):
        self.path=path
        self.mode=mode
    def __getitem__(self, index):
        mode = self.mode
        if mode=='train':
            fname = '/train-%04d.jpg'
        elif mode=='test':
            fname = '/test-%04d.jpg'
            
        if mode=='train':
            fname1 = 'train-comp-%04d.npy'
        elif mode=='test':
            fname1 = 'test-comp-%04d.npy'
        img = PIL.Image.open(self.path+fname%index)
        vect = np.load(self.path+fname1%index)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, ), (0.5, ))
                                       ])
        img = transform(img)
        if mode=='train':
            img.requires_grad=True
        vect = torch.FloatTensor(np.concatenate(vect)) 
        return img, vect 

# Initialize dataset iterators and find gpu if available 
train_data = Dataset('./data/training/',mode='train')
test_data = Dataset('./data/testing/',mode='test')
print('data is loaded')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device is '+ str(device))


class Net(nn.Module):
    """ this is lenet 5 adapted to the problem """
    def __init__(self):
        super(Net, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 6, 7, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(6, 16, 5, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), 2),
            nn.Conv2d(16, 120, 5, 1),
            nn.ReLU(inplace=True)
                                    )
        self.fc = nn.Sequential(
            nn.Linear(120*13**2, 5000),
            nn.ReLU(),
            nn.Linear(5000, 500),
                                )
    def forward(self, img):
        output = self.convnet(img)
        output = output.view(-1, 120*13**2)
        output = self.fc(output)
        return output

model = Net().to(device).train()

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True)
losses = []

# set up the training loop and dataset iterator 
k = 250 #size of batch 
N = 500 #number epochs
b = int(len(train_data)/k) #number of batches
train_loader = DataLoader(train_data , batch_size = k, shuffle = True) #batch data loader

# train the network 
for epoch in range(N): # epoch iterator 
    epoch_loss = 0 # mean loss per epoch 
    for i, (inputs, targets) in enumerate(train_loader): # batch iterator 
        inputs, targets = inputs.to(device), targets.to(device) # batch to gpu
        optimizer.zero_grad() # zero gradients
        outputs = model(inputs) # model prediction
        loss = criterion(outputs,targets)  # loss computation
        loss.backward() # backpropagation
        optimizer.step() # gradient descent 
        epoch_loss+=loss.cpu().data.item() # pull the batch losses 
    epoch_loss /= i
    print('epoch loss: ',round(epoch_loss,2)) # print/store loss
    if epoch%10==0 and epoch!=0:     
        n = epoch
        torch.save(model,'./partial-trains/%04d-epochs.pt'%n) # save partially trained model 
    losses.append(epoch_loss) # keep the losses 
scheduler.step(epoch_loss) # possibly modify the learning rate 


