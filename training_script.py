import numpy as np
import os 
import torch
from torchvision import transforms
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.utils.data import DataLoader
from torch.autograd import Variable
import PIL
import torch.optim as optim

##################################################################################
# Dataloader 

class Dataset:
    def __init__(self, path):
        self.path=path

    def __getitem__(self, index):
        img = PIL.Image.open(self.path+'/train-%04d.jpg'%index)
        #img = PIL.Image.fromarray(np.stack((img,)*3,-1))
        vect = np.load(self.path+'/train-comp-%04d.npy'%index)
        transform = transforms.Compose([transforms.Resize((227,227)), #224? 
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                       ])
        img = transform(img)
        img.requires_grad=True
        vect = torch.FloatTensor(np.concatenate(vect)) 
        return img, vect 

    def __len__(self):
        return len([f for f in os.listdir(self.path) if f.endswith('.jpg')])

train_data = Dataset('./data/training/')

##################################################################################
# Define neural net 


class AlexNet(nn.Module):

    def __init__(self,D_out):
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device is ' , device)
model = torch.load('./partial-trains/200epochs.pt')
model.train()
model.to(device) 

####################################################################################

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

#################################################################################
losses = []
k = 30 #size of batch
N = 1000 #number epochs
b = int(len(train_data)/k) #number of batches

train_loader = DataLoader(train_data, batch_size = k, shuffle = True) #batch data loader

for epoch in range(N): # epoch iterator 
    
    running_loss = 0 # mean loss per epoch 
    
    for i, (inputs, targets) in enumerate(train_loader): # batch iterator 
        
        ##########################################
        # TRAINING
        ##########################################
        inputs, targets = inputs.to(device), targets.to(device) # batch to gpu
        optimizer.zero_grad() # zero gradients
        outputs = model(inputs) # model prediction
        loss = criterion(outputs,targets)  # loss computation
        loss.backward() #backpropagation
        optimizer.step() #gradient descent 
        ##########################################

        running_loss+=loss.cpu().data.item()

    # print/store loss
    # clear_output(wait=True)
    print('epoch loss: ',round(running_loss/i,2))
    if epoch%10==0 and epoch!=0:
        n = epoch 
        torch.save(model,'./partial-trains/%05d-epochs.pt'%n)
        np.save('./partial-trains/losses.npy',np.array(losses))
    losses.append(running_loss/i)

torch.save(model,'./partial-trains/finalstate.pt'%n)
np.save('./partial-trains/losses.npy',np.array(losses))
