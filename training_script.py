import numpy as np
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device is ' , device)
model = torch.load('200epochs.pt')
model.train()
model.to(device) 

####################################################################################

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

#################################################################################

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
        n = epoch + 200
        torch.save(model,'%03d-epochs.pt'%n)
    losses.append(running_loss/i)

torch.save(model,'currentstate.pt'%n)
