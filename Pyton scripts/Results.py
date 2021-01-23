import torch
from PIL import Image, ImageOps
import pandas as pd
from os import walk
import os
import argparse

from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler
from torchvision.transforms import transforms
from sklearn.model_selection import KFold
import torchvision
import torch.nn as nn

import torchvision.models as models

from torch.autograd import Variable

import numpy as np

from torch.utils.tensorboard import SummaryWriter

path_train = "/home/usuaris/imatge/marc.bo/proj/headpose/mixed_depth/splits/train_2_depth.txt"
path_test =  "/home/usuaris/imatge/marc.bo/proj/headpose/mixed_depth/splits/test_2_depth.txt"
path_checkpoints = "/home/usuaris/imatge/marc.bo/proj/headpose/checkpoints"


'''
This code contains the training and validation process of the analysis of the results. 
It also inculdes the visualization of all these results. 

This code uses:
- Two Resnet 50 Model to predict the Yaw and Pitch angles.
- Custom loss function
- Epochs: 160
- Learning Rate: 0.001
- Optimizer: SGD


Tensorboard displays the yaw and pitch loss for all the epochs,
as well as the positions x,y in all the frames.
It also shows a comparison between the prediced yaw and pitch angle
and the ground truth for each trajectory.
'''
ap = argparse.ArgumentParser()

ap.add_argument("-name", "--name", required=True, help ="filename")
ap.add_argument("-lr", "--learningRate", required=True)
ap.add_argument("-ep", "--epochs", required=True)
ap.add_argument("-whichcheckpoint", "--checkpoint", required= False, help ="pathToCheckpoint")
ap.add_argument("-usecheckpoint", "--use", required= False)


vars = vars(ap.parse_args())

run = vars['name'] + '_' + vars['learningRate'] + '_' + vars['epochs']
writer = SummaryWriter('runs/Resultats/' + run)
path_checkpoints += vars['name'] + '_' + vars['learningRate'] + '_' + vars['epochs']
lrate = float(vars['learningRate'])
epochs = int(vars['epochs'])
checkpointpath = vars['checkpoint']
useCheckpoint = bool(vars['use'])

def angular_error_pitch(a1,a2):
    #Custom loss function
    phi = torch.abs(a1 - a2) % 360
    phi = torch.mean(phi)
    dist = 360 - phi if phi > 180 else phi
    return dist

def angular_error_yaw(a1,a2):
    #Custom loss function
    phi = torch.abs(a1 - a2) % 360
    phi = torch.mean(phi)
    dist = 360 - phi if phi > 180 else phi
    return dist



class AngleData(Dataset):
    def __init__(self, split, transform=None):
        self.paths = pd.read_csv(split, sep=' ', header=None)[0].values.tolist()
        self.transform = transform
        self.samples = []
        self.posx = pd.read_csv(split, sep=' ', header=None)[4].values.tolist()
        self.posy = pd.read_csv(split, sep=' ', header=None)[5].values.tolist()
        self.labels = [pd.read_csv(split, sep=' ', header=None)[1].values.tolist(), pd.read_csv(split, sep=' ',header=None)[2].values.tolist()]
        self.mean = 0
        self.var = 0
        for i in range(len(self.paths)):
            self.samples.append((self.paths[i], [float(self.labels[0][i]), float(self.labels[1][i])],
                [self.posx[i][1:self.posx[i].find(',')], self.posy[i][:-1]]))
            image = Image.open(self.paths[i])
            pixels = np.asarray(image)
            pixels = pixels.astype('float32')
            self.mean += pixels.mean()
            self.var += pixels.var()

        self.mean /= len(self.paths)
        self.var /= len(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
            print(idx)

        path, label, pos = self.samples[idx]
        img = Image.open(path)

        img_arr = np.asarray(img)

        img_arr = (img_arr - self.mean) / self.var 

        if self.transform:
            img = self.transform(img)
        return img, label, pos

train_dataset = AngleData(split=path_train, transform=transforms.Compose([transforms.Resize((640, 480), 2),
                                                                          transforms.ToTensor()]))
test_dataset = AngleData(split=path_test, transform=transforms.Compose([transforms.Resize((640, 480), 2),
                                                                        transforms.ToTensor()]))

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=4, shuffle=False, num_workers=4)



def set_parameters_require_grad(model, feature_extracting):
    if feature_extracting:
        for name, param in model.named_parameters():
            param.requires_grad = False


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.model_resnet = models.resnet50(pretrained=True)
        feature_extracting = False
        set_parameters_require_grad(self.model_resnet, feature_extracting)
        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1)
        )

        #print("params to learn")
        self.params_to_update = []
        for name, param in self.model_resnet.named_parameters():
            if feature_extracting:
                if param.requires_grad == True:
                    self.params_to_update.append(param)
                    #print("\t",name)
            else:
                if param.requires_grad == True:
                    self.params_to_update.append(param)
                    #print("\t", name)


    def forward(self, x):
        out1 = self.model_resnet(x)
        return out1


model_yaw = MyModel()
model_pitch = MyModel()

model_yaw = model_yaw.cuda()
model_pitch = model_pitch.cuda()
def train(epoch, model_yaw, model_pitch, optimizer_yaw, optimizer_pitch, criterion, loader):
    model_yaw.train()
    model_pitch.train()
    loss_pitch_tot = 0
    loss_yaw_tot = 0
    i = 0
    for data in loader:
        i += 1
        inputs, labels, pos = data
        inputs = inputs.float()
        inputs = inputs.cuda()
        labels = [labels[0].cuda(), labels[1].cuda()]

        optimizer_yaw.zero_grad()
        optimizer_pitch.zero_grad()

        outputsYaw= model_yaw(inputs.float())
        outputsPitch = model_pitch(inputs.float())

        outputsYaw = outputsYaw.double()
        outputsPitch = outputsPitch.double()
        outputsPitch = outputsPitch.squeeze(-1)
        outputsYaw = outputsYaw.squeeze(-1)

        loss_h = angular_error_pitch(labels[0], outputsPitch)
        loss_v = angular_error_yaw(labels[1], outputsYaw)
        loss_pitch_tot += loss_h.item()
        loss_yaw_tot += loss_v.item()
        loss_h.backward()
        loss_v.backward()

        optimizer_yaw.step()
        optimizer_pitch.step()
    writer.add_scalar('Loss/train_Pitch', loss_pitch_tot / i, epoch)
    writer.add_scalar('Loss/train_Yaw', loss_yaw_tot/i, epoch)
    print('Train Epoch: {} Loss: Horizontal {:.6f} Vertical: {:.6f}'.format(epoch + 1, loss_pitch_tot / (i),loss_yaw_tot / (i)))
    return loss_pitch_tot / i, loss_yaw_tot / i

def test(model_yaw, model_pitch, criterion, loader):
    model_yaw.eval()
    model_pitch.eval()
    i = 0
    loss_pitch_tot = 0
    loss_yaw_tot = 0
    for data in loader:
        i += 1
        inputs, labels, pos = data
        inputs = inputs.float()
        inputs = inputs.cuda()
        labels = [labels[0].cuda(), labels[1].cuda()]

        outputsYaw = model_yaw(inputs.float())
        outputsPitch = model_pitch(inputs.float())
        
        outputsYaw = outputsYaw.double()
        outputsPitch = outputsPitch.double()

        outputsYaw = outputsYaw.squeeze(-1)
        outputsPitch = outputsPitch.squeeze(-1)

        loss_pitch = angular_error_pitch(labels[0], outputsPitch)
        loss_yaw = angular_error_yaw(labels[1], outputsYaw)
        loss_pitch_tot += loss_pitch.item()
        loss_yaw_tot += loss_yaw.item()

    writer.add_scalar('Loss/test_Pitch', loss_pitch_tot/i, epoch)
    writer.add_scalar('Loss/test_Yaw', loss_yaw_tot/i, epoch)
    print('\nTest set: Average loss: Pitch {:.4f} Yaw {:.4f}\n'.format(loss_pitch_tot/(i), loss_yaw_tot/(i)))
    return loss_pitch_tot/i, loss_yaw_tot/i

#lr = 0.001/10
criterion = nn.MSELoss()

optimizer_yaw = torch.optim.SGD(model_yaw.params_to_update, lr=lrate, momentum=0.9)
optimizer_pitch = torch.optim.SGD(model_pitch.params_to_update, lr=lrate, momentum=0.9)

train_losses = []
test_losses = []
print('start training')

for epoch in range(epochs):
    train(epoch, model_yaw, model_pitch, optimizer_yaw, optimizer_pitch, criterion, train_loader)
    test(model_yaw, model_pitch, criterion, test_loader)

print('Finished Training')

root = '/home/usuaris/imatge/marc.bo/proj/headpose/paths_real_depth/'

files = [w for w in os.listdir(root) if w != 'split.py' or w != 'splits']
for file1 in files:
    if file1 != 'split.py' and file1 != 'splits':
        val_dataset = AngleData(split=os.path.join(root, file1), transform = transforms.Compose([transforms.Resize((640, 480),2), transforms.ToTensor()]))
        val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=False, num_workers=4)

        labels_yaw = []
        labels_pitch = []
        pred_yaw = []
        pred_pitch = []
        #position_x = []
        #position_y = []

        for data in val_loader:

            images, labels, pos = data
            inputs = images.cuda()
            inputs = inputs.float()
            outputYaw = model_yaw(inputs)
            outputPitch = model_pitch(inputs)

            for i in range(len(labels[0])):
            
                labels_yaw.append(labels[1][i].item())
                labels_pitch.append(labels[0][i].item())
                pred_yaw.append(outputYaw[i].item())
                #position_x.append(float(pos[0][i]))
                #position_y.append(float(pos[1][i]))
                pred_pitch.append(outputPitch[i].item())
                print('Pitch {} / {} \n Yaw {} / {} \n'.format(labels[0][i].item(), outputPitch[i].item(), labels[1][i].item(), outputYaw[i].item()))
        
        for j in range(len(labels_yaw)):
            writer.add_scalars('{} Yaw: '.format(file1), {'Pred':pred_yaw[j],
                                        'truth':labels_yaw[j]}, j)
            writer.add_scalars('{} Pitch: '.format(file1), {'Pred':pred_pitch[j],
                                        'truth':labels_pitch[j]}, j)
            #writer.add_scalars('{} Position: '.format(file1), {'x':position_x[j],
                                         #   'y':position_y[j]}, j)