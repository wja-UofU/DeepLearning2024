import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from torchvision import transforms 
from augment_utils import augment_dataset
import random
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt




class base_CNN(torch.nn.Module):
    # see: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html  for pytorch tutorial
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6,kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16,kernel_size=5)
        self.fc1 = nn.Linear(16*61*61,120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
    
    def forward(self,x):
        x = self.conv1(x)
        
        x=self.pool(x)
        x=F.relu(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_model(self,x,y,x_val,y_val,epochs,device,optimizer = optim.Adam,loss_function = nn.CrossEntropyLoss,learning_rate = 0.001):
        x = torch.tensor(x,dtype=torch.float32)
        if x.shape[1] != 3: 
            x = x.permute(0, 3, 1, 2)
        
        y= torch.asarray(y,dtype=torch.long)
        net = base_CNN()

        # move to the device currently in use
        x=x.to(device)
        y=y.to(device)
        net=self.to(device)
        
        running_loss = []
        running_val_loss=[]
        optimizer = optimizer(net.parameters(),lr=learning_rate)
        loss_function = loss_function()
        for epoch in range(epochs):
            self.train()
            print('Training Epoch {}/{}'.format(epoch,epochs))
            optimizer.zero_grad()
            
            forward = net(x)
            loss = loss_function(forward,y)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

            val_loss = self.validation(x_val, y_val, device, loss_function)
            running_val_loss.append(val_loss)
            
            print('Current Loss: ',loss.item())

        return net,running_loss,running_val_loss

    def predict(self,x,device):
        x = torch.tensor(x,dtype=torch.float32)
        if x.shape[1] != 3: 
            x = x.permute(0, 3, 1, 2)
        x=x.to(device)
        self.to(device)
        self.eval()
        predictions =self.forward(x)
        predictions = F.softmax(predictions, dim = 1)
        predictions_max = np.argmax(predictions.cpu().detach().numpy(),axis=1)
        predictions = predictions.cpu().detach().numpy()
        return predictions_max,predictions
    
    def validation(self,x_val,y_val,device,loss_function):
            x_val = torch.tensor(x_val,dtype=torch.float32).to(device)
            if x_val.shape[1] != 3: 
                x_val = x_val.permute(0, 3, 1, 2)
           
            y_val = torch.tensor(y_val, dtype=torch.long).to(device)

            # Forward pass for validation
            val_outputs = self.forward(x_val)
            val_loss = loss_function(val_outputs, y_val) 
            return val_loss.item()

def calculate_accuracy(x,y):
    correct = 0
    for i,item in enumerate(x):
        if item == y[i]:
            correct+=1
        
    pct_correct = correct/len(x)
    return pct_correct



class small_data_CNN(torch.nn.Module):
    # see: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html  for pytorch tutorial
    def __init__(self):
        seed = 33  
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=4,kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv2 = nn.Conv2d(in_channels=4, out_channels=8,kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=8,out_channels=8,kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3)
        self.conv6 = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3)
        self.fc1 = nn.Linear(16*2*2,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32, 3)
       
    
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.pool(F.relu(self.conv6(x)))
        x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_model(self,x,y,x_val,y_val,epochs,device,optimizer = optim.Adam,loss_function = nn.CrossEntropyLoss,learning_rate = 0.001,augmentation=None,add=False):
        x = torch.tensor(x,dtype=torch.float32)
        if x.shape[1] != 3: 
            x = x.permute(0, 3, 1, 2)
        


        y= torch.asarray(y,dtype=torch.long)
        net = small_data_CNN()

        # move to the device currently in use
        x=x.to(device)
        y=y.to(device)
        net=self.to(device)
        
        running_loss = []
        running_val_loss=[]
        optimizer = optimizer(net.parameters(),lr=learning_rate)
        loss_function = loss_function()
        if augmentation != None and add == True:
                augmented_x = augment_dataset(x, augmentation)
                x = torch.cat((x, augmented_x), dim=0)
                if augmentation == 'best':
                       y = torch.cat((y, y, y), dim=0)
                else:
                    y = torch.cat((y, y), dim=0)  # Duplicate labels for augmented data
        for epoch in range(epochs):
            if augmentation != None and add == False:
                x = augment_dataset(x,augmentation)
            
            self.train()
            print('Training Epoch {}/{}'.format(epoch,epochs))
            optimizer.zero_grad()
            
            forward = net(x)
            loss = loss_function(forward,y)
            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

            val_loss = self.validation(x_val, y_val, device, loss_function)
            running_val_loss.append(val_loss)
            
            print('Current Loss: ',loss.item())



        return net,running_loss,running_val_loss

    def predict(self,x,device):
        x = torch.tensor(x,dtype=torch.float32)
        if x.shape[1] != 3: 
            x = x.permute(0, 3, 1, 2)
        
        x=x.to(device)
        self.to(device)
        self.eval()
        predictions =self.forward(x)
        predictions = F.softmax(predictions, dim = 1)
        predictions_max = np.argmax(predictions.cpu().detach().numpy(),axis=1)
        predictions = predictions.cpu().detach().numpy()
        return predictions_max,predictions
    
    def validation(self,x_val,y_val,device,loss_function):
            x_val = torch.tensor(x_val, dtype=torch.float32).to(device)
            if x_val.shape[1] != 3: 
                x_val = x_val.permute(0, 3, 1, 2)
            
            y_val = torch.tensor(y_val, dtype=torch.long).to(device)

            # Forward pass for validation
            val_outputs = self.forward(x_val)
            val_loss = loss_function(val_outputs, y_val) 
            return val_loss.item()
    
    def calculate_accuracy_full(self,x_test,y_test,x_val,y_val,device='cpu'):
        with torch.no_grad():
            predictions_max,predictions = self.predict(x_val,device=device)
            correct = calculate_accuracy(predictions_max,y_val)
        print('Validation Accuracy ', correct )



        with torch.no_grad():
            predictions_max,predictions = self.predict(x_test,device=device)
            correct = calculate_accuracy(predictions_max,y_test)
        print('Test Accuracy ', correct )


        roc_auc = roc_auc_score(y_test,predictions,multi_class='ovr')
        print('ROC-AUC', roc_auc)

    # def imshow_bv(self,image, ax, title=None):
    
    #     image = image.permute(1, 2, 0).detach().numpy()
    #     ax.imshow(image)
    #     ax.axis('off')  
    #     if title:
    #         ax.set_title(title)           

    def visualize_output(self,x_test,y_test,device='cpu',display_num = 5):
        x_test.requires_grad_()
        predictions_max,predictions = self.predict(x_test,device=device)
        indices = torch.randperm(x_test.shape[0])[:display_num]
        images = x_test[indices]
        predicted_labels = predictions_max[indices]
        y_test = torch.tensor(y_test, dtype=torch.long, device=device)
        actual_labels = y_test[indices]

        output = self.forward(x_test)  # Assuming `self.model` is the trained model
        loss = torch.nn.functional.cross_entropy(output, y_test)
        self.zero_grad()
        loss.backward()
    
        gradients = x_test.grad.data
        saliency, _ = torch.max(gradients.abs(), dim=1)  

      

        fig, axes = plt.subplots(display_num, 2, figsize=(5, 15))
        
        for i in range(display_num):
            image = images[i].cpu()
            saliency_map = saliency[indices[i]].cpu().numpy()
            
            pred_label = str(predicted_labels[i].item())
            actual_label = str(actual_labels[i].item())
            title_string = f"Predicted: {pred_label}, Actual: {actual_label}"
            
            # Original image
            # self.imshow_bv(image, axes[0, i], title=title_string)
            axes[i,0].imshow(image[0,:,:].detach().numpy(),cmap='gray')
            axes[i,0].set_title(title_string)
            axes[i,0].axis('off')
            # Saliency map
            saliency_map = np.log1p(saliency_map)  # Log transform
           

            axes[i, 1].imshow(saliency_map, cmap="hot",vmin=0, vmax=1e-4)
            axes[i, 1].axis('off')
            axes[i, 1].set_title("Saliency Map")
        
        plt.tight_layout()
        plt.show()
