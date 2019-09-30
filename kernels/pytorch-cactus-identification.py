#!/usr/bin/env python
# coding: utf-8

# ### Overview
# To assess the impact of climate change on Earth's flora and fauna, it is vital to quantify how human activities
# such as logging, mining, and agriculture are impacting our protected natural areas. Researchers in Mexico have created
# the VIGIA project, which aims to build a system for autonomous surveillance of protected areas. A first step in such an
# effort is the ability to recognize the vegetation inside the protected areas. In this competition, you are tasked with
# creation of an algorithm that can identify a specific type of cactus in aerial imagery.
#
# <p align="center">
#     <img src="http://www.crwphoto.com/Cactus/CactusIMG_7764.jpg" width="600" height="auto">
# </p>

# ### Authors
#
# - Amitrajit Bose

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

from collections import OrderedDict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch import optim
from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.image as mpimg

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().system('ls ../input/train/train | wc -l')


# In[3]:


get_ipython().system('ls ../input/test/test | wc -l')


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[5]:


data_dir = '../input'
train_dir = data_dir + '/train/train/'
test_dir = data_dir + '/test/test/'


# In[6]:


labels = pd.read_csv('../input/train.csv')
labels.has_cactus.value_counts().plot.pie()
plt.show()

# dfdict = df.set_index('id')['has_cactus'].to_dict()


# ### Oversampling

# In[7]:


df1 = labels[labels.has_cactus == 0].copy()
df2 = df1.copy()
labels = labels.append([df1, df2], ignore_index=True)


# In[8]:


labels.has_cactus.value_counts().plot.pie()
plt.show()


# In[9]:


labels.head()


# In[10]:


labels.shape


# ### Splitting To Train & Validation Sets

# In[11]:


train_df, validation_df = train_test_split(
    labels, stratify=labels.has_cactus, test_size=0.1)
train_df.shape, validation_df.shape


# In[12]:


train_df.head()


# In[13]:


validation_df.head()


# In[14]:


labels['has_cactus'].value_counts().plot.barh(alpha=0.6, grid=True)
plt.legend()
plt.show()


# In[15]:


# plot 10 random images from the training dataset
fig = plt.figure(figsize=(25, 8))
train_imgs = os.listdir("../input/train/train")
for idx, img in enumerate(np.random.choice(train_imgs, 10)):
    ax = fig.add_subplot(2, 10//2, idx+1, xticks=[], yticks=[])
    im = mpimg.imread("../input/train/train/" + img)
    plt.imshow(im, cmap="hot")
    lab = labels.loc[labels['id'] == img, 'has_cactus'].values[0]
    ax.set_title(f'Label: {lab}')


# ### Custom Helper Class For Dataloading

# In[16]:


class CactusData(Dataset):
    def __init__(self, df, data_dir, transform):
        super().__init__()
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_name = self.df.id.iloc[index]
        label = self.df.has_cactus.iloc[index]

        img_path = os.path.join(self.data_dir, img_name)
        image = mpimg.imread(img_path)
        image = self.transform(image)
        return image, label


# In[17]:


epochs = 10
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ### Data Augmentation And Loading

# In[18]:


train_transf = transforms.Compose([transforms.ToPILImage(), transforms.RandomRotation(30), transforms.RandomResizedCrop(
    224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
val_transf = transforms.Compose([transforms.ToPILImage(), transforms.Resize(255), transforms.CenterCrop(
    224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# [transforms.ToPILImage(), transforms.ToTensor()]

train_data = CactusData(df=train_df, data_dir=train_dir,
                        transform=train_transf)
train_loader = DataLoader(
    dataset=train_data, batch_size=batch_size, shuffle=False)

val_data = CactusData(
    df=validation_df, data_dir=train_dir, transform=val_transf)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)

full_train_data = CactusData(
    df=labels, data_dir=train_dir, transform=train_transf)
full_train_loader = DataLoader(
    dataset=full_train_data, batch_size=batch_size, shuffle=False)


# ### Model Definition

# In[19]:


model = models.densenet161(pretrained=True)  # Loading pretrained DenseNet-161

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(2208, 1104)),
                          ('relu1', nn.ReLU()),
                          ('fc2', nn.Linear(1104, 512)),
                          ('relu2', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(512, 256)),
                          ('relu3', nn.ReLU()),
                          ('dropout2', nn.Dropout(0.1)),
                          ('fc4', nn.Linear(256, 2)),
                          ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier


# In[20]:


criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model = model.to(device)


# In[21]:


trainlosses = []
trainaccuracies = []
vallosses = []
valaccuracies = []
print("Default Device:", device)


# In[22]:


get_ipython().run_cell_magic('time', '',
                             "# Train model\nfor epoch in range(epochs):\n    for i, (images, labels) in enumerate(train_loader):\n        # Move inputs and labels to default device \n        images, labels = images.to(device), labels.to(device)\n        \n        # Forward\n        logps = model.forward(images)\n        loss = criterion(logps, labels)\n        \n        # Backward and optimize\n        optimizer.zero_grad()\n        loss.backward()\n        optimizer.step()\n        \n        if (i+1) % len(train_loader) == 0:\n            ps = torch.exp(logps)\n            top_p, top_class = ps.topk(1, dim=1)\n            equals = top_class == labels.view(*top_class.shape)\n            acc = torch.mean(equals.type(torch.FloatTensor)).item()\n            \n            trainlosses.append(loss.item())\n            trainaccuracies.append(acc)\n            \n            # Validation For The Corresponding Epoch\n            model.eval() # Turning model to evaluation mode\n            val_accuracy_epoch = 0 # Val accuracy for this epoch\n            val_loss_epoch = 0\n            with torch.no_grad():\n                for inputs, labels in val_loader:\n                    inputs, labels = inputs.to(device), labels.to(device)\n                    log_ps = model.forward(inputs)\n                    loss_val = criterion(log_ps, labels)\n                    ps_val = torch.exp(log_ps)\n                    top_p, top_class = ps_val.topk(1, dim=1)\n                    equals = top_class == labels.view(*top_class.shape)\n                    val_acc = torch.mean(equals.type(torch.FloatTensor)).item()\n                    val_loss_epoch += loss_val.item()\n                    val_accuracy_epoch += val_acc\n            vallosses.append(val_loss_epoch/len(val_loader))\n            valaccuracies.append(val_accuracy_epoch/len(val_loader))\n            model.train() # Turning model back to train mode\n            print ('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(epoch+1, epochs, loss.item(), acc, vallosses[-1], valaccuracies[-1]))")


# In[23]:


plt.plot([x for x in range(1, epochs+1)],
         trainaccuracies, label='Train Accuracy')
plt.plot([x for x in range(1, epochs+1)],
         valaccuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid()
plt.show()


# In[24]:


plt.plot([x for x in range(1, epochs+1)], trainlosses, label='Train Loss')
plt.plot([x for x in range(1, epochs+1)], vallosses, label='Validation Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid()
plt.show()


# ### Submission

# In[25]:


submit = pd.read_csv('../input/sample_submission.csv')
test_data = CactusData(df=submit, data_dir=test_dir, transform=val_transf)
test_loader = DataLoader(dataset=test_data, shuffle=False)


# In[26]:


predict = []
model.eval()
with torch.no_grad():
    for batch_i, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        output = model.forward(data)
        top_p, top_class = torch.exp(output).topk(1, dim=1)
        #print(top_p, top_class)
        if top_class.item() == 1:
            predict.append(top_p.item())  # probability of presence of cactus
        elif top_class.item() == 0:
            # probablity of presence of cactus
            predict.append(1 - top_p.item())

submit['has_cactus'] = predict
submit.to_csv('submission.csv', index=False)


# In[27]:


submit.head(10)
