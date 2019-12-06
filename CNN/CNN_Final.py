#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchvision.transforms as transforms


# In[32]:



transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
transform_test = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
])

mean = [0.5, 0.5, 0.5]
std  = [0.5, 0.5, 0.5]

class CNNData:
    
    def unpickle(self, file):
        
        with open(file,'rb') as fo:
            dictonary = pickle.load(fo, encoding='bytes')
        
        return dictonary
    
    def store(self):
        
        final_dictionary = {}
        
        
        for i in range(1,6):
            fileName = "data_batch_" + str(i)
            dictionary = self.unpickle(fileName)
            #print((dictionary[b'data']))
            if i == 1:
                final_dictionary = dictionary
            else:
                final_dictionary[b'labels'].extend(dictionary[b'labels'])
                final_dictionary[b'data'] = np.append(final_dictionary[b'data'],dictionary[b'data'], axis=0)
            
        del final_dictionary[b'batch_label']
        del final_dictionary[b'filenames']
                
        return final_dictionary
    
    def showTestImage(self,data):
        i=0
        for key, value in data.items():
            i += 1
            if i==2:
                plt.imshow(value[0].reshape(3,32,32).transpose(1,2,0))
                
    def modelaccuracies(self,data):
        
        models = []
        max_depths = [10,11,12,13]
        
        for i in max_depths:
            
            clf = DecisionTreeClassifier(random_state=0,max_depth=i)
            clf.fit(X_train, y_train)
            models.append(clf)
        
        return models

    def splitting_data(self,X,y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                            test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test
    
    def checkAccuracies(self,X_train, X_test, y_train, y_test):
        for i in models:
            training_score = i.score(X_train, y_train)
            test_score = i.score(X_test, y_test)
            print("training Accuracy " + str(training_score))
            print("test Accuracy " + str(test_score))
            
    def convertDataForCNN(self, X, y):
       
        X = X / 127.5
        X = X -1        
        return X, y
    def convertData(self, X):     
        X = X / 255     
        return X
    
    def convertTensor(self,data,dtype):
        return torch.tensor(data.astype(dtype))
    
    def getoutputclass(self,y):
        y_temp = y.max().detach().numpy()
        for i in range(len(y[0])):
            if y_temp == y[0][i].detach().numpy():
                return i

    def calcAccuracy(self,x_test,y_test,model):
        tp = 0
        fp = 0
        for i in range(len(x_test)-1):
            y_pred = self.getoutputclass(torch.softmax(model(x_test[i].reshape(1,3,32,32)),dim=1))
            if y_pred == y_test[i]:
                tp = tp + 1
            else:
                fp = fp + 1
        return tp,fp 
    def transformdata(self,data):
        X1 = data.reshape(3,32,32).transpose(1,2,0)
        X1 = transforms.ToPILImage()(X1)
        #.convert("RGB")
        X1 = transform_train(X1)
        X1 = X1.numpy()
        return X1
    def transformtestdata(self,data):
        X1 = data.reshape(3,32,32).transpose(1,2,0)
        X1 = transforms.ToPILImage()(X1)
        #.convert("RGB")
        X1 = transform_test(X1)
        X1 = X1.numpy()
        return X1


# In[144]:


cnnData = CNNData()
final_data = cnnData.store() 
#cnnData.showTestImage(final_data)
X = final_data[b'data']
y = final_data[b'labels']
X_train, X_test, y_train, y_test = cnnData.splitting_data(X, y)


# In[ ]:





# In[34]:




y_train = np.array(y_train)
y_test = np.array(y_test)


# In[146]:


X_aug_train = np.empty(shape=[80000, 3072])
y_aug_train = np.empty(shape=[80000, ])
X_aug_test = np.empty(shape=[10000, 3072])
j = len(X_train)
for i in range(len(X_train)):
    X_aug_train[i] = (cnnData.transformdata(X_train[i]).reshape(3072,))
    y_aug_train[i] = y_train[i]
    X_aug_train[j] = (cnnData.transformdata(X_train[i]).reshape(3072,))
    y_aug_train[j] = y_train[i]
    j = j + 1
for i in range(len(X_test)):
    X_aug_test[i] = (cnnData.transformtestdata(X_test[i]).reshape(3072,))
X_train = X_aug_train 
y_train = y_aug_train
X_test  = X_aug_test
X_train = cnnData.convertTensor(X_train,np.float32)
X_test = cnnData.convertTensor(X_test,np.float32)
y_train = cnnData.convertTensor(y_train,np.int64)
y_test = cnnData.convertTensor(y_test,np.int64)


# In[37]:


torch.manual_seed(0)
C=3
num_filter = 10
filter_size = 3
model = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=C,
                    out_channels = 10,
                    kernel_size=filter_size,
                    padding=filter_size//2),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2,stride = 2),
    torch.nn.Conv2d(in_channels=10,
                    out_channels = 20,
                    kernel_size=filter_size,
                    padding=filter_size//2),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2,stride = 2),  
    torch.nn.Conv2d(in_channels=20,
                    out_channels = 40,
                    kernel_size=filter_size,
                    padding=filter_size//2),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2,stride = 2),
    torch.nn.Conv2d(in_channels=40,
                    out_channels = 60,
                    kernel_size=filter_size,
                    padding=filter_size//2),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=2,stride = 2),
    torch.nn.Flatten(),
    torch.nn.Linear(60*2*2, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50,10),
)


# In[153]:
if torch.cuda.is_available():
    model = model.to('cuda')

batch_size = 100
num_epoch = 100

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#weight_decay=1e-3


# In[154]:


for epoch in range(1, num_epoch+1):
    for i in range(0, len(X_train), batch_size):        
        X = X_train[i:i+batch_size].reshape(batch_size,3,32,32)
        y = y_train[i:i+batch_size]
        if torch.cuda.is_available():
            X, y = X.to('cuda'), y.to('cuda')
        y_pred = model(X)
        l = loss(y_pred, y)
        
        model.zero_grad()
        l.backward()
        optimizer.step()
    print("Epoch %d final minibatch had loss %.4f" % (epoch, l.item()))    
    #tp,fp = cnnData.calcAccuracy(X_train,y_train,model)
    #print(tp / len(X_train))
    torch.save(model,'cnn_' + str(epoch+1))
    
tp,fp = cnnData.calcAccuracy(X_train.to('cuda'),y_train,model.to('cuda'))
print(tp / len(X_train))    
torch.save(model,'cnn_final')



m1 = torch.load("cnn_final",map_location=torch.device('cpu'))

# Traning Accuracy
tp,fp = cnnData.calcAccuracy(X_train,y_train,m1)
print("True positives:",tp)
print("False positives:",fp)
print("Train accuracy", tp / len(X_train))


#Test accuracy
tp,fp = cnnData.calcAccuracy(X_test,y_test,m1)
print("True positives:",tp)
print("False positives:",fp)
print(len(X_new_test))
print("Test accuracy", tp / len(X_new_test))


#Activation Maximization
#x = np.zeros([3,32,32])
x =torch.zeros((3,32,32),requires_grad=True,dtype=torch.float32)
#x = cnnData.convertTensor(x,np.float32)
y = np.array([9])
y = cnnData.convertTensor(y,np.int64)
x = x.reshape(1,3,32,32)
loss = torch.nn.CrossEntropyLoss()
for epoch in range(1, 100):
        y_pred = m1(x)
        x.retain_grad()
        model.zero_grad()
        l = loss(y_pred, y)        
        l.backward()
        x.data -= 0.2*x.grad.data
        x.grad.data.zero_()
    #print("Epoch %d final minibatch had loss %.4f" % (epoch, l.item()))    
plt.imshow(x.detach().numpy().reshape(3,32,32).transpose(1,2,0))   

