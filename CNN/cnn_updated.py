#!/usr/bin/env python
# coding: utf-8

# In[85]:


import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import torch

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
        for i in range(len(x_test)):
            y_pred = self.getoutputclass(torch.softmax(model(x_test[i].reshape(1,3,32,32)),dim=1))
            if y_pred == y_test[i]:
                tp = tp + 1
            else:
                fp = fp + 1
        return tp,fp        


# In[86]:


#if __name__ == "__main__":
cnnData = CNNData()
final_data = cnnData.store() 
cnnData.showTestImage(final_data)
X = final_data[b'data']
y = final_data[b'labels']
X_train, X_test, y_train, y_test = cnnData.splitting_data(X, y)
print(len(y))
print(X.shape)
#cnnData.checkAccuracies(X_train, X_test, y_train, y_test)



#For CNN
X, y = cnnData.convertDataForCNN(X, np.array(y))

X_train, X_test, y_train, y_test = cnnData.splitting_data(X, y)

#X_train = X_train.reshape(3,32,32)
#X_test = X_test.reshape(3,32,32)

X_train = cnnData.convertTensor(X_train,np.float32)
X_test = cnnData.convertTensor(X_test,np.float32)
y_train = cnnData.convertTensor(y_train,np.int64)
y_test = cnnData.convertTensor(y_test,np.int64)


#Creating a CNN Model



# In[118]:


torch.manual_seed(0)
C=3
num_filter = 20
filter_size = 2
model = torch.nn.Sequential(

    torch.nn.Conv2d(in_channels=C,
                    out_channels = num_filter,
                    kernel_size=filter_size,
                    padding=filter_size//2),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(kernel_size=filter_size,stride = 2),
    torch.nn.Flatten(),
    torch.nn.Linear(num_filter*16*16, (num_filter*16*16)*2),
    torch.nn.ReLU(),
    torch.nn.Linear((num_filter*16*16)*2, (num_filter*16*16)),
    torch.nn.ReLU(),
    #torch.nn.Linear((num_filter*16*16)*2, 256),
    #torch.nn.ReLU(),
    torch.nn.Linear((num_filter*16*16), 10),
)


# In[119]:


batch_size = 100
num_epoch = 5

# Your code to define loss function and optimizer here. Aim for 2 lines.
loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9,weight_decay=1e-3)
#weight_decay=1e-3


# In[120]:


for epoch in range(1, num_epoch+1):
    for i in range(0, len(X_train), batch_size):        
        X = X_train[i:i+batch_size].reshape(batch_size,3,32,32)
        y = y_train[i:i+batch_size]

        y_pred = model(X)
        l = loss(y_pred, y)
        
        model.zero_grad()
        l.backward()
        optimizer.step()
    print("Epoch %d final minibatch had loss %.4f" % (epoch, l.item()))    


# In[74]:



y_pred = torch.softmax(model(X_test[0].reshape(1,3,32,32)),dim=1)

print(getoutputclass(y_pred))
print(y_test[0])


# In[121]:


tp,fp = cnnData.calcAccuracy(X_test,y_test,model)


# In[122]:


print("True positives:",tp)
print("False positives:",fp)
print(len(X_test))
print(tp / len(X_test))


# In[ ]:




